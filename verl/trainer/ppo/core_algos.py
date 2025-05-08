# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_meta_rm_scores(
    log_probs_samples: torch.Tensor,
    response_mask: torch.Tensor,
    difficulty_coeff: torch.Tensor = None,
    agent_performance: torch.Tensor = None,
    curriculum_progress: float = 0.5,
    mode: str = "variance",
):
    """Compute meta-RM scores with curriculum progress adjustment."""
    with torch.no_grad():
        if mode == "variance":
            masked_log_probs = log_probs_samples * response_mask.unsqueeze(0)
            variance = torch.var(masked_log_probs, dim=-1, keepdim=True).squeeze(-1)
            scores = 1.0 / (1.0 + variance)
        elif mode == "entropy":
            probs = torch.softmax(log_probs_samples, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            entropy = (entropy * response_mask.unsqueeze(0)).sum(-1) / response_mask.sum(-1).unsqueeze(0)
            scores = 1.0 / (1.0 + entropy)
        
        # Adjust scores based on curriculum progress
        if agent_performance is not None:
            performance_factor = torch.sigmoid(agent_performance)
            scores = scores * (1 - curriculum_progress + curriculum_progress * performance_factor)
        
        if difficulty_coeff is not None:
            scores = scores * difficulty_coeff.view(1, -1)
        
        scores = torch.clamp(scores, 0.0, 1.0)
    return scores

def hierarchical_multi_gate_weights(
    scores: torch.Tensor,
    difficulty_coeff: torch.Tensor = None,
    curriculum_progress: float = 0.5,
    gate_factors: list = None,
    gate_weights: list = None,
):
    """Compute hierarchical multi-gate weights with curriculum progress."""
    if gate_factors is None:
        gate_factors = ["score", "difficulty", "progress"]
        gate_weights = [1.0, 0.5, 0.3]
    
    weights = torch.ones_like(scores)
    for factor, weight in zip(gate_factors, gate_weights):
        if factor == "score":
            weights = weights * scores ** weight
        elif factor == "difficulty" and difficulty_coeff is not None:
            weights = weights * (difficulty_coeff.view(1, -1) ** weight)
        elif factor == "progress":
            weights = weights * (1.0 + curriculum_progress) ** weight
    
    weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
    return weights

def generate_curriculum(
    task_pool: list,
    agent_performance: torch.Tensor,
    difficulty_coeffs: torch.Tensor,
    mode: str = "zpd",
    task_correlations: torch.Tensor = None,
) -> list:
    """Generate a curriculum of tasks based on agent performance and difficulty.
    
    Args:
        task_pool: (list) List of tasks (e.g., task IDs or configurations)
        agent_performance: (torch.Tensor) Shape: (num_tasks,) Agent's performance on each task
        difficulty_coeffs: (torch.Tensor) Shape: (num_tasks,) Difficulty coefficient for each task
        mode: (str) Curriculum generation mode ("zpd", "correlation")
        task_correlations: (torch.Tensor) Shape: (num_tasks, num_tasks) Task correlation matrix
    
    Returns:
        curriculum: (list) Ordered list of task indices
    """
    if mode == "zpd":
        # Select tasks where difficulty is slightly above current performance
        performance_threshold = agent_performance.mean() + agent_performance.std()
        valid_tasks = [
            i for i, diff in enumerate(difficulty_coeffs)
            if performance_threshold * 0.8 <= diff <= performance_threshold * 1.2
        ]
        curriculum = sorted(valid_tasks, key=lambda i: difficulty_coeffs[i])
    
    elif mode == "correlation" and task_correlations is not None:
        # Select tasks with high correlation to well-performing tasks
        well_performed = torch.topk(agent_performance, k=len(task_pool) // 2).indices
        correlation_scores = task_correlations[well_performed].mean(0)
        curriculum = torch.argsort(correlation_scores * difficulty_coeffs).tolist()
    
    else:
        raise ValueError(f"Invalid curriculum mode: {mode}")
    
    return [task_pool[i] for i in curriculum]


def distributed_compute_difficulty_coeff(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    task_features: dict = None,
    k_fold: int = 5,
    mode: str = "k_fold",
    dist_group=None,
):
    """Distributed version of compute_difficulty_coeff."""
    difficulty_coeff = compute_difficulty_coeff(
        log_probs, values, rewards, response_mask, task_features, k_fold, mode
    )
    
    if dist_group is not None and torch.distributed.is_initialized():
        torch.distributed.all_reduce(difficulty_coeff, op=torch.distributed.ReduceOp.SUM, group=dist_group)
        difficulty_coeff /= torch.distributed.get_world_size(dist_group)
    
    return difficulty_coeff

def compute_difficulty_coeff(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    task_features: dict = None,
    k_fold: int = 5,
    mode: str = "k_fold",
) -> torch.Tensor:
    """Compute dynamic difficulty coefficient based on agent performance or task features.
    
    Args:
        log_probs: (torch.Tensor) Shape: (batch_size, seq_len)
        values: (torch.Tensor) Shape: (batch_size, seq_len)
        rewards: (torch.Tensor) Shape: (batch_size, seq_len)
        response_mask: (torch.Tensor) Shape: (batch_size, seq_len)
        task_features: (dict) Task-specific features (e.g., seq_length, reward_sparsity)
        k_fold: (int) Number of folds for K-Fold cross-validation
        mode: (str) Difficulty computation mode ("k_fold", "zpd", "feature_based")
    
    Returns:
        difficulty_coeff: (torch.Tensor) Shape: (batch_size,)
    """
    with torch.no_grad():
        if mode == "k_fold":
            # Split samples into K folds
            batch_size = log_probs.shape[0]
            fold_size = batch_size // k_fold
            difficulty_scores = torch.zeros(batch_size, device=log_probs.device)
            
            for k in range(k_fold):
                # Train teacher model on K-1 folds
                train_indices = list(range(0, k * fold_size)) + list(range((k + 1) * fold_size, batch_size))
                test_indices = list(range(k * fold_size, (k + 1) * fold_size))
                
                # Simulate teacher model evaluation (replace with actual model training)
                teacher_loss = torch.mean((log_probs[test_indices] - log_probs[train_indices].mean(0)) ** 2, dim=-1)
                difficulty_scores[test_indices] = teacher_loss
            
            # Normalize to [0, 1]
            difficulty_coeff = (difficulty_scores - difficulty_scores.min()) / (
                difficulty_scores.max() - difficulty_scores.min() + 1e-8
            )
        
        elif mode == "zpd":
            # Compute performance metric (e.g., value error)
            value_error = torch.mean((values - rewards) ** 2 * response_mask, dim=-1)
            # Map to difficulty_coeff: higher error -> higher difficulty
            difficulty_coeff = torch.sigmoid(value_error)
        
        elif mode == "feature_based" and task_features:
            # Example: linear combination of sequence length and reward sparsity
            seq_length = torch.tensor(task_features.get("seq_length", 1.0), device=log_probs.device)
            reward_sparsity = torch.tensor(task_features.get("reward_sparsity", 1.0), device=log_probs.device)
            difficulty_coeff = 0.5 * seq_length / seq_length.max() + 0.5 * reward_sparsity
            difficulty_coeff = torch.clamp(difficulty_coeff, 0.0, 1.0)
        
        else:
            raise ValueError(f"Invalid difficulty mode: {mode}")
    
    return difficulty_coeff


def compute_weighted_log_probs(
    log_probs_samples: torch.Tensor,
    response_mask: torch.Tensor,
    difficulty_coeff: torch.Tensor = None,
    meta_rm_mode: str = "variance",
    threshold: float = 0.5,
    use_hierarchical_gating: bool = False,
    gate_factors: list = None,
    gate_weights: list = None,
):
    """Compute weighted average of log-prob samples using meta-RM scores and optional hierarchical gating.
    
    Args:
        log_probs_samples: (torch.Tensor) Shape: (num_samples, batch_size, seq_len)
        response_mask: (torch.Tensor) Shape: (batch_size, seq_len)
        difficulty_coeff: (torch.Tensor) Shape: (batch_size,) or None
        meta_rm_mode: (str) Meta-RM scoring mode ("variance" or "entropy")
        threshold: (float) Score threshold for filtering samples
        use_hierarchical_gating: (bool) Whether to use HoME-inspired hierarchical gating
        gate_factors: (list) Factors for hierarchical gating
        gate_weights: (list) Weights for gating factors
    
    Returns:
        log_probs: (torch.Tensor) Shape: (batch_size, seq_len)
    """
    # Compute meta-RM scores
    scores = compute_meta_rm_scores(
        log_probs_samples, response_mask, difficulty_coeff, mode=meta_rm_mode
    )

    # Filter samples with scores above threshold
    mask = scores > threshold
    filtered_log_probs = log_probs_samples[mask]

    # Fallback to mean if no samples pass
    if filtered_log_probs.numel() == 0:
        filtered_log_probs = log_probs_samples.mean(dim=0, keepdim=True)
        filtered_scores = torch.ones_like(scores[0:1])
    else:
        filtered_scores = scores[mask]

    # Compute weights
    if use_hierarchical_gating:
        weights = hierarchical_multi_gate_weights(
            filtered_scores, difficulty_coeff, gate_factors, gate_weights
        )
    else:
        weights = filtered_scores / (filtered_scores.sum(dim=0, keepdim=True) + 1e-8)

    # Weighted average of log-probs
    log_probs = torch.sum(filtered_log_probs * weights.view(-1, 1, 1), dim=0)
    return log_probs

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty: str, difficulty_coeff: torch.Tensor = None) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError
    if difficulty_coeff is not None:
        penalty = penalty * difficulty_coeff.unsqueeze(-1)
    return penalty   
    #raise NotImplementedError
