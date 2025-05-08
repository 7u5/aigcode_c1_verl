# Copyright 2024 AIGCode Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller, enhanced with meta-learning, DAPO, and VAPO.
This trainer supports model-agnostic model initialization with HuggingFace.
"""

import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, List, Tuple, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role, AdvantageEstimator, ResourcePoolManager, apply_kl_penalty, _timer, compute_response_mask, compute_advantage, WorkerType
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

class AIGCodeC1Trainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, ray.actor.ActorClass],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        self.meta_lr = config.meta_learning.meta_lr
        self.inner_lr = config.meta_learning.inner_lr
        self.meta_steps = config.meta_learning.meta_steps
        self.inner_steps = config.meta_learning.inner_steps
        self.use_vapo = config.algorithm.use_vapo
        self.use_preference = config.algorithm.use_preference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if not ray.is_initialized():
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            ray.init(
                num_gpus=num_gpus,
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    }
                },
            )
            print(f"Ray initialized with {num_gpus} GPUs")

        print("Role worker mapping:", role_worker_mapping)
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = (
            config.actor_rollout_ref.ref.log_prob_micro_batch_size
            or config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu
        )
        self.use_rm = config.reward_model.enable
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self.use_critic = config.algorithm.adv_estimator == AdvantageEstimator.GAE
        if not self.use_critic and config.algorithm.adv_estimator not in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            raise NotImplementedError(f"Unsupported advantage estimator: {config.algorithm.adv_estimator}")

        self._validate_config()
        self._create_dataloader()

    def _meta_rm_score(self, log_probs: torch.Tensor, batch: 'DataProto') -> torch.Tensor:
        """Placeholder for meta RM: scores sample quality based on log-prob variance."""
        # Simulate meta RM by scoring samples based on stability (e.g., low variance)
        variance = torch.var(log_probs, dim=-1, keepdim=True)
        scores = 1.0 / (1.0 + variance)  # Higher score for lower variance
        return torch.clamp(scores, 0.0, 1.0)

    def _compute_preference_loss(self, batch: 'DataProto') -> torch.Tensor:
        if not self.use_preference:
            return torch.tensor(0.0, device=self.device)

        pos_batch, neg_batch = batch.split_pairs()
        if pos_batch is None or neg_batch is None:
            return torch.tensor(0.0, device=self.device)

        # Generate multiple samples for positive and negative batches
        pos_log_probs_samples = []
        neg_log_probs_samples = []
        for _ in range(self.num_samples):
            pos_result = ray.get(self.actor_rollout_wg.compute_log_prob.remote(pos_batch))
            neg_result = ray.get(self.actor_rollout_wg.compute_log_prob.remote(neg_batch))
            pos_log_probs_samples.append(pos_result["log_probs"])
            neg_log_probs_samples.append(neg_result["log_probs"])

        # Stack samples: [num_samples, batch_size]
        pos_log_probs_samples = torch.stack(pos_log_probs_samples, dim=0)
        neg_log_probs_samples = torch.stack(neg_log_probs_samples, dim=0)

        # Compute meta RM scores for each sample
        pos_scores = self._meta_rm_score(pos_log_probs_samples, pos_batch)
        neg_scores = self._meta_rm_score(neg_log_probs_samples, neg_batch)

        # Filter samples: keep those with scores > threshold (e.g., 0.5)
        threshold = 0.5
        pos_mask = pos_scores > threshold
        neg_mask = neg_scores > threshold
        pos_log_probs_filtered = pos_log_probs_samples[pos_mask]
        neg_log_probs_filtered = neg_log_probs_samples[neg_mask]

        # If no samples pass the threshold, use mean as fallback
        if pos_log_probs_filtered.numel() == 0:
            pos_log_probs_filtered = pos_log_probs_samples.mean(dim=0, keepdim=True)
            pos_scores = torch.ones_like(pos_scores[0:1])
        if neg_log_probs_filtered.numel() == 0:
            neg_log_probs_filtered = neg_log_probs_samples.mean(dim=0, keepdim=True)
            neg_scores = torch.ones_like(neg_scores[0:1])

        # Voting: Weighted average based on meta RM scores
        pos_weights = pos_scores[pos_mask] / pos_scores[pos_mask].sum()
        neg_weights = neg_scores[neg_mask] / neg_scores[neg_mask].sum()
        pos_log_probs = torch.sum(pos_log_probs_filtered * pos_weights.view(-1, 1), dim=0)
        neg_log_probs = torch.sum(neg_log_probs_filtered * neg_weights.view(-1, 1), dim=0)

        # Apply difficulty coefficient (delta) from batch
        delta = batch.batch.get("difficulty_coeff", torch.tensor(1.0, device=self.device))
        logits = delta * (pos_log_probs - neg_log_probs)
        logits = torch.clamp(logits, -10, 10)
        loss = -torch.mean(torch.log(torch.sigmoid(logits)))
        return loss

    def _compute_value_aware_loss(self, batch: 'DataProto') -> torch.Tensor:
        if not self.use_vapo:
            return torch.tensor(0.0, device=self.device)

        # Compute values
        values = ray.get(self.critic_wg.compute_values.remote(batch))
        values = (values - values.mean()) / (values.std() + 1e-8)
        advantages = batch.batch["advantages"]

        # Generate multiple samples for log-probabilities
        log_probs_samples = []
        for _ in range(self.num_samples):
            result = ray.get(self.actor_rollout_wg.compute_log_prob.remote(batch))
            log_probs_samples.append(result["log_probs"])

        # Stack samples: [num_samples, batch_size]
        log_probs_samples = torch.stack(log_probs_samples, dim=0)

        # Compute meta RM scores
        scores = self._meta_rm_score(log_probs_samples, batch)

        # Filter samples
        mask = scores > 0.5
        log_probs_filtered = log_probs_samples[mask]

        # Fallback to mean if no samples pass
        if log_probs_filtered.numel() == 0:
            log_probs_filtered = log_probs_samples.mean(dim=0, keepdim=True)
            scores = torch.ones_like(scores[0:1])

        # Voting: Weighted average
        weights = scores[mask] / scores[mask].sum()
        log_probs = torch.sum(log_probs_filtered * weights.view(-1, 1), dim=0)

        # Apply difficulty coefficient
        delta = batch.batch.get("difficulty_coeff", torch.tensor(1.0, device=self.device))
        value_aware_loss = -torch.mean(delta * log_probs * advantages * values)
        return value_aware_loss

    def _inner_loop_adaptation(self, batch: 'DataProto') -> dict:
        try:
            model_state = ray.get(self.actor_rollout_wg.get_model_state.remote())
        except AttributeError as e:
            raise RuntimeError("Actor worker does not support get_model_state") from e

        optimizer = torch.optim.Adam(model_state["parameters"], lr=self.inner_lr)

        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            loss = 0
            if self.use_preference:
                loss += self._compute_preference_loss(batch)
            if self.use_vapo:
                loss += self._compute_value_aware_loss(batch)
            else:
                # Generate multiple samples for default loss
                log_probs_samples = []
                for _ in range(self.num_samples):
                    result = ray.get(self.actor_rollout_wg.compute_log_prob.remote(batch))
                    log_probs_samples.append(result["log_probs"])

                log_probs_samples = torch.stack(log_probs_samples, dim=0)
                scores = self._meta_rm_score(log_probs_samples, batch)
                mask = scores > 0.5
                log_probs_filtered = log_probs_samples[mask]

                if log_probs_filtered.numel() == 0:
                    log_probs_filtered = log_probs_samples.mean(dim=0, keepdim=True)
                    scores = torch.ones_like(scores[0:1])

                weights = scores[mask] / scores[mask].sum()
                log_probs = torch.sum(log_probs_filtered * weights.view(-1, 1), dim=0)

                advantages = batch.batch["advantages"]
                delta = batch.batch.get("difficulty_coeff", torch.tensor(1.0, device=self.device))
                loss += -torch.mean(delta * log_probs * advantages)

            loss.backward()
            optimizer.step()

        return model_state
    
    def _get_worker_parameters(self, worker):
        try:
            params = ray.get(worker.get_model_parameters.remote())
            return params
        except AttributeError as e:
            raise RuntimeError(f"Worker {worker} does not support get_model_parameters") from e

    def _set_worker_parameters(self, worker, params):
        try:
            ray.get(worker.set_model_parameters.remote(params))
        except AttributeError as e:
            raise RuntimeError(f"Worker {worker} does not support set_model_parameters") from e

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }
            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"
                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )
                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )
            if self.use_reference_policy:
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward KL and KL loss.")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated. "
                "Validation datasets are sent to inference engines as a whole batch."
            )

        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "Validation gen temperature should be greater than 0 when enabling do_sample"
            )

        #if self.use_preference and not hasattr(self.train_dataset, "split_pairs"):
        #    raise ValueError("Dataset must support split_pairs for preference optimization (DAPO).")

        print("[validate_config] All configuration checks passed successfully!")

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        meta_optimizer = torch.optim.Adam(self._get_worker_parameters(self.actor_rollout_wg), lr=self.meta_lr)

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch = DataProto.from_single_dict(batch_dict)

                keys_to_pop = (
                    (
                        ["input_ids", "attention_mask", "position_ids"],
                        ["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                    if "multi_modal_inputs" in batch.non_tensor_batch
                    else (["input_ids", "attention_mask", "position_ids"], ["raw_prompt_ids"])
                )
                gen_batch = batch.pop(batch_keys=keys_to_pop[0], non_tensor_batch_keys=keys_to_pop[1])

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = ray.get(self.actor_rollout_wg.generate_sequences.remote(gen_batch))

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = ray.get(self.actor_rollout_wg.generate_sequences.remote(gen_baseline_batch))
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = ray.get(self.rm_wg.compute_rm_score.remote(batch))
                            batch = batch.union(reward_tensor)
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = ray.get(self.actor_rollout_wg.compute_log_prob.remote(batch))
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = ray.get(self.ref_policy_wg.compute_ref_log_prob.remote(batch))
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = ray.get(self.critic_wg.compute_values.remote(batch))
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        )

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = ray.get(self.critic_wg.update_critic.remote(batch))
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            actor_output = ray.get(self.actor_rollout_wg.update_actor.remote(batch))
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        with _timer("meta_update", timing_raw):
                            meta_optimizer.zero_grad()
                            for _ in range(self.meta_steps):
                                adapted_state = self._inner_loop_adaptation(batch)
                                self._set_worker_parameters(self.actor_rollout_wg, adapted_state["parameters"])
                                meta_loss = (
                                    self._compute_value_aware_loss(batch)
                                    if self.use_vapo
                                    else -torch.mean(
                                        ray.get(self.actor_rollout_wg.compute_log_prob.remote(batch))["log_probs"]
                                        * batch.batch["advantages"]
                                    )
                                )
                                meta_loss.backward()
                            meta_optimizer.step()
                            self._set_worker_parameters(self.actor_rollout_wg, meta_optimizer.param_groups[0]["params"])

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1