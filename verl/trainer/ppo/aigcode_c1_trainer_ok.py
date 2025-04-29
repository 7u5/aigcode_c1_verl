# verl/trainer/ppo/aigcode_c1_trainer.py
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, apply_kl_penalty
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    _timer,
    compute_advantage,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.tracking import Tracking


class AIGCodeC1Trainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        reward_fn: Callable = None,
        val_reward_fn: Callable = None,
        role_worker_mapping: Dict[str, List[Tuple[str, int]]] = None,
        processor=None,
        ray_worker_group_cls=None,
        resource_pool_manager: ResourcePoolManager = None,
    ):
        if role_worker_mapping is None:
            raise ValueError("role_worker_mapping cannot be None")
        if Role.ActorRollout not in role_worker_mapping:
            raise ValueError(f"role_worker_mapping must contain {Role.ActorRollout}")
        if Role.Critic not in role_worker_mapping:
            raise ValueError(f"role_worker_mapping must contain {Role.Critic}")

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            role_worker_mapping=role_worker_mapping,
            processor=processor,
            ray_worker_group_cls=ray_worker_group_cls,
            resource_pool_manager=resource_pool_manager,
        )

        self.use_vapo = config.algorithm.get("use_vapo", False)
        self.use_preference = config.algorithm.get("use_preference", False)
        self.norm_adv_by_std_in_grpo = config.algorithm.get("norm_adv_by_std_in_grpo", True)
        self.meta_lr = config.meta_learning.get("meta_lr", 1e-5)
        self.inner_lr = config.meta_learning.get("inner_lr", 1e-6)
        self.meta_steps = config.meta_learning.get("meta_steps", 3)
        self.inner_steps = config.meta_learning.get("inner_steps", 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_preference_loss(self, batch: DataProto) -> torch.Tensor:
        """Compute DAPO loss using pairwise preference data."""
        if not (self.use_preference and "chosen" in batch.non_tensor_batch and "rejected" in batch.non_tensor_batch):
            return torch.tensor(0.0, device=self.device)

        try:
            chosen_log_probs = self.actor_rollout_wg.compute_log_probs(batch, key="chosen")
            rejected_log_probs = self.actor_rollout_wg.compute_log_probs(batch, key="rejected")
            preference_loss = -torch.mean(chosen_log_probs - rejected_log_probs)
        except KeyError as e:
            print(f"Warning: Missing keys for preference loss computation: {e}")
            preference_loss = torch.tensor(0.0, device=self.device)

        return preference_loss.to(self.device)

    def _compute_value_aware_loss(self, batch: DataProto) -> Tuple[Dict[str, float], torch.Tensor]:
        """Compute VAPO loss with value function integration."""
        metrics = {}
        total_loss = torch.tensor(0.0, device=self.device)

        if self.use_vapo:
            try:
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
            except Exception as e:
                print(f"Warning: Failed to compute values for VAPO: {e}")
                values = None

        preference_loss = self._compute_preference_loss(batch)

        if not self.use_vapo:
            log_probs = self.actor_rollout_wg.compute_log_probs(batch)
            total_loss = -torch.mean(log_probs) if log_probs is not None else preference_loss
        else:
            batch = compute_advantage(
                batch,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                adv_estimator=self.config.algorithm.adv_estimator,
                norm_adv_by_std_in_grpo=self.norm_adv_by_std_in_grpo,
            )
            critic_output = self.critic_wg.update_critic(batch)
            critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_metrics)
            total_loss = preference_loss + critic_output.meta_info["loss"]

        metrics["preference_loss"] = preference_loss.item()
        metrics["total_loss"] = total_loss.item()
        return metrics, total_loss.to(self.device)

    def _inner_loop_adaptation(self, batch: DataProto) -> Dict[str, Any]:
        """Perform inner loop adaptation for meta-learning."""
        inner_params = [
            p.clone().detach().requires_grad_(True).to(self.device) for p in self.actor_rollout_wg.parameters()
        ]
        inner_optimizer = torch.optim.Adam(inner_params, lr=self.inner_lr)

        for _ in range(self.inner_steps):
            metrics, loss = self._compute_value_aware_loss(batch)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return {"inner_params": inner_params, "metrics": metrics}

    def fit(self):
        """Training loop with DAPO, VAPO, GRPO, and meta-learning."""
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
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        meta_optimizer = torch.optim.Adam(self.actor_rollout_wg.parameters(), lr=self.meta_lr)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Pop keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
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
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
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
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("inner_loop", timing_raw):
                        inner_result = self._inner_loop_adaptation(batch)
                        inner_params = inner_result["inner_params"]
                        inner_metrics = inner_result["metrics"]
                        metrics.update({f"inner/{k}": v for k, v in inner_metrics.items()})

                    with _timer("meta_loss", timing_raw):
                        val_batch_dict = next(iter(self.val_dataloader), None)
                        if val_batch_dict is None:
                            self.val_dataloader = iter(self.val_dataloader)
                            val_batch_dict = next(self.val_dataloader)
                        meta_batch: DataProto = DataProto.from_single_dict(val_batch_dict)

                        original_params = [p.clone() for p in self.actor_rollout_wg.parameters()]
                        for p, inner_p in zip(self.actor_rollout_wg.parameters(), inner_params):
                            p.data.copy_(inner_p.data)

                        meta_metrics, meta_loss = self._compute_value_aware_loss(meta_batch)
                        meta_optimizer.zero_grad()
                        meta_loss.backward()
                        meta_optimizer.step()

                        for p, orig_p in zip(self.actor_rollout_wg.parameters(), original_params):
                            p.data.copy_(orig_p.data)

                        metrics.update({f"meta/{k}": v for k, v in meta_metrics.items()})

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
                            norm_adv_by_std_in_grpo=self.norm_adv_by_std_in_grpo,
                        )

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
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
                    print(f"Final validation metrics: {val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
