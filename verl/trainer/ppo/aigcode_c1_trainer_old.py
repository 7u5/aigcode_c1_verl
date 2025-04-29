# verl/trainer/ppo/aigcode_c1_trainer.py
"""
DAPO + VAPO + Meta-Learning Trainer for RLHF
"""

from typing import Any, Callable, Dict

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_advantage
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tracking import Tracking


class AIGCodeC1Trainer(RayPPOTrainer):
    def __init__(self, config, reward_fn: Callable = None, val_reward_fn: Callable = None):
        super().__init__(config, reward_fn, val_reward_fn)
        self.meta_lr = config.meta_learning.meta_lr  # Outer loop learning rate for meta-learning
        self.inner_lr = config.meta_learning.inner_lr  # Inner loop learning rate for adaptation
        self.meta_steps = config.meta_learning.meta_steps  # Number of meta-training steps
        self.use_vapo = config.algorithm.use_vapo  # Enable VAPO (value-aware updates)

    def _compute_preference_loss(self, batch: DataProto) -> torch.Tensor:
        """
        Compute DAPO loss using pairwise preference data.
        Assumes batch contains 'chosen' and 'rejected' responses.
        """
        chosen_log_probs = self.actor_rollout_wg.compute_log_probs(batch, key="chosen")
        rejected_log_probs = self.actor_rollout_wg.compute_log_probs(batch, key="rejected")
        # DAPO loss: maximize the difference between chosen and rejected log probs
        preference_loss = -torch.mean(chosen_log_probs - rejected_log_probs)
        return preference_loss

    def _compute_value_aware_loss(self, batch: DataProto) -> Dict[str, torch.Tensor]:
        """
        Compute VAPO loss by incorporating value function into DAPO.
        """
        metrics = {}
        values = self.critic_wg.compute_values(batch)
        batch = batch.union(values)
        preference_loss = self._compute_preference_loss(batch)

        if self.use_vapo:
            # Compute advantages using the critic's value function
            batch = compute_advantage(
                batch,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                adv_estimator=self.config.algorithm.adv_estimator,
            )
            # Update critic to ensure value awareness
            critic_output = self.critic_wg.update_critic(batch)
            critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_metrics)
            # Combine preference loss with value-aware updates
            total_loss = preference_loss + critic_output.meta_info["loss"]
        else:
            total_loss = preference_loss

        metrics["preference_loss"] = preference_loss.item()
        metrics["total_loss"] = total_loss.item()
        return metrics, total_loss

    def _inner_loop_adaptation(self, batch: DataProto) -> Dict[str, Any]:
        """
        Perform inner loop adaptation for meta-learning.
        """
        # Clone model parameters for inner loop
        inner_params = [p.clone().detach().requires_grad_(True) for p in self.actor_rollout_wg.parameters()]
        inner_optimizer = torch.optim.Adam(inner_params, lr=self.inner_lr)

        # Inner loop update
        for _ in range(self.config.meta_learning.inner_steps):
            metrics, loss = self._compute_value_aware_loss(batch)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return {"inner_params": inner_params, "metrics": metrics}

    def fit(self):
        """
        Training loop with DAPO, VAPO, and meta-learning.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

        # Meta-training loop
        meta_optimizer = torch.optim.Adam(self.actor_rollout_wg.parameters(), lr=self.meta_lr)
        for meta_step in range(self.meta_steps):
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Inner loop adaptation (fast adaptation on a batch)
                inner_result = self._inner_loop_adaptation(batch)
                inner_params = inner_result["inner_params"]
                metrics = inner_result["metrics"]

                # Outer loop: Compute meta-loss on a new batch
                meta_batch_dict = next(iter(self.val_dataloader))
                meta_batch: DataProto = DataProto.from_single_dict(meta_batch_dict)

                # Temporarily set model parameters to inner loop parameters
                original_params = [p.clone() for p in self.actor_rollout_wg.parameters()]
                for p, inner_p in zip(self.actor_rollout_wg.parameters(), inner_params):
                    p.data.copy_(inner_p.data)

                # Compute meta-loss
                meta_metrics, meta_loss = self._compute_value_aware_loss(meta_batch)

                # Restore original parameters and compute gradients for meta-update
                for p, orig_p in zip(self.actor_rollout_wg.parameters(), original_params):
                    p.data.copy_(orig_p.data)

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

                # Update metrics
                metrics.update({f"meta/{k}": v for k, v in meta_metrics.items()})
                metrics["meta_step"] = meta_step

                # Log metrics
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1

                # Validate periodically
                if self.val_reward_fn is not None and (self.global_steps + 1) % self.config.trainer.test_freq == 0:
                    val_metrics = self._validate()
                    val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                    metrics.update(val_metrics)
                    logger.log(data=val_metrics, step=self.global_steps)

                # Save checkpoint periodically
                if (self.global_steps + 1) % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()
