import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Type, List, Tuple
import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import logging
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from verl.utils.megatron.optimizer import get_megatron_optimizer
from verl.utils.megatron_utils import init_megatron_optim_config
from verl.trainer.ppo.ray_trainer import _timer
from verl.utils.tracking import Tracking
from verl import DataProto
from .core_algos import agg_loss
from .metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from .reward import compute_reward, compute_reward_async
from .ray_trainer import AdvantageEstimator, apply_kl_penalty, compute_advantage, compute_response_mask

logger = logging.getLogger(__name__)

class AIGCodeC1Trainer(RayPPOTrainer):
    def get_model_and_optim(self, config, for_meta_learning: bool = False) -> Tuple[torch.optim.Optimizer, GPTModel]:
        """
        Get model and optimizer, reusing existing distributed context.
        """
        env_vars = {k: v for k, v in os.environ.items() if k in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES']}
        logger.info(f"get_model_and_optim: Environment variables: {env_vars}")
        logger.info(f"get_model_and_optim: torch.distributed initialized: {torch.distributed.is_initialized()}")
        logger.info(f"get_model_and_optim: Available GPUs: {torch.cuda.device_count()}")

        # Reuse existing distributed context (set by actor_rollout_wg)
        if not torch.distributed.is_initialized():
            backend = "gloo" if not torch.cuda.is_available() else "nccl"
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "8265")
            logger.info(f"Initializing process group: backend={backend}, rank={rank}, world_size={world_size}")
            torch.distributed.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=rank
            )
            if torch.cuda.is_available():
                torch.cuda.set_device(rank)
        else:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            backend = torch.distributed.get_backend()
            logger.info(f"Reusing process group: backend={backend}, rank={rank}, world_size={world_size}")

        # Initialize Megatron parallelism if not already done
        if not mpu.is_initialized():
            logger.info("Initializing Megatron model parallel")
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=config.actor_rollout_ref.actor.megatron.get("virtual_pipeline_model_parallel_size", None),
                context_parallel_size=config.actor_rollout_ref.actor.megatron.get("context_parallel_size", 1),
                use_sharp=False,
                nccl_communicator_config_path=None,
            )

        # Get model configuration
        transformer_config = self.get_model_config()

        # Create model
        if for_meta_learning:
            logger.info("Creating meta-model for meta-learning")
            meta_model = GPTModel(transformer_config)
            # Load state from actor_rollout_wg if available
            try:
                global_state_dict = aggregate_state_dicts(self.actor_rollout_wg.actors, config=transformer_config)
                meta_model.load_state_dict(global_state_dict)
                logger.info("Loaded actor state into meta-model")
            except Exception as e:
                logger.warning(f"Failed to load actor state into meta-model: {e}, using initialized model")
        else:
            logger.info("Creating actor model")
            meta_model = GPTModel(transformer_config)
            global_state_dict = aggregate_state_dicts(self.actor_rollout_wg.actors, config=transformer_config)
            meta_model.load_state_dict(global_state_dict)

        # Handle pipeline parallelism
        if isinstance(meta_model, list):
            meta_model = meta_model[0]

        # Initialize optimizer
        optim_config = init_megatron_optim_config(config.actor_rollout_ref.actor.optim)
        optimizer = get_megatron_optimizer(model=meta_model, config=optim_config)
        logger.info("Optimizer initialized")

        return optimizer, meta_model

    def get_combined_optimizer(self, actor_model: GPTModel, meta_model: GPTModel, config) -> torch.optim.Optimizer:
        """
        Create a single optimizer for both actor and meta-model parameters.
        """
        optim_config = init_megatron_optim_config(config.actor_rollout_ref.actor.optim)
        
        # Collect parameters from both models
        actor_params = [{"params": [p for p in actor_model.parameters() if p.requires_grad], "name": "actor"}]
        meta_params = [{"params": [p for p in meta_model.parameters() if p.requires_grad], "name": "meta"}]
        combined_params = actor_params + meta_params

        logger.info(f"Combined optimizer: {len(actor_params[0]['params'])} actor params, {len(meta_params[0]['params'])} meta params")
        
        # Create optimizer
        optimizer = get_megatron_optimizer(model=None, config=optim_config, params=combined_params)
        return optimizer

    def _inner_loop_adaptation(self, batch: 'DataProto', optimizer: torch.optim.Optimizer, model: GPTModel) -> Dict:
        """
        Perform inner-loop adaptation on the actor model.
        """
        model.train()
        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            loss = 0
            if self.use_preference:
                loss += self._compute_preference_loss(batch)
            if self.use_vapo:
                loss += self._compute_value_aware_loss(batch)
            else:
                log_probs_samples = []
                for _ in range(self.num_samples):
                    result = self.actor_rollout_wg.compute_log_prob(batch)
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
        
        return {"parameters": [p.detach() for p in model.parameters()]}

    def fit(self):
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
            logger.info(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        logger.info(f"actor_rollout_wg type: {type(self.actor_rollout_wg)}")
        logger.info(f"actor_rollout_wg attributes: {dir(self.actor_rollout_wg)}")
        
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        # Initialize models after actor_rollout_wg distributed setup
        logger.info("Initializing actor and meta-model after worker distributed init")
        actor_optimizer, actor_model = self.get_model_and_optim(self.config.actor_rollout_ref, for_meta_learning=False)
        _, meta_model = self.get_model_and_optim(self.config.actor_rollout_ref, for_meta_learning=True)
        
        # Create combined optimizer for both models
        logger.info("Creating combined optimizer for actor and meta-model")
        combined_optimizer = self.get_combined_optimizer(actor_model, meta_model, self.config)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch = DataProto.from_single_dict(batch_dict)

                keys_to_pop = (
                    (["input_ids", "attention_mask", "position_ids"], ["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"])
                    if "multi_modal_inputs" in batch.non_tensor_batch
                    else (["input_ids", "attention_mask", "position_ids"], ["raw_prompt_ids"])
                )
                gen_batch = batch.pop(batch_keys=keys_to_pop[0], non_tensor_batch_keys=keys_to_pop[1])

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
                            reward_tensor = ray.get(self.rm_wg.compute_rm_score.remote(batch))
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
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        with _timer("meta_update", timing_raw):
                            combined_optimizer.zero_grad()
                            for _ in range(self.meta_steps):
                                # Inner-loop adaptation on actor model
                                adapted_state = self._inner_loop_adaptation(batch, combined_optimizer, actor_model)
                                # Update actor_rollout_wg with adapted parameters
                                self._set_worker_parameters(self.actor_rollout_wg, adapted_state["parameters"])
                                
                                # Compute meta-loss (on meta-model)
                                meta_model.train()
                                meta_loss = (
                                    self._compute_value_aware_loss(batch)
                                    if self.use_vapo
                                    else -torch.mean(
                                        self.actor_rollout_wg.compute_log_prob(batch)["log_probs"]
                                        * batch.batch["advantages"]
                                    )
                                )
                                meta_loss.backward()
                            
                            # Update both models' parameters
                            combined_optimizer.step()
                            # Sync actor parameters to actor_rollout_wg
                            actor_params = [p for p in actor_model.parameters()]
                            self._set_worker_parameters(self.actor_rollout_wg, actor_params)

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
                    logger.info(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1