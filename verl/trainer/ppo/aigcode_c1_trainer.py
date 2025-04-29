# verl/trainer/ppo/aigcode_c1_trainer.py
from copy import deepcopy

import numpy as np
import ray
import torch
from tqdm import tqdm
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, _timer, apply_kl_penalty, compute_reward, compute_reward_async, AdvantageEstimator, compute_response_mask, uuid
#from verl.single_controller.ray.actor_rollout_worker import ActorRolloutWorker

class AIGCodeC1Trainer(RayPPOTrainer):
    def __init__(self, config, **kwargs):
        if "role_worker_mapping" in kwargs:
            if config.actor_rollout_ref.actor.strategy == "fsdp":
                from verl.workers.fsdp_workers import ActorRolloutRefWorker
            elif config.actor_rollout_ref.actor.strategy == "megatron":
                from verl.workers.megatron_workers import ActorRolloutRefWorker
            kwargs["role_worker_mapping"][Role.ActorRollout] = ActorRolloutRefWorker
        super().__init__(config, **kwargs)
        self.meta_lr = config.meta_learning.meta_lr
        self.inner_lr = config.meta_learning.inner_lr
        self.meta_steps = config.meta_learning.meta_steps
        self.inner_steps = config.meta_learning.inner_steps
        self.use_vapo = config.algorithm.use_vapo
        self.use_preference = config.algorithm.use_preference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _compute_preference_loss(self, batch: DataProto, model):
        # DAPO: Compute preference loss based on paired data
        pos_batch, neg_batch = batch.split_pairs()
        pos_log_probs = model.compute_log_prob(pos_batch)
        neg_log_probs = model.compute_log_prob(neg_batch)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_log_probs - neg_log_probs)))
        return loss

    def _compute_value_aware_loss(self, batch: DataProto, model):
        # VAPO: Compute value-aware loss
        values = self.critic_wg.compute_values(batch)
        advantages = batch.batch["advantages"]
        log_probs = model.compute_log_prob(batch)
        value_aware_loss = -torch.mean(log_probs * advantages * values)
        return value_aware_loss

    def _inner_loop_adaptation(self, batch: DataProto, model_ref):
        # Inner loop for meta-learning
        model = deepcopy(model_ref)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            loss = 0
            if self.use_preference:
                loss += self._compute_preference_loss(batch, model)
            if self.use_vapo:
                loss += self._compute_value_aware_loss(batch, model)
            else:
                log_probs = model.compute_log_prob(batch)
                advantages = batch.batch["advantages"]
                loss += -torch.mean(log_probs * advantages)
            loss.backward()
            optimizer.step()

        return model

    def _get_worker_parameters(self, worker):
        print(f"Worker type: {type(worker)}")
        print(f"Worker methods: {dir(worker)}")
        print(f"get_model_parameters type: {type(worker.get_model_parameters)}")
        try:
            print(f"get_model_parameters __self__: {worker.get_model_parameters.__self__}")
            print(f"get_model_parameters __qualname__: {worker.get_model_parameters.__qualname__}")
            print(f"get_model_parameters has remote: {hasattr(worker.get_model_parameters, 'remote')}")
        except AttributeError as e:
            print(f"Error inspecting get_model_parameters: {e}")
        if hasattr(worker, 'workers'):
            for i, w in enumerate(worker.workers):
                print(f"Worker {i} type: {type(w)}")
                print(f"Worker {i} methods: {dir(w)}")
                try:
                    if hasattr(w, 'get_model_parameters'):
                        method_type = type(w.get_model_parameters)
                        print(f"Worker {i} get_model_parameters type: {method_type}")
                        print(f"Worker {i} get_model_parameters __self__: {w.get_model_parameters.__self__}")
                        print(f"Worker {i} get_model_parameters __qualname__: {w.get_model_parameters.__qualname__}")
                        print(f"Worker {i} get_model_parameters has remote: {hasattr(w.get_model_parameters, 'remote')}")
                    else:
                        print(f"Worker {i} get_model_parameters: Not found")
                except Exception as e:
                    print(f"Worker {i} get_model_parameters error: {e}")
        try:
            print(f"Attempting to call worker.get_model_parameters.remote()")
            worker_params = ray.get(worker.get_model_parameters)
        except AttributeError as e:
            print(f"Error calling get_model_parameters.remote(): {e}")
            raise
        return worker_params

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
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        try:
            actor_parameters = self._get_worker_parameters(self.actor_rollout_wg)
        except Exception as e:
            print(f"Error in _get_worker_parameters: {e}")
            raise
        
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        # Get parameters for meta-optimizer
        #actor_parameters = self._get_worker_parameters(self.actor_rollout_wg)

        meta_optimizer = torch.optim.Adam(actor_parameters, lr=self.meta_lr)

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
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("meta_update", timing_raw):
                            meta_optimizer.zero_grad()
                            # Perform meta-learning
                            for _ in range(self.meta_steps):
                                adapted_model = self._inner_loop_adaptation(batch, self.actor_rollout_wg)
                                meta_loss = (
                                    self._compute_value_aware_loss(batch, adapted_model)
                                    if self.use_vapo
                                    else -torch.mean(adapted_model.compute_log_prob(batch) * batch.batch["advantages"])
                                )
                                meta_loss.backward()
                            meta_optimizer.step()

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
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
