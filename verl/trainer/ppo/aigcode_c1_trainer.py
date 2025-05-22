# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

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

from verl.utils.megatron.optimizer import get_megatron_optimizer
from verl.utils.megatron_utils import get_model, init_megatron_optim_config, mcore_model_parallel_config
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, WorkerType, Role, AdvantageEstimator, ResourcePoolManager, apply_kl_penalty, compute_advantage, compute_response_mask, _timer
from verl.models.mcore import hf_to_mcore_config
import torch.optim as optim
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt.gpt_model import ModelType

from verl.utils.megatron_utils import unwrap_model
from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel, _megatron_calc_layer_map
from transformers import AutoModelForCausalLM, AutoConfig

import logging
logger = logging.getLogger(__name__)


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



def gather_tp_shards(tensor_name, state_dict, config, tp_size):
    from torch.distributed import all_gather
    tp_rank = mpu.get_tensor_model_parallel_rank()
    full_tensor = state_dict[tensor_name]
    tensor_chunks = torch.chunk(full_tensor, tp_size, dim=0)
    gathered_chunks = [torch.empty_like(tensor_chunks[0]) for _ in range(tp_size)]
    all_gather(gathered_chunks, tensor_chunks[tp_rank], group=mpu.get_tensor_model_parallel_group())
    return torch.cat(gathered_chunks, dim=0)

def aggregate_state_dicts(actors, config = None):
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    state_dicts = [ray.get(actor.get_state_dict.remote()) for actor in actors]
    global_state_dict = {}
    if pp_rank == 0:
        global_state_dict["model.embed_tokens.weight"] = gather_tp_shards(
            "model.embed_tokens.weight", state_dicts[0], config, mpu.get_tensor_model_parallel_world_size()
        )
    layer_map = _megatron_calc_layer_map(config)
    for layer in range(config.num_hidden_layers):
        layer_name = f"model.layers.{layer}"
        dst_pp_rank, dst_virtual_pp_rank, dst_layer_idx = layer_map[layer]
        if pp_rank == dst_pp_rank:
            state_dict = state_dicts[dst_virtual_pp_rank]
            for param_name in [
                f"{layer_name}.input_layernorm.weight",
                f"{layer_name}.self_attn.q_proj.weight",
                f"{layer_name}.self_attn.k_proj.weight",
                f"{layer_name}.self_attn.v_proj.weight",
                f"{layer_name}.self_attn.o_proj.weight",
                f"{layer_name}.post_attention_layernorm.weight",
                f"{layer_name}.mlp.gate_proj.weight",
                f"{layer_name}.mlp.up_proj.weight",
                f"{layer_name}.mlp.down_proj.weight",
            ]:
                global_state_dict[param_name] = gather_tp_shards(
                    param_name, state_dict, config, mpu.get_tensor_model_parallel_world_size()
                )
    if pp_rank == mpu.get_pipeline_model_parallel_world_size() - 1:
        global_state_dict["model.norm.weight"] = gather_tp_shards(
            "model.norm.weight", state_dicts[-1], config, mpu.get_tensor_model_parallel_world_size()
        )
        global_state_dict["lm_head.weight"] = gather_tp_shards(
            "lm_head.weight", state_dicts[-1], config, mpu.get_tensor_model_parallel_world_size()
        )
    return global_state_dict


class AIGCodeC1Trainer(RayPPOTrainer):
         
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
        difficulty_coeff = compute_difficulty_coeff(
            log_probs=pos_log_probs_samples.mean(0),
            values=ray.get(self.critic_wg.compute_values.remote(pos_batch)),
            rewards=pos_batch.batch["rewards"],
            response_mask=pos_batch.batch["response_mask"],
            task_features={"seq_length": pos_batch.batch["seq_length"], "reward_sparsity": 0.5},
            mode="k_fold"
        )
        batch.batch.set("difficulty_coeff", torch.tensor(difficulty_coeff, device=self.device))
        #delta = batch.batch.get("difficulty_coeff", torch.tensor(1.0, device=self.device))
        delta = difficulty_coeff
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

    def get_model_and_optim(self, config):
        from megatron.core import mpu
        import torch
        from megatron.core.models.gpt import GPTModel
        from verl.utils.megatron_utils import init_megatron_optim_config

        # Debug environment and distributed state
        env_vars = {k: v for k, v in os.environ.items() if k in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES']}
        print(f"get_model_and_optim: Environment variables: {env_vars}")
        print(f"get_model_and_optim: torch.distributed initialized: {torch.distributed.is_initialized()}")
        print(f"get_model_and_optim: Available GPUs: {torch.cuda.device_count()}")

        # Ensure distributed process group is initialized
        
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "8265")
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=rank
            )
            torch.cuda.set_device(rank)
        else:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            backend = torch.distributed.get_backend()
            

        # Ensure Megatron parallelism is initialized
        if not mpu.is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=config.actor_rollout_ref.actor.megatron.get("virtual_pipeline_model_parallel_size", None),
                context_parallel_size=config.actor_rollout_ref.actor.megatron.get("context_parallel_size", 1),
                use_sharp=False,
                nccl_communicator_config_path=None,
            )

        # Aggregate state dictionaries from worker group actors
        actors = self.actor_rollout_wg.actors  # Access actors attribute
        global_state_dict = aggregate_state_dicts(actors, config=self.get_model_config())

        # Create meta-model
        transformer_config = self.get_model_config()
        meta_model = GPTModel(transformer_config)

        # Load aggregated state dictionary
        meta_model.load_state_dict(global_state_dict)

        # Handle pipeline parallelism
        if isinstance(meta_model, list):
            meta_model = meta_model[0]

        # Initialize optimizer
        optim_config = init_megatron_optim_config(config.actor_rollout_ref.actor.optim)
        optimizer = get_megatron_optimizer(model=meta_model, config=optim_config)

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


    def get_model_and_optimizer_state(self):
        if isinstance(self.actor_rollout_wg.actor_module, list):
            model_state_dict = {i: module.state_dict() for i, module in enumerate(self.actor_rollout_wg.actor_module)}
        else:
            model_state_dict = self.actor_rollout_wg.actor_module.state_dict()
        return model_state_dict

    def get_model_config(self):
        # Load Hugging Face model configuration
        hf_config = AutoConfig.from_pretrained(self.config.actor_rollout_ref.model.path)
        # Convert to Megatron TransformerConfig
        megatron_config = self.config.actor_rollout_ref.actor.megatron
        transformer_config = TransformerConfig(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            ffn_hidden_size=hf_config.intermediate_size,
            activation_func=torch.nn.functional.silu,
            normalization="RMSNorm",
            gated_linear_unit=True,
            use_cpu_initialization=True,
            add_bias_linear=False,
            moe_token_dispatcher_type="alltoall",  # Explicitly set to alltoall
            tensor_model_parallel_size=megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=megatron_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=megatron_config.get("virtual_pipeline_model_parallel_size", None),
            context_parallel_size=megatron_config.get("context_parallel_size", 1),
            overlap_p2p_comm=megatron_config.get("overlap_p2p_comm", False),
            batch_p2p_comm=megatron_config.get("batch_p2p_comm", False),
            pipeline_dtype=torch.bfloat16,
            params_dtype=torch.bfloat16,
            sequence_parallel=megatron_config.tensor_model_parallel_size > 1,
            variable_seq_lengths=True,
            masked_softmax_fusion=True,
            attention_dropout=hf_config.attention_dropout,
            hidden_dropout=getattr(hf_config, "hidden_dropout", 0.0),
            add_qkv_bias=getattr(hf_config, "attention_bias", False),
            attention_backend="flash",
            bf16=True,
        )
            
        return transformer_config

    def create_model_provider(self, config):
        def model_provider_func(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
            transformer_config = self.get_model_config()
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(
                config.model.path,
                torch_dtype=torch.bfloat16,
                device_map=None,  # Megatron handles device placement
                trust_remote_code=True,
            )

            # Enable gradient checkpointing
            if config.model.enable_gradient_checkpointing:
                model.gradient_checkpointing_enable()

            # Set Megatron-specific attributes
            model.config = transformer_config
            model.pre_process = pre_process
            model.post_process = post_process
            return model
        return model_provider_func


     
    def _inner_loop_adaptation(self, batch: 'DataProto', optimizer = None) -> dict:
        # Fetch model and optimizer state from actor_rollout_wg
        #optimizer, _ = self.get_model_and_optim(self.config.actor_rollout_ref)
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

        return optimizer
    
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

        # Debug actor_rollout_wg initialization
        print(f"actor_rollout_wg type: {type(self.actor_rollout_wg)}")
        print(f"actor_rollout_wg attributes: {dir(self.actor_rollout_wg)}")
        
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        # Initialize meta_optimizer after workers are set up
        print("------------------- meta_optim, initializing after worker init -----------------")
        meta_optimizer = None  # Will be initialized in the first iteration if needed
          
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
                        #gen_batch_output = ray.get(self.actor_rollout_wg.generate_sequences.remote(gen_batch))
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
                        #old_log_prob = ray.get(self.actor_rollout_wg.compute_log_prob.remote(batch))
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
                            #ref_log_prob = ray.get(self.ref_policy_wg.compute_ref_log_prob.remote(batch))
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
                        
                    if self.config.meta_learning:
                        with _timer("meta_update", timing_raw):
                            # Initialize meta_optimizer if not already done
                            if meta_optimizer is None:
                                print("------------------- Initializing meta_optimizer -----------------")
                                #meta_optimizer, _ = self.get_model_and_optim(self.config.actor_rollout_ref)
                                meta_optimizer = self.get_combined_optimizer(self.actor_rollout_wg.actor_module, self.actor_rollout_re, self.config.actor_rollout_ref)
                            meta_optimizer.zero_grad()
                            for _ in range(self.meta_steps):
                                adapted_state = self._inner_loop_adaptation(batch, meta_optimizer)
                                self._set_worker_parameters(self.actor_rollout_wg, adapted_state["parameters"])
                                meta_loss = (
                                    self._compute_value_aware_loss(batch)
                                    if self.use_vapo
                                    else -torch.mean(
                                        self.actor_rollout_wg.compute_log_prob(batch)["log_probs"]
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

    def _meta_rm_score(self, log_probs: torch.Tensor, batch: 'DataProto') -> torch.Tensor:
        """Placeholder for meta RM: scores sample quality based on log-prob variance."""
        # Simulate meta RM by scoring samples based on stability (e.g., low variance)
        variance = torch.var(log_probs, dim=-1, keepdim=True)
        scores = 1.0 / (1.0 + variance)  # Higher score for lower variance
        return torch.clamp(scores, 0.0, 1.0)