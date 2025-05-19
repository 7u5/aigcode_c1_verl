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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray
import torch
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.aigcode_c1_trainer import AIGCodeC1Trainer
from verl.trainer.ppo.reward import load_reward_manager
import datetime
import logging
logger = logging.getLogger(__name__)

def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path="config", config_name="aigcode_c1_megatron_trainer", version_base=None)
def main(config):
    run_c1(config)
    
import socket
def is_port_free(port, host="localhost"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, int(port))) != 0
    
def run_c1(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8266"

    # Check port availability
    master_port = os.environ["MASTER_PORT"]
    if not is_port_free(master_port):
        raise RuntimeError(f"Port {master_port} is already in use. Choose a different MASTER_PORT.")

    # Calculate world_size
    required_gpus = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * \
                    config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < required_gpus:
        raise ValueError(f"Available GPUs ({gpu_count}) less than required ({required_gpus})")
    world_size = required_gpus
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = os.environ.get("RANK", "0")
    
     # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            #num_gpus=gpu_count,
            num_cpus=config.ray_init.num_cpus,
            dashboard_port=8888,  # Consistent with provided code
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "MASTER_ADDR": os.environ["MASTER_ADDR"],
                    "MASTER_PORT": os.environ["MASTER_PORT"],
                    "WORLD_SIZE": str(world_size),
                    "RANK": os.environ["RANK"],
                }
            },
        )
        print("run_c1: Ray initialized with resources:", ray.cluster_resources())
        print("run_c1: Ray available resources:", ray.available_resources())
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

@ray.remote
class MegatronActor:
    def __init__(self, model, config):
        import os
        from megatron.core import mpu

        self.model = model
        self.config = config

        # Initialize torch.distributed
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "29500")

            torch.distributed.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank,
                world_size=world_size
            )

        # Initialize Megatron parallelism
        if not mpu.is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=config.tensor_model_parallel_size,
                pipeline_model_parallel_size=config.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=config.get("virtual_pipeline_model_parallel_size", None),
                context_parallel_size=config.get("context_parallel_size", 1),
            )

    def get_model(self):
        return self.model

    def get_state_dict(self):
        return self.model.state_dict()

    def cleanup(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):   
        from verl.utils import hf_processor, hf_tokenizer
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "8265")
        # Initialize torch.distributed
        print(os.environ.get("CUDA_VISIBLE_DEVICES", ""))

        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        '''
        # Check available GPUs
        ray_available_resources = ray.cluster_resources()
        available_gpus = int(ray_available_resources.get("GPU", 0))
        logger.info(f"Ray available resources: {ray_available_resources}")
        logger.info(f"Ray available GPUs: {available_gpus}")
        
        world_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
        total_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if available_gpus < total_gpus:
            logger.warning(
                f"Available GPUs ({available_gpus}) is less than desired GPUs ({total_gpus}). "
                f"Adjusting tensor_model_parallel_size and pipeline_model_parallel_size."
            )
            config.actor.megatron.tensor_model_parallel_size = available_gpus
            config.actor.megatron.pipeline_model_parallel_size = 1
            config.actor_rollout_ref.rollout.tensor_model_parallel_size = available_gpus
            config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = available_gpus
            config.actor_rollout_ref.ref.megatron.pipeline_model_parallel_size = 1
            config.trainer.n_gpus_per_node = available_gpus
            world_size = available_gpus
        actors = []
        for rank in range(world_size):
            env = {
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
                "CUDA_VISIBLE_DEVICES": str(rank % world_size),
            }

            actor = MegatronActor.options(
                #num_cpus=config.ray_init.num_cpus,
                num_gpus=1,
                runtime_env={"env_vars": env}  # Use runtime_env to pass environment variables
            ).remote(model=None, config=config.actor_rollout_ref.actor.megatron)
            actors.append(actor)
        '''
            
        trainer = AIGCodeC1Trainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        #worker_group = NVMegatronRayWorkerGroup(actors, config=config.actor.megatron)
        #worker_group.world_size = world_size  # Set if needed
        #trainer.actor_rollout_wg = worker_group
        trainer.fit()
        
if __name__ == "__main__":
    main()
