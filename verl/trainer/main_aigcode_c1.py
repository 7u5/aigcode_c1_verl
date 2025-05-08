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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray
import wandb
import torch.distributed as dist
from verl.trainer.ppo.aigcode_c1_trainer import AIGCodeC1Trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.main_ppo import TaskRunner, get_custom_reward_fn

#@hydra.main(config_path="config", config_name="ppo_megatron_trainer", version_base=None)
@hydra.main(config_path="config", config_name="aigcode_c1_megatron_trainer", version_base=None)
def main(config):
    wandb.init(project=config.trainer.project_name, name=config.trainer.experiment_name)
    run_aigcode_c1(config)
    exit()
    num_gpus=config.trainer.n_gpus_per_node * config.trainer.nnodes
    os.environ.setdefault('WORLD_SIZE', str(num_gpus))
    os.environ.setdefault('RANK', "0")
    # Initialize PyTorch distributed process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",  # Use NCCL for GPU communication
            init_method="env://",  # Use environment variables for rank/world_size
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ.get("RANK", 0))
        )
    ray.init(num_gpus=num_gpus, runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "false",
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                }
            }, ignore_reinit_error=True)
    if dist.get_rank() == 0:
        wandb.init(project=config.trainer.project_name, name=config.trainer.experiment_name)
    print("============++++==== Launch =============+++++++")
    try:
        run_aigcode_c1(config)
        print("============++++==== End =============+++++++")
    finally:
        if dist.get_rank() == 0:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
        print("============++++==== Finalized =============+++++++")


def run_aigcode_c1(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()