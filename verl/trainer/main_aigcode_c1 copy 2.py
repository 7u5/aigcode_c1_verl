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
from omegaconf import OmegaConf
from verl.utils.config import merge_yaml_args
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="aigcode_c1_megatron_trainer", version_base=None)
def main(config):
    """Main entry point for the script."""
    # Merge Hydra config with command-line args
    #config = merge_yaml_args(config)
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Enable full stack traces
    
    # Get rank for distributed training
    rank = int(os.environ.get('RANK', '0'))
    
    if rank == 0:
        wandb.init(project=config.trainer.project_name, name=config.trainer.experiment_name)
        print("============++++==== Launch =============+++++++")
        run_aigcode_c1(config)
        print("============++++==== End =============+++++++")
    
    # Clean up
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

 
def run_aigcode_c1(config):
    """Run the AIGCode C1 training pipeline."""
    logger.info("Starting run_aigcode_c1")

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
    # Run the training task
    try:
        ray.get(runner.run.remote(config))
    except Exception as e:
        logger.error(f"Error in TaskRunner: {e}")
        raise
    finally:
        ray.shutdown()
    
    logger.info("run_aigcode_c1 completed")

if __name__ == "__main__":
    main()