# verl/trainer/main_aigcode_c1.py
import os

import hydra
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import torch.distributed as dist
import datetime
from verl.trainer.ppo.aigcode_c1_trainer import AIGCodeC1Trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

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
    sys.modules["custom_module"] = module
    spec.loader.exec_module(module)

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"Using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    return lambda *args, **kwargs: raw_fn(*args, **kwargs, **reward_kwargs)


#@hydra.main(config_path="config", config_name="aigcode_c1_trainer", version_base=None)
@hydra.main(config_path="config", config_name="aigcode_c1_trainer", version_base=None)
def main(config):
    print(f"[PID={os.getpid()}] RANK={os.environ['RANK']} "
      f"using GPU {torch.cuda.current_device()}")
    os.environ.setdefault("MASTER_ADDR", "10.0.15.226")
    os.environ.setdefault("MASTER_ADDR", "10.0.15.226")
    os.environ.setdefault("MASTER_PORT", "29500")
    num_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    os.environ.setdefault("WORLD_SIZE", str(num_gpus))
    os.environ.setdefault("RANK", os.environ.get("RANK", "0"))
    # Initialize PyTorch distributed process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",  # Use NCCL for GPU communication
            init_method="env://",  # Use environment variables for rank/world_size
            # world_size=int(os.environ["WORLD_SIZE"]),
            world_size=1,
            rank=int(os.environ.get("RANK", 0)),
            timeout=datetime.timedelta(seconds=10)
        )

    # ray.init(num_gpus=num_gpus, runtime_env={
    #             "env_vars": {
    #                 "TOKENIZERS_PARALLELISM": "false",
    #                 "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    #                 "MASTER_ADDR": os.environ["MASTER_ADDR"],
    #                 "MASTER_PORT": os.environ["MASTER_PORT"],
    #                 # "WORLD_SIZE": os.environ["WORLD_SIZE"],
    #             }
    #         }, ignore_reinit_error=True)
    ray.init(num_gpus=num_gpus, ignore_reinit_error=True)
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
    # Set CUDA_VISIBLE_DEVICES if not already set
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))

    # Initialize Ray
    if not ray.is_initialized():
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        ray.init(
            num_gpus=num_gpus,
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "false",
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "MASTER_ADDR": "10.0.15.226",
                    "MASTER_PORT": "29500",
                    "WORLD_SIZE": os.environ.get("WORLD_SIZE", 1),
                }
            },
        )

    # Print GPU information
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {gpu_count}")
    print(f"GPU names: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}" if gpu_count else "No GPUs")
    print(f"Ray cluster resources: {ray.cluster_resources()}")

    # Adjust config for GPU availability
    if gpu_count == 0:
        config.trainer.n_gpus_per_node = 0
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
        config.critic.ppo_micro_batch_size_per_gpu = 1
        config.actor_rollout_ref.model.dtype = "float32"
        config.critic.model.dtype = "float32"
    elif gpu_count < config.trainer.n_gpus_per_node:
        config.trainer.n_gpus_per_node = gpu_count
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = max(1, gpu_count // 2)
        config.critic.ppo_micro_batch_size_per_gpu = max(1, gpu_count // 2)

    # Load tokenizer
    model_path = config.actor_rollout_ref.model.path
    try:
        from verl.utils import hf_tokenizer
        #tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        tokenizer = hf_tokenizer(model_path, correct_pad_token=True)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer for model '{model_path}': {str(e)}")

    # Set torch_dtype
    try:
        from transformer_engine.common import recipe
        dtype_map = {"fp8": recipe.DelayedScaling, "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    except  AttributeError:
        print("Warning:transformer_engine fp8 not available. Falling back to torch.float8_e4m3fn or bfloat16.")
        dtype_map = {"fp8": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.bfloat16, "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        
    
    torch_dtype = dtype_map.get(config.actor_rollout_ref.model.dtype, torch.bfloat16)

    # Preload model to verify compatibility
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cuda" if gpu_count > 0 else "cpu",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if gpu_count > 0 else "eager",
        )
        if gpu_count > 0:
            model.to("cuda")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to preload model: {e}. Proceeding with worker initialization.")

    # Update config to enforce Flash Attention and bfloat16
    config.actor_rollout_ref.model.attn_implementation = "flash_attention_2" if gpu_count > 0 else "eager"
    config.critic.model.attn_implementation = "flash_attention_2" if gpu_count > 0 else "eager"

    # Define worker classes
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
        raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")

    # Resource allocation
    role_worker_mapping = {
        Role.ActorRollout: ActorRolloutRefWorker,
        Role.Critic: CriticWorker,
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [max(1, gpu_count)] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ActorRolloutRefWorker  
        #role_worker_mapping[Role.RefPolicy] = ray.remote(num_gpus=1 if gpu_count else 0)(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    essential_roles = [Role.ActorRollout, Role.Critic]
    if config.get("reward_model", {}).get("enable", False):
        essential_roles.append(Role.RewardModel)
    if config.get("actor_rollout_ref", {}).get("ref", {}).get("log_prob_micro_batch_size", 0) or config.get(
        "actor_rollout_ref", {}
    ).get("ref", {}).get("log_prob_micro_batch_size_per_gpu", 0):
        essential_roles.append(Role.RefPolicy)
        essential_roles.append(Role.ActorRolloutRef)
        mapping[Role.ActorRolloutRef] = global_pool_id

    for role in essential_roles:
        if not mapping.get(role):
            raise ValueError(f"No resources assigned to '{role}'. Check resource pool configuration.")

    from verl.trainer.ppo.reward import load_reward_manager

    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    from verl.utils import hf_processor

    processor = hf_processor(local_path, use_fast=True)

    if config.algorithm.type == "aigcode_c1":
        trainer = AIGCodeC1Trainer(
            config,
            tokenizer=tokenizer,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            role_worker_mapping=role_worker_mapping,
            ray_worker_group_cls=ray_worker_group_cls,
            resource_pool_manager=resource_pool_manager,
        )
    else:
        trainer = RayPPOTrainer(
            config,
            tokenizer=tokenizer,
            processor=processor,
            reward_fn=get_custom_reward_fn(config),
            val_reward_fn=val_reward_fn,
            role_worker_mapping=role_worker_mapping,
            ray_worker_group_cls=ray_worker_group_cls,
            resource_pool_manager=resource_pool_manager,
        )

    trainer.init_workers()
    trainer.fit()

if __name__ == "__main__":
    main()
