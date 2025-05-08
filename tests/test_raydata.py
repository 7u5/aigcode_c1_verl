import os
import ray
import logging
from transformers import AutoTokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset
import omegaconf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote
def test_rlhfdataset(config):
    logger.info("Running RLHFDataset test in Ray")
    logger.info(f"Config: {config}")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
    dataset = RLHFDataset(
        data_files=config.data.train_files,
        tokenizer=tokenizer,
        config=config.data,
        cache_dir=config.data.cache_dir,
        max_length=config.data.max_prompt_length,
        difficulty_mode=config.algorithm.difficulty_mode
    )
    logger.info(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        logger.info(f"First pair: {dataset[0]}")
    return len(dataset)

if __name__ == "__main__":
    config = omegaconf.OmegaConf.create({
        "data": {
            #"train_files": "/sharedata/hf/BytedTsinghua-SIA_DAPO-Math-17k/data/dapo-math-17k.parquet",
            "train_files": "/sharedata/hf/BytedTsinghua-SIA_AIME-2024/data/aime-2024.parquet",
            "max_prompt_length": 2048,
            "cache_dir": "~/.cache/verl/rlhf",
            "dataset_name": None,
            "prompt_key": "prompt",
            "truncation": "left",
            "filter_overlong_prompts": False,
            "use_preference": True,
            "max_samples": 1000
        },
        "algorithm": {
            "difficulty_mode": None,
            "use_preference": True
        },
        "ray_init": {
            "num_cpus": None
        }
    })

    os.environ["PYTHONPATH"] = "/sharedata/qiuwu/aigcode_c1_verl:" + os.environ.get("PYTHONPATH", "")
    #ray.init(num_cpus=config.ray_init.num_cpus, runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true"}})
    ray.init(num_cpus=config.ray_init.num_cpus, runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true"}})
    logger.info("Submitting Ray task")
    result = ray.get(test_rlhfdataset.remote(config))
    logger.info(f"Result: Dataset size = {result}")
    ray.shutdown()
