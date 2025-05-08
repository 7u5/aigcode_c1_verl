import sys
import os
sys.path.insert(0, '/sharedata/qiuwu/aigcode_c1_verl')

from verl.utils.dataset.rl_dataset import PreferencePairDataset
from transformers import AutoTokenizer
import omegaconf
import logging
import json
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_SAMPLES = 1000  # Limit for testing

def convert_pair_to_serializable(pair):
    """Convert tensors in a pair to lists for JSON serialization."""
    serializable_pair = {}
    for key, value in pair.items():
        if isinstance(value, torch.Tensor):
            serializable_pair[key] = value.tolist()
        else:
            serializable_pair[key] = value
    return serializable_pair

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
config = omegaconf.OmegaConf.create({
    "data": {
        #"train_files": "/sharedata/hf/BytedTsinghua-SIA_DAPO-Math-17k/data/dapo-math-17k.parquet",
        "train_files": "/sharedata/hf/BytedTsinghua-SIA_DAPO-Math-17k/data/dapo-math-17k.parquet",
        "max_prompt_length": 1024,
        "cache_dir": "~/.cache/verl/rlhf",
        "dataset_name": None,
        "prompt_key": "prompt",
        "truncation": "left",
        "filter_overlong_prompts": False
    },
    "algorithm": {
        "difficulty_mode": None,
        "use_preference": True
    }
})

logger.info("Initializing PreferencePairDataset")
try:
    dataset = PreferencePairDataset(
        data_files=config.data.train_files,
        tokenizer=tokenizer,
        config=config.data,
        cache_dir=config.data.cache_dir,
        mode=config.algorithm.difficulty_mode,
        max_samples=MAX_SAMPLES
    )
except Exception as e:
    logger.error(f"Failed to initialize dataset: {e}")
    raise

logger.info(f"Dataset size: {len(dataset)}")
if len(dataset) > 0:
    first_pair = convert_pair_to_serializable(dataset[0])
    logger.info(f"First pair: {first_pair}")
    # Save first 10 pairs to file
    try:
        with open("sample_pairs.json", "w") as f:
            json.dump([convert_pair_to_serializable(pair) for pair in dataset[:10]], f, indent=2)
        logger.info("Saved sample pairs to sample_pairs.json")
    except Exception as e:
        logger.error(f"Failed to save pairs: {e}")
else:
    logger.error("Dataset is empty")
