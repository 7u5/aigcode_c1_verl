from verl.utils.dataset.rl_dataset import RLHFDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
config = {
    "train_files": "/sharedata/hf/BytedTsinghua-SIA_DAPO-Math-17k/data/dapo-math-17k.parquet",
    "max_prompt_length": 512,
    "cache_dir": "~/.cache/huggingface",
}
dataset = RLHFDataset(
    data_files=config["train_files"],
    tokenizer=tokenizer,
    config=config,
	cache_dir=config["cache_dir"],
    max_prompt_length=config["max_prompt_length"],
    difficulty_mode="k_fold"
)
print(f"Dataset size: {len(dataset)}")
if len(dataset) > 0:
    print(f"First sample: {dataset[0]}")
