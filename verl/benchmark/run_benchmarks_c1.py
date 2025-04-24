# verl/benchmarks/run_benchmarks.py
import torch
from verl.algorithms.aigcode_c1 import AIGCode_C1
from verl.utils.megatron import initialize_megatron, get_tokenizer
from datasets import load_dataset as hf_load_dataset
import argparse
import json
from sklearn.metrics import accuracy_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--value_model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="beyond_aime", choices=["beyond_aime", "codeforces", "gpqa", "custom_views"])
    parser.add_argument("--output_file", type=str, default="benchmark_results.json")
    return parser.parse_args()

def load_benchmark_dataset(dataset_name):
    dataset_dir = "utils/dataset"
    if dataset_name == "beyond_aime":
        dataset = hf_load_dataset("HuggingFaceH4/aime_2024")["test"]
    elif dataset_name == "codeforces":
        dataset = hf_load_dataset("open-r1/codeforces")["test"]
    elif dataset_name == "gpqa":
        dataset = hf_load_dataset("Idavidrein/gpqa")["test"]
    elif dataset_name == "custom_views":
        dataset = load_custom_views_dataset()  # 用户系统数据集
    return dataset

def load_custom_views_dataset():
    # 模拟用户系统的视图字段选择数据集
    return [
        {
            "prompt": "Generate fields for task management in project_db",
            "chosen": ["node_code", "status", "input_data"],
            "rejected": ["address", "gender"]
        },
        # 更多样本...
    ]

def evaluate(model, dataset, tokenizer):
    predictions = []
    ground_truths = []
    
    for example in dataset:
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        batch = {
            "prompt": [prompt],
            "chosen": [chosen],
            "rejected": [rejected]
        }
        loss = model.compute_loss(batch)
        
        chosen_score = model.reward_model(tokenizer(chosen, return_tensors="pt")["input_ids"].to(model.model.device)).mean()
        rejected_score = model.reward_model(tokenizer(rejected, return_tensors="pt")["input_ids"].to(model.model.device)).mean()
        prediction = 1 if chosen_score > rejected_score else 0
        ground_truth = 1
        
        predictions.append(prediction)
        ground_truths.append(ground_truth)
    
    accuracy = accuracy_score(ground_truths, predictions)
    return {"accuracy": accuracy, "avg_loss": loss.item(), "beta": model.beta}

def main():
    args = parse_args()
    initialize_megatron()

    model = torch.load(args.model_path)
    reward_model = torch.load(args.reward_model_path)
    value_model = torch.load(args.value_model_path)
    tokenizer = get_tokenizer()
    
    aigcode_c1 = AIGCode_C1(model, reward_model, value_model)
    
    dataset = load_benchmark_dataset(args.dataset_name)
    results = evaluate(aigcode_c1, dataset, tokenizer)
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark results for {args.dataset_name}: {results}")

if __name__ == "__main__":
    main()
