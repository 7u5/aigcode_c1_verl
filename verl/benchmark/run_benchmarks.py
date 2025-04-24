# verl/benchmarks/run_benchmarks.py
import torch
from verl.algorithms.vapo_dapo import VAPO_DAPO
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
    parser.add_argument("--dataset_name", type=str, default="beyond_aime", choices=["beyond_aime", "codeforces", "gpqa"])
    parser.add_argument("--output_file", type=str, default="benchmark_results.json")
    return parser.parse_args()

def load_benchmark_dataset(dataset_name):
    if dataset_name == "beyond_aime":
        # 假设 BeyondAIME 数据集格式
        dataset = hf_load_dataset("path/to/beyond_aime")  # 需替换为实际路径
    elif dataset_name == "codeforces":
        dataset = hf_load_dataset("path/to/codeforces")
    elif dataset_name == "gpqa":
        dataset = hf_load_dataset("gpqa")
    return dataset["test"]

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
        
        # 假设选择更优回复作为预测
        chosen_score = model.reward_model(tokenizer(chosen, return_tensors="pt")["input_ids"].to(model.model.device)).mean()
        rejected_score = model.reward_model(tokenizer(rejected, return_tensors="pt")["input_ids"].to(model.model.device)).mean()
        prediction = 1 if chosen_score > rejected_score else 0
        ground_truth = 1  # 假设 chosen 是正确答案
        
        predictions.append(prediction)
        ground_truths.append(ground_truth)
    
    accuracy = accuracy_score(ground_truths, predictions)
    return {"accuracy": accuracy, "avg_loss": loss.item()}

def main():
    args = parse_args()
    initialize_megatron()

    # 加载模型
    model = torch.load(args.model_path)
    reward_model = torch.load(args.reward_model_path)
    value_model = torch.load(args.value_model_path)
    tokenizer = get_tokenizer()
    
    vapo_dapo = VAPO_DAPO(model, reward_model, value_model)
    
    # 加载基准测试数据集
    dataset = load_benchmark_dataset(args.dataset_name)
    
    # 评估
    results = evaluate(vapo_dapo, dataset, tokenizer)
    
    # 保存结果
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark results for {args.dataset_name}: {results}")

if __name__ == "__main__":
    main()
