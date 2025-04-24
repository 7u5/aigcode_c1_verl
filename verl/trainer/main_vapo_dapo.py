# verl/trainer/main_vapo_dapo.py
import torch
from verl.algorithms.vapo_dapo import VAPO_DAPO
from verl.workers.fsdp_workers import FSDPWorker
from verl.utils.megatron import initialize_megatron
from verl.datasets import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to reward model")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to value model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to preference dataset")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--initial_beta", type=float, default=0.1, help="Initial temperature parameter")
    parser.add_argument("--lambda_weight", type=float, default=0.5, help="Weight for value function")
    return parser.parse_args()

def main():
    args = parse_args()
    initialize_megatron()

    # 加载模型
    model = torch.load(args.model_path)
    reward_model = torch.load(args.reward_model_path)
    value_model = torch.load(args.value_model_path)

    # 加载数据集
    data_loader = load_dataset(args.dataset_path, batch_size=16)

    # 初始化 VAPO-DAPO 和工作进程
    vapo_dapo = VAPO_DAPO(model, reward_model, value_model, args.initial_beta, args.lambda_weight)
    workers = [FSDPWorker(model, rank) for rank in range(torch.distributed.get_world_size())]

    # 训练
    for step in range(args.max_steps):
        batch = next(data_loader)
        loss = vapo_dapo.train_step(batch)
        workers[0].reshard("training")
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss}, Beta: {vapo_dapo.beta}")

        # 保存检查点
        if step % 100 == 0:
            torch.save(model.state_dict(), f"checkpoint_step_{step}.pt")

if __name__ == "__main__":
    main()
