# verl/trainer/main_aigcode_c1.py
import torch
from verl.algorithms.aigcode_c1 import AIGCode_C1 
from verl.workers.fsdp_workers import FSDPWorker
from verl.utils.megatron import initialize_megatron
from verl.datasets import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--value_model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--initial_beta", type=float, default=0.1)
    parser.add_argument("--lambda_weight", type=float, default=0.5)
    parser.add_argument("--meta_lr", type=float, default=1e-5)
    parser.add_argument("--tasks_per_step", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    initialize_megatron()

    model = torch.load(args.model_path)
    reward_model = torch.load(args.reward_model_path)
    value_model = torch.load(args.value_model_path)

    data_loader = load_dataset(args.dataset_path, batch_size=16)

    aigcode_c1 = AIGCode_C1(model, reward_model, value_model, args.initial_beta, args.lambda_weight, args.meta_lr)
    workers = [FSDPWorker(model, rank) for rank in range(torch.distributed.get_world_size())]

    for step in range(args.max_steps):
        tasks = [next(data_loader) for _ in range(args.tasks_per_step)]
        aigcode_c1.meta_update(tasks)
        workers[0].reshard("training")
        if step % 10 == 0:
            batch = next(data_loader)
            loss = aigcode_c1.compute_loss(batch)
            print(f"Step {step}, Loss: {loss}, Beta: {aigcode_c1.beta}")

        if step % 100 == 0:
            torch.save(model.state_dict(), f"checkpoint_step_{step}.pt")
            aigcode_c1.save_for_opencompass(f"checkpoint_step_{step}")

if __name__ == "__main__":
    main()
