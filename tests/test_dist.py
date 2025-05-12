import os
import torch
import torch.distributed as dist

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "8265"
os.environ["WORLD_SIZE"] = "8"
os.environ["RANK"] = os.environ.get("RANK", "0")

dist.init_process_group(backend="nccl", init_method="env://")
print(f"Rank {dist.get_rank()} initialized successfully")
dist.destroy_process_group()
