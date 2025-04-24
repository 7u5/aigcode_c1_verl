# verl/algorithms/vapo_dapo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from verl.algorithms.dpo import DPO
from verl.utils.megatron import get_tokenizer

class ValueModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, -1, :]
        value = self.value_head(hidden_states)
        return value

class ComplexityEstimator:
    def __init__(self, base_beta=0.1, max_beta=1.0, min_beta=0.01):
        self.base_beta = base_beta
        self.max_beta = max_beta
        self.min_beta = min_beta

    def __call__(self, variance):
        beta = self.base_beta + 0.01 * variance
        return max(self.min_beta, min(self.max_beta, beta))

class VAPO_DAPO(DPO):
    def __init__(self, model, reward_model, value_model, initial_beta=0.1, lambda_weight=0.5, tokenizer=None):
        super().__init__(model, reward_model)
        self.value_model = ValueModel(value_model) if value_model else None
        self.beta = initial_beta
        self.lambda_weight = lambda_weight
        self.complexity_estimator = ComplexityEstimator(initial_beta)
        self.tokenizer = tokenizer or get_tokenizer()

    def tokenize_batch(self, batch):
        prompts = batch["prompt"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        chosen_inputs = self.tokenizer(chosen, return_tensors="pt", padding=True, truncation=True)
        rejected_inputs = self.tokenizer(rejected, return_tensors="pt", padding=True, truncation=True)
        
        return {
            "prompt": prompt_inputs,
            "chosen": chosen_inputs,
            "rejected": rejected_inputs
        }

    def update_beta(self, rewards):
        variance = torch.var(rewards)
        self.beta = self.complexity_estimator(variance)

    def compute_loss(self, batch):
        tokenized = self.tokenize_batch(batch)
        prompt_ids = tokenized["prompt"]["input_ids"].to(self.model.device)
        chosen_ids = tokenized["chosen"]["input_ids"].to(self.model.device)
        rejected_ids = tokenized["rejected"]["input_ids"].to(self.model.device)
        
        prompt_mask = tokenized["prompt"]["attention_mask"].to(self.model.device)
        chosen_mask = tokenized["chosen"]["attention_mask"].to(self.model.device)
        rejected_mask = tokenized["rejected"]["attention_mask"].to(self.model.device)

        # 计算奖励
        chosen_rewards = self.reward_model(chosen_ids, attention_mask=chosen_mask)
        rejected_rewards = self.reward_model(rejected_ids, attention_mask=rejected_mask)

        # 计算价值
        chosen_values = self.value_model(chosen_ids, attention_mask=chosen_mask)
        rejected_values = self.value_model(rejected_ids, attention_mask=rejected_mask)

        # 更新 beta
        self.update_beta(torch.cat([chosen_rewards, rejected_rewards]))

        # VAPO-DAPO 损失
        diff = chosen_rewards - rejected_rewards + self.lambda_weight * (chosen_values - rejected_values)
        loss = -torch.log(torch.sigmoid(self.beta * diff)).mean()
        return loss

    def train_step(self, batch):
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
