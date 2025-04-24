# verl/algorithms/aigcode_c1.py
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

class AIGCode_C1(DPO):
    def __init__(self, model, reward_model, value_model, initial_beta=0.1, lambda_weight=0.5, meta_lr=1e-5, tokenizer=None):
        super().__init__(model, reward_model)
        self.value_model = ValueModel(value_model) if value_model else None
        self.beta = initial_beta
        self.lambda_weight = lambda_weight
        self.complexity_estimator = ComplexityEstimator(initial_beta)
        self.meta_lr = meta_lr
        self.tokenizer = tokenizer or get_tokenizer()

    def to_huggingface_format(self):
        """将模型转换为 HuggingFace 格式，供 OpenCompass 使用"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(self.model.config._name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
        return model, tokenizer

    def save_for_opencompass(self, save_path):
        """保存模型以供 OpenCompass 加载"""
        os.makedirs(save_path, exist_ok=True)
        model, tokenizer = self.to_huggingface_format()
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

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

    def meta_update(self, tasks):
        meta_loss = 0.0
        for task in tasks:
            # 内循环：适配任务
            task_batch = self.tokenize_batch(task)
            prompt_ids = task_batch["prompt"]["input_ids"].to(self.model.device)
            chosen_ids = task_batch["chosen"]["input_ids"].to(self.model.device)
            rejected_ids = task_batch["rejected"]["input_ids"].to(self.model.device)
            
            chosen_mask = task_batch["chosen"]["attention_mask"].to(self.model.device)
            rejected_mask = task_batch["rejected"]["attention_mask"].to(self.model.device)

            # 复制参数
            reward_params = {k: v.clone() for k, v in self.reward_model.named_parameters()}
            value_params = {k: v.clone() for k, v in self.value_model.named_parameters()}

            # 内循环损失
            chosen_rewards = self.reward_model(chosen_ids, attention_mask=chosen_mask)
            rejected_rewards = self.reward_model(rejected_ids, attention_mask=rejected_mask)
            chosen_values = self.value_model(chosen_ids, attention_mask=chosen_mask)
            rejected_values = self.value_model(rejected_ids, attention_mask=rejected_mask)

            diff = chosen_rewards - rejected_rewards + self.lambda_weight * (chosen_values - rejected_values)
            task_loss = -torch.log(torch.sigmoid(self.beta * diff)).mean()
            
            # 内循环更新
            task_grads = torch.autograd.grad(task_loss, reward_params.values(), create_graph=True)
            for (k, v), g in zip(reward_params.items(), task_grads):
                reward_params[k] = v - self.meta_lr * g
            
            task_grads = torch.autograd.grad(task_loss, value_params.values(), create_graph=True)
            for (k, v), g in zip(value_params.items(), task_grads):
                value_params[k] = v - self.meta_lr * g

            # 外循环：计算元损失
            meta_batch = self.tokenize_batch(task)
            chosen_rewards = self.reward_model(chosen_ids, attention_mask=chosen_mask, params=reward_params)
            rejected_rewards = self.reward_model(rejected_ids, attention_mask=rejected_mask, params=reward_params)
            chosen_values = self.value_model(chosen_ids, attention_mask=chosen_mask, params=value_params)
            rejected_values = self.value_model(rejected_ids, attention_mask=rejected_mask, params=value_params)

            diff = chosen_rewards - rejected_rewards + self.lambda_weight * (chosen_values - rejected_values)
            meta_loss += -torch.log(torch.sigmoid(self.beta * diff)).mean()

        # 外循环更新
        meta_grads = torch.autograd.grad(meta_loss / len(tasks), list(self.reward_model.parameters()) + list(self.value_model.parameters()))
        for param, grad in zip(list(self.reward_model.parameters()) + list(self.value_model.parameters()), meta_grads):
            if grad is not None:
                param.grad = grad
        self.optimizer.step()
        self.optimizer.zero_grad()

    def compute_loss(self, batch):
        tokenized = self.tokenize_batch(batch)
        prompt_ids = tokenized["prompt"]["input_ids"].to(self.model.device)
        chosen_ids = tokenized["chosen"]["input_ids"].to(self.model.device)
        rejected_ids = tokenized["rejected"]["input_ids"].to(self.model.device)
        
        chosen_mask = tokenized["chosen"]["attention_mask"].to(self.model.device)
        rejected_mask = tokenized["rejected"]["attention_mask"].to(self.model.device)

        chosen_rewards = self.reward_model(chosen_ids, attention_mask=chosen_mask)
        rejected_rewards = self.reward_model(rejected_ids, attention_mask=rejected_mask)
        chosen_values = self.value_model(chosen_ids, attention_mask=chosen_mask)
        rejected_values = self.value_model(rejected_ids, attention_mask=rejected_mask)

        self.update_beta(torch.cat([chosen_rewards, rejected_rewards]))

        diff = chosen_rewards - rejected_rewards + self.lambda_weight * (chosen_values - rejected_values)
        loss = -torch.log(torch.sigmoid(self.beta * diff)).mean()
        return loss

    def train_step(self, batch):
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
