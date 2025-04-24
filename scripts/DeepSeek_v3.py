# scripts/DeepSeek_v3.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from transformers import AutoConfig, AutoTokenizer

# 手动指定模型类型
model_name = "deepseek-ai/DeepSeek-V3-0324"
config = AutoConfig.from_pretrained(model_name, model_type="deepseek_v3")  # 假设模型类型是 deepseek_v3
tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)

# Model ID
model_name = "deepseek-ai/DeepSeek-V3-0324"  # Updated to correct model ID
hf_token = os.environ.get("HF_TOKEN", "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")  # Replace with your token

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_type="deepseek_v3", token=hf_token)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

# Load model with quantization for memory efficiency
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        model_type="deepseek_v3",
        device_map="auto",
        token=hf_token,
        trust_remote_code=True  # Required for custom models
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Example input for testing
prompt = "Generate fields for task management in a project database."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated response:", response)