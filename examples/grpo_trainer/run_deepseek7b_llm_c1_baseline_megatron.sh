#!/bin/bash
set -x
# Ensure Ray detects GPUs correctly
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
echo $CUDA_VISIBLE_DEVICES
# Validate GPU availability
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "No GPUs detected. Falling back to CPU mode."
  GPU_PER_NODE=0
  MICRO_BATCH_SIZE=1
  TENSOR_PARALLEL=1
  DTYPE="float32"
else
  GPU_PER_NODE=${NUM_GPUS}
  MICRO_BATCH_SIZE=4
  TENSOR_PARALLEL=2
  # Check if GPUs support FP8 (e.g., NVIDIA H100 or newer)
  if nvidia-smi --query-gpu=name --format=csv,noheader | grep "H20Z"|wc -l; then
    DTYPE="fp8"  # H20Z/H200 fully support FP8; fallback to bfloat16
  else
    DTYPE="bfloat16"
  fi
fi
DTYPE="bfloat16"
# Validate dataset paths
HOME=/home/aigc
DATA_DIR="$HOME/data/gsm8k"
TRAIN_FILE="$DATA_DIR/train.parquet"
VAL_FILE="$DATA_DIR/test.parquet"
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
  echo "Error: Dataset files not found at $DATA_DIR"
  exit 1
fi
MODEL_PATH="deepseek-ai/deepseek-llm-7b-chat"

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

# 0. download the model
#huggingface-cli download $MODEL_PATH
TIME_NOW=`date +%Y%m%d_%H%M%S`
DIST_CKPT_PATH="ckpt.$TIME_NOW"
CKPT_DIR="checkpoints"
mkdir -p $DIST_CKPT_PATH
# 1. convert the model to mcore format
#python -c "from transformers import AutoModel; AutoModel.from_pretrained('deepseek-ai/deepseek-llm-7b-chat', trust_remote_code=True, model_type="deepseek_v3")
# local downloaded model path = /sharedata/osmodels/deepseek-math-7b-instruct
if ! python -c "from transformers import AutoModel; AutoModel.from_pretrained('$MODEL_PATH', trust_remote_code=True)" >/dev/null 2>&1; then
  echo "Error: Model $MODEL_PATH not accessible"
  exit 1
fi
python scripts/converter_hf_to_mcore.py --hf_model_path $MODEL_PATH --output_path $DIST_CKPT_PATH

# 2. run the script
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
train_files=$gsm8k_train_path
test_files=$gsm8k_test_path

NODES=$NUM_GPUS
PP=2
TP=$TENSOR_PARALLEL
CP=1
VLLM_TP=$TENSOR_PARALLEL
#MASTER_ADDR="10.0.15.226"
#MASTER_PORT="29500"
HYDRA_FULL_ERROR=1
# Validate model path
#    trainer.default_local_dir=$DIST_CKPT_PATH \
export PYTHONPATH=/sharedata/qiuwu/aigcode_c1:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet
#train_files="['$gsm8k_train_path', '$math_train_path']"
#test_files="['$gsm8k_test_path', '$math_test_path']"
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"
config_file=""
#    --config-name='ppo_megatron_trainer.yaml'\
python3 -m verl.trainer.main_aigcode_c1 --config-path=config \
    --config-name='aigcode_c1_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_aigcode_c1' \
    trainer.experiment_name='deepseek_llm_7b_chatbot_megatron' \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@