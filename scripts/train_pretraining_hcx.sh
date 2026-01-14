#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

set -ex

DEBUG=${DEBUG:-0}

# Set environment variables
# Assuming running from the project root
export PYTHONPATH=.:$PYTHONPATH
export WANDB_DIR=./debug_data/wandb_logs
export GLOO_SOCKET_IFNAME=lo0
export NCCL_SOCKET_IFNAME=lo0

# Configuration
data_path=./example
SAVE_MODEL_NAME=hcx_omni_8b_stage1
SAVE_PATH=./checkpoints/$SAVE_MODEL_NAME
WANDB_TOKEN=xx
MODEL_PATH="/Users/user/Hands-on/ml-clara/checkpoints/fixed_model_8bit_stripped"

mkdir -p $SAVE_PATH
mkdir -p $WANDB_DIR

# Extract distributed parameters dynamically
# Adjusted for local Mac execution (single node)
NCCL_DEBUG=INFO
NUM_NODES=1
MASTER=127.0.0.1
MASTER_PORT=29500
NODE_RANK=0
# Detect number of GPUs or default to 1 for MPS/CPU cases (OpenRLHF might need GPU)
# On Mac with MPS, usually we treat it as 1 device or rely on the code handling 'mps'
NUM_LOCAL_GPUS=1 
WORLD_SIZE=$((NUM_LOCAL_GPUS * NUM_NODES))


echo "Number of nodes: ${NUM_NODES}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "Number of local GPUs: ${NUM_LOCAL_GPUS}"
echo "Master: ${MASTER}"
echo "Master port: ${MASTER_PORT}"
echo "Node rank: ${NODE_RANK}"

echo "Currently using $(which python)"

# Training command with torchrun
# Reduced batch sizes for local execution
training_commands="openrlhf.cli.train_sft \
   --max_len 1024 \
   --dataset $data_path/pretrain_data.jsonl \
   --pretrain $MODEL_PATH \
   --train_batch_size 4 \
   --micro_train_batch_size 1 \
   --ckpt_path $SAVE_PATH \
   --max_samples 500 \
   --save_path $SAVE_PATH \
   --save_steps -2 \
   --logging_steps 1 \
   --eval_steps 20 \
   --zero_stage 2 \
   --quantization int8 \
   --max_epochs 1 \
   --bf16 \
   --learning_rate 1e-4 \
   --stage stage1 \
   --generation_top_k 1 \
   --qa_loss \
   --doc_max_length 256 \
   --compress_rate 32 \
   --mse_loss \
   --gradient_checkpointing"

   # Removed --flash_attn temporarily as it might need specific install on Mac/MPS
   # Add back if installed and supported

# Build distributed arguments
# Run training directly with python for single device (bypassing torchrun/distributed issues on Mac)
# Manually set distributed environment variables for DeepSpeed
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

echo "Starting CLaRa training with HCX Omni 8B (Single Process)..."
python -m $training_commands

# Copy model file
cp openrlhf/models/modeling_clara.py $SAVE_PATH

echo "CLaRa training completed successfully!"
