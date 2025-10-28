#!/bin/bash

# --- 1. GPU 数量 ---
NUM_GPUS=2

# --- 2. 优化参数 ---
DATA_PATH="./data/fmow-sentinel/train"
LMBDA=1

# --- 3. 批大小 ---
BATCH_SIZE=16 # (每块 GPU 16 个)

# --- 4. Worker 数量 ---
WORKERS=32

# --- 5. (关键修复) 使用 torchrun 启动 DDP 训练 ---
# 确保所有 \ 字符是该行的 *绝对* 最后一个字符
# 后面不能有空格、Tab 或注释

torchrun --nproc_per_node=$NUM_GPUS src/train_ms.py \
    -m elic2022_ms \
    -d "$DATA_PATH" \
    -l $LMBDA \
    -b $BATCH_SIZE \
    -e 1000 \
    -lr 1e-4 \
    --patch-size 256 \
    -n $WORKERS \
    --cuda \
    --save \

echo "DDP (4 GPU, BS=16) 训练完成"