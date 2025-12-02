#!/bin/bash
# elic-rs/run_training.sh

# --- 默认值 ---
DEFAULT_NUM_GPUS=2
DEFAULT_LMBDA=0.01 # 可以设置一个默认 lambda
DEFAULT_PORT=29501 # 默认端口
DEFAULT_BATCH_SIZE=16
DEFAULT_WORKERS=16 # 每个进程的 worker 数 (2 GPU * 8 workers/GPU)
DEFAULT_EPOCHS=1000
DEFAULT_LR=1e-6
DEFAULT_PATCH_SIZE=256
DEFAULT_CHECKPOINT_DIR="./checkpoints"
DATA_PATH="./data/fmow-sentinel/train"
RESUME_CHECKPOINT="" # 默认为空，表示不续训

# --- 解析命令行参数 ---
# 使用 getopt 进行更健壮的参数解析
TEMP=$(getopt -o '' --long num-gpus:,lmbda:,rdzv-port:,batch-size:,workers:,epochs:,lr:,patch-size:,checkpoint-dir:,checkpoint: -n 'run_training.sh' -- "$@")
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"

NUM_GPUS=$DEFAULT_NUM_GPUS
LMBDA=$DEFAULT_LMBDA
PORT=$DEFAULT_PORT
BATCH_SIZE=$DEFAULT_BATCH_SIZE
WORKERS=$DEFAULT_WORKERS
EPOCHS=$DEFAULT_EPOCHS
LR=$DEFAULT_LR
PATCH_SIZE=$DEFAULT_PATCH_SIZE
CHECKPOINT_DIR=$DEFAULT_CHECKPOINT_DIR

while true; do
  case "$1" in
    --num-gpus ) NUM_GPUS="$2"; shift 2 ;;
    --lmbda ) LMBDA="$2"; shift 2 ;;
    --rdzv-port ) PORT="$2"; shift 2 ;;
    --batch-size ) BATCH_SIZE="$2"; shift 2 ;;
    --workers ) WORKERS="$2"; shift 2 ;;
    --epochs ) EPOCHS="$2"; shift 2 ;;
    --lr ) LR="$2"; shift 2 ;;
    --patch-size ) PATCH_SIZE="$2"; shift 2 ;;
    --checkpoint-dir ) CHECKPOINT_DIR="$2"; shift 2 ;;
    --checkpoint ) RESUME_CHECKPOINT="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

# --- 构建 torchrun 命令 ---
CMD="torchrun --nproc_per_node=$NUM_GPUS \
    --rdzv_endpoint=\"localhost:$PORT\" \
    src/train_ms.py \
    -m elic2022_ms \
    -d \"$DATA_PATH\" \
    -l $LMBDA \
    -b $BATCH_SIZE \
    -e $EPOCHS \
    -lr $LR \
    --patch-size $PATCH_SIZE \
    -n $WORKERS \
    --cuda \
    --save \
    --checkpoint-dir \"$CHECKPOINT_DIR\""

# 如果提供了 checkpoint 参数，则添加到命令中
if [ ! -z "$RESUME_CHECKPOINT" ]; then
  CMD="$CMD --checkpoint \"$RESUME_CHECKPOINT\""
fi

# --- 执行命令 ---
echo "--- Starting Training ---"
echo "GPUs: $CUDA_VISIBLE_DEVICES (Count: $NUM_GPUS)"
echo "Lambda: $LMBDA"
echo "Port: $PORT"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Workers per process: $WORKERS"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
if [ ! -z "$RESUME_CHECKPOINT" ]; then
  echo "Resuming from: $RESUME_CHECKPOINT"
fi
echo "-------------------------"
echo "Executing: $CMD"
echo "-------------------------"

eval $CMD # 使用 eval 来执行包含引号和变量的命令

echo "DDP (Lambda=$LMBDA) 训练完成"