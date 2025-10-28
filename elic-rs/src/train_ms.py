# src/train_ms.py
# 最终合并版 V2：
# 1. 包含 DDP + AMP + tqdm
# 2. 修正了所有 DDP/AMP 错误 (aux_loss, find_unused, warnings)
# 3. 包含 --checkpoint 参数以支持断点续训
# 4. (新功能) 添加 --checkpoint-dir 参数，用于指定保存检查点的目录

import argparse
import math
import random
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- DDP 导入 ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- AMP 导入 ---
from torch.amp import GradScaler, autocast 

# --- 自定义模块导入 ---
from ms_models import Elic2022MultiSpectral
from ms_datasets import FMoWS2Dataset, collate_fn
from compressai.losses import RateDistortionLoss

CUSTOM_MODELS = {
    "elic2022_ms": Elic2022MultiSpectral,
}

def ddp_setup():
    """初始化 DDP 进程组"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, 
    scaler, local_rank
):
    # (这个函数无需修改)
    model.train()
    device = next(model.parameters()).device
    pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch}", disable=(local_rank != 0))
    for i, d in enumerate(pbar):
        if d.shape[0] == 0:
            if local_rank == 0: pbar.set_postfix(status="Skipped empty batch")
            continue
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        with autocast(device_type="cuda"):
            out_net = model(d)
            out_criterion = criterion(out_net, d)
        scaler.scale(out_criterion["loss"]).backward()
        if clip_max_norm > 0:
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        scaler.step(optimizer)
        aux_loss = model.module.aux_loss() 
        scaler.scale(aux_loss).backward()
        scaler.step(aux_optimizer)
        scaler.update() 
        if local_rank == 0:
            pbar.set_postfix(loss=f'{out_criterion["loss"].item():.3f}', mse=f'{out_criterion["mse_loss"].item():.5f}', bpp=f'{out_criterion["bpp_loss"].item():.3f}', aux=f'{aux_loss.item():.2f}')

def main(argv):
    local_rank, world_size = ddp_setup()
    
    parser = argparse.ArgumentParser(description="Multi-Spectral ELIC DDP Training")
    # (所有 parser.add_argument(...) 参数保持不变)
    parser.add_argument("-m", "--model", default="elic2022_ms", choices=CUSTOM_MODELS.keys(), help="Custom model")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset path")
    parser.add_argument("-l", "--lmbda", type=float, default=0.01, help="Lambda value")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size PER GPU")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--patch-size", type=int, default=256, help="Training patch size")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", help="Save model checkpoint")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="Gradient clipping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to resume checkpoint")
    
    # --- (新功能) 添加 --checkpoint-dir 参数 ---
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="./checkpoints", # 默认保存到 checkpoints 文件夹
        help="Directory to save checkpoints"
    )
    # --- (添加结束) ---
    
    args = parser.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = f"cuda:{local_rank}"

    # --- 数据集设置 (无需修改) ---
    if local_rank == 0: print("Loading custom fMoW-S2 dataset...")
    train_dataset = FMoWS2Dataset(root=args.dataset, patch_size=args.patch_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=collate_fn, sampler=train_sampler)
    
    # --- 模型设置 (无需修改) ---
    if local_rank == 0: print(f"Creating model: {args.model}")
    net = CUSTOM_MODELS[args.model]()
    net = net.to(device)
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)

    # --- 优化器设置 (无需修改) ---
    if local_rank == 0: print("Setting up optimizers...")
    parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles")}
    optimizer = optim.Adam((p for n, p in net.named_parameters() if n in parameters), lr=args.learning_rate)
    aux_parameters = {n for n, p in net.named_parameters() if n.endswith(".quantiles")}
    aux_optimizer = optim.Adam((p for n, p in net.named_parameters() if n in aux_parameters), lr=args.learning_rate * 10)

    # --- 损失函数 和 AMP Scaler (无需修改) ---
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric="mse")
    scaler = GradScaler()

    # --- 加载检查点 (无需修改) ---
    start_epoch = 0
    if args.checkpoint is not None:
        if local_rank == 0: print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.module.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if local_rank == 0: print(f"Resuming training from Epoch {start_epoch}")

    # --- (新功能) 检查并创建检查点目录 ---
    # 只有主进程 (rank 0) 需要创建目录
    if args.save and local_rank == 0:
        if not os.path.exists(args.checkpoint_dir):
            print(f"创建检查点保存目录: {args.checkpoint_dir}")
            os.makedirs(args.checkpoint_dir)
    # --- (添加结束) ---

    # --- DDP 训练循环 ---
    if local_rank == 0: print(f"Start DDP training from Epoch {start_epoch} to {args.epochs} on {world_size} GPUs.")
    for epoch in range(start_epoch, args.epochs): 
        train_sampler.set_epoch(epoch)
        train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, scaler, local_rank)

        # --- (关键修改) 修改保存路径 ---
        if args.save and local_rank == 0:
            # 基础文件名
            base_filename = f"elic2022_ms_lmbda_{args.lmbda}_epoch_{epoch}.pth.tar"
            # 完整的保存路径 (目录 + 文件名)
            save_path = os.path.join(args.checkpoint_dir, base_filename) 
            
            print(f"Saving checkpoint: {save_path}")
            torch.save({
                "epoch": epoch,
                "state_dict": net.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
            }, save_path)
        # --- (修改结束) ---
            
    if local_rank == 0: print("Training finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main(sys.argv[1:])