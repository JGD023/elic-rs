# src/ms_datasets.py

import os
import glob
import torch
import torch.nn.functional as F
import rasterio
import numpy as np
from torch.utils.data import Dataset

class FMoWS2Dataset(Dataset):
    """
    用于 fMoW-S2 多光谱 (13 波段) 数据的自定义 Dataset。
    
    *** 新版本：
    1. 支持对小于 patch_size 的图像进行 0 填充。
    2. 在遇到通道数不匹配或读取失败时，返回 empty(0) 以便 collate_fn 过滤。
    ***
    """
    def __init__(self, root, patch_size=256, split='train'):
        self.root = root
        self.patch_size = patch_size
        self.split = split
        
        search_path = os.path.join(self.root, "**", "*.tif")
        print(f"正在 {self.root} 目录中递归搜索 *.tif 文件...")
        self.image_files = sorted(glob.glob(search_path, recursive=True))
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"在 {self.root} 及其子目录中未找到任何 .tif 图像文件")
            
        print(f"已找到 {len(self.image_files)} 个图像文件。")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filepath = self.image_files[idx]
        
        try:
            with rasterio.open(filepath) as src:
                img = src.read().astype(np.float32) 
        except Exception as e:
            print(f"警告：读取文件 {filepath} 失败: {e}")
            # --- (关键修改 1) ---
            # 返回空张量，让 collate_fn 过滤掉
            return torch.empty(0) 

        # --- (关键修改 2) ---
        if img.shape[0] != 13:
            # 打印警告，但只在训练刚开始时少量打印
            if idx < 50: # 避免刷屏
                 print(f"警告：文件 {filepath} 的通道数不是 13 (而是 {img.shape[0]})，已跳过。")
            # 返回空张量，让 collate_fn 过滤掉
            return torch.empty(0)

        # 归一化
        img = np.clip(img / 10000.0, 0.0, 1.0)
        img_tensor = torch.from_numpy(img)
        
        # 填充逻辑
        c, h, w = img_tensor.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            padding = (0, pad_w, 0, pad_h) 
            img_tensor = F.pad(img_tensor, padding, mode='constant', value=0)
            
        # 随机裁剪
        c, h_padded, w_padded = img_tensor.shape
        top = torch.randint(0, h_padded - self.patch_size + 1, (1,)).item()
        left = torch.randint(0, w_padded - self.patch_size + 1, (1,)).item()
        
        patch = img_tensor[:, top : top + self.patch_size, left : left + self.patch_size]
        
        return patch

# 这个函数现在至关重要，它会过滤掉所有返回 torch.empty(0) 的坏数据
def collate_fn(batch):
    # 过滤掉那些读取失败的空张量
    batch = [b for b in batch if b.shape[0] > 0]
    if len(batch) == 0:
        # 如果整个批次都是坏数据，返回一个空张量，训练循环会跳过它
        return torch.empty(0)
    # 用剩余的好数据组成批次
    return torch.utils.data.dataloader.default_collate(batch)