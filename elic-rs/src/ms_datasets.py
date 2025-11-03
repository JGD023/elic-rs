# src/ms_datasets.py
# (V6 修复版)

import os
import glob
import torch
import torch.nn.functional as F
import rasterio
import numpy as np
from torch.utils.data import Dataset

# (V6 修复)
# 我们定义一个最小的"信号"阈值。
# 如果一个 patch 的 (max - min) 小于这个值，我们就认为它是不稳定的。
MIN_SIGNAL_RANGE = 1e-5 

class FMoWS2Dataset(Dataset):
    """
    用于 fMoW-S2 多光谱 (13 波段) 数据的自定义 Dataset。
    
    (V6 修复): 检查数值稳定性 (max - min < 1e-5)
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
            return torch.empty(0) 

        # (V3 修复) 检查 NaN/Inf
        if np.isnan(img).any() or np.isinf(img).any():
            if idx < 50: 
                print(f"警告：文件 {filepath} 包含 NaN/Inf，已跳过。")
            return torch.empty(0)

        # 检查通道数
        if img.shape[0] != 13:
            if idx < 50:
                 print(f"警告：文件 {filepath} 的通道数不是 13 (而是 {img.shape[0]})，已跳过。")
            return torch.empty(0)

        # 归一化
        img = np.clip(img / 10000.0, 0.0, 1.0)
        img_tensor = torch.from_numpy(img)
        
        # 填充
        c, h, w = img_tensor.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            padding = (0, pad_w, 0, pad_h) 
            img_tensor = F.pad(img_tensor, padding, mode='constant', value=0)
            
        # 裁剪
        c, h_padded, w_padded = img_tensor.shape
        top = torch.randint(0, h_padded - self.patch_size + 1, (1,)).item()
        left = torch.randint(0, w_padded - self.patch_size + 1, (1,)).item()
        
        patch = img_tensor[:, top : top + self.patch_size, left : left + self.patch_size]
        
        # --- (!!! 最终修复 V6 !!!) ---
        if (patch.max() - patch.min()) < MIN_SIGNAL_RANGE:
            if idx < 50: # 避免刷屏
                print(f"警告：文件 {filepath} 产生了数值不稳定 (范围 < {MIN_SIGNAL_RANGE}) patch，已跳过。")
            return torch.empty(0)
        # --- (修复结束) ---
        
        return patch

# collate_fn (无需修改)
def collate_fn(batch):
    batch = [b for b in batch if b.shape[0] > 0]
    if len(batch) == 0:
        return torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)