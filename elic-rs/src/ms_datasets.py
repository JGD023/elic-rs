# elic-rs/src/ms_datasets.py
# (最终版 V7: 增加了对混合通道的“通道填充”支持)

import os
import glob
import torch
import torch.nn.functional as F
import rasterio
import numpy as np
from torch.utils.data import Dataset

MIN_SIGNAL_RANGE = 1e-5 

class FMoWS2Dataset(Dataset):
    """
    用于 fMoW-S2 多光谱数据的自定义 Dataset。
    
    (V7 修复): 
    - 检查数值稳定性 (max - min < 1e-5)
    - (新功能) 支持混合通道输入，自动将通道数 < 13 的图像
      用 0 填充到 13 通道。
    """
    
    # (新功能) 将最大通道数定义为类属性
    MAX_CHANNELS = 13 
    
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

        # 检查 NaN/Inf
        if np.isnan(img).any() or np.isinf(img).any():
            if idx < 50: 
                print(f"警告：文件 {filepath} 包含 NaN/Inf，已跳过。")
            return torch.empty(0)

        # --- (!!! 关键修改：通道填充逻辑 !!!) ---
        
        # 1. 检查通道数
        current_channels = img.shape[0]

        if current_channels > self.MAX_CHANNELS:
            # 如果图像通道数 > 13，跳过它
            if idx < 50:
                 print(f"警告：文件 {filepath} 的通道数 ({current_channels}) > {self.MAX_CHANNELS}，已跳过。")
            return torch.empty(0)

        # 2. 归一化 (在填充前)
        img = np.clip(img / 10000.0, 0.0, 1.0)

        # 3. (新) 如果通道数 < 13，则进行填充
        if current_channels < self.MAX_CHANNELS:
            h, w = img.shape[1], img.shape[2]
            # 创建一个 (N, H, W) 的 0 填充张量
            padding = np.zeros((self.MAX_CHANNELS - current_channels, h, w), dtype=np.float32)
            # 在通道维度 (axis=0) 上连接
            img = np.concatenate((img, padding), axis=0)
            
        # --- (修改结束) ---

        # 此刻，'img' 变量保证是 (13, H, W) 的 numpy 数组
        
        img_tensor = torch.from_numpy(img)
        
        # 填充 (空间维度)
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
        
        # 检查数值稳定性
        if (patch.max() - patch.min()) < MIN_SIGNAL_RANGE:
            if idx < 50: # 避免刷屏
                print(f"警告：文件 {filepath} 产生了数值不稳定 (范围 < {MIN_SIGNAL_RANGE}) patch，已跳过。")
            return torch.empty(0)
        
        return patch

# collate_fn (无需修改)
def collate_fn(batch):
    batch = [b for b in batch if b.shape[0] > 0]
    if len(batch) == 0:
        return torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)