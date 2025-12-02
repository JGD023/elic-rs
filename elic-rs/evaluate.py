# elic-rs/evaluate.py
# (最终版 V5: 支持混合通道的“真实指标”评估)
#
# 1. (新) Dataset 返回 (padded_image, original_channels, filepath)
# 2. (新) Collate_fn 适配
# 3. (新) main 函数在计算 PSNR/MS-SSIM 前对 x 和 x_hat 进行切片
# 4. (新) BPP 仍然在完整的码流上计算 (真实成本)
# 5. (新) 实例化模型时使用 (C_in_out=13)

import argparse
import math
import sys
import os
import glob
import torch
import torch.nn.functional as F
import rasterio
import numpy as np # <-- (新) 导入 numpy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict 
import traceback 

# --- (新功能) 导入 MS-SSIM 库 ---
try:
    from pytorch_msssim import ms_ssim
except ImportError:
    print("错误：未找到 'pytorch-msssim' 库。")
    print("请先运行: pip install pytorch-msssim")
    sys.exit(1)
# --- (添加结束) ---

# --- 自定义模块导入 ---
try:
    # (新) 导入我们灵活的模型
    from src.ms_models import Elic2022MultiSpectral
except ImportError:
    print("错误：无法导入 'src.ms_models.Elic2022MultiSpectral'。")
    print("请确保您在 'elic-rs/' 根目录下运行此脚本。")
    sys.exit(1)

# --- 辅助函数 (compute_psnr, compute_bpp 无需修改) ---
def compute_psnr(a, b, data_range=1.0):
    # (此函数无需修改，因为它会在切片后的张量上计算)
    mse = torch.mean((a - b)**2).item()
    if mse == 0: return float('inf')
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def compute_bpp(out_net, num_pixels):
    # (此函数无需修改，BPP 必须基于完整的 13 通道码流计算)
    if 'likelihoods' not in out_net or not out_net['likelihoods']:
        print("\n[警告] 模型前向传播输出缺少 'likelihoods'，无法计算 BPP。")
        return float('nan')
    likelihoods = out_net['likelihoods']
    bpp = sum( (torch.log(likelihoods[k]).sum() / (-math.log(2) * num_pixels)) for k in likelihoods )
    return bpp.item()

# --- (新功能) MS-SSIM 计算辅助函数 ---
def compute_msssim(a, b, data_range=1.0):
    """计算 a 和 b 之间的 MS-SSIM 值 (0-1)"""
    # (此函数无需修改)
    # 它会基于传入的张量 a (已被切片) 的通道数 a.size(1) 来循环
    
    total_msssim = 0.0
    num_channels = a.size(1) 
    
    if num_channels == 0:
        return 0.0 # 避免除以 0
        
    for c in range(num_channels):
        channel_a = a[:, c:c+1, :, :] 
        channel_b = b[:, c:c+1, :, :]
        total_msssim += ms_ssim(channel_a, channel_b, data_range=data_range, size_average=False).item()
        
    return total_msssim / num_channels 

def compute_msssim_db(msssim_val):
    """将 MS-SSIM 值 (0-1) 转换为 dB"""
    # (此函数无需修改)
    if msssim_val >= 1.0: 
        return float('inf')
    return -10 * math.log10(1 - msssim_val)
# --- (添加结束) ---


# --- (!!! 关键修改: 评估数据集 !!!) ---
class EvalDatasetCenterCrop(Dataset):
    """
    (V5 修改)
    1. 支持混合通道，填充到 13
    2. 返回 (padded_crop, original_channels, filepath)
    """
    MAX_CHANNELS = 13 # (新) 定义最大通道
    
    def __init__(self, root, crop_size=256):
        self.root = root
        self.crop_size = crop_size
        search_path = os.path.join(self.root, "**", "*.tif")
        print(f"正在 {self.root} 目录中递归搜索 *.tif 文件...")
        self.image_files = sorted(glob.glob(search_path, recursive=True))
        if not self.image_files: raise RuntimeError(f"未找到 .tif 文件")
        print(f"找到 {len(self.image_files)} 个评估图像文件 (将使用 {crop_size}x{crop_size} 中心裁剪)。")
        
    def __len__(self): return len(self.image_files)
    
    def __getitem__(self, idx):
        filepath = self.image_files[idx]
        try:
            with rasterio.open(filepath) as src:
                # (新) 1. 获取原始通道数
                original_channels = src.count
                if original_channels > self.MAX_CHANNELS:
                    print(f"\n[警告] 图像 {filepath} 通道数 ({original_channels}) > {self.MAX_CHANNELS}，已跳过。")
                    return None, None, filepath
                    
                if src.height < self.crop_size or src.width < self.crop_size:
                    # print(f"\n[警告] 图像 {filepath} ({src.height}x{src.width}) 小于裁剪尺寸 {self.crop_size}x{self.crop_size}，已跳过。")
                    return None, None, filepath 
                
                img = src.read().astype(np.float32)
                
            # (新) 2. 归一化
            img = np.clip(img / 10000.0, 0.0, 1.0)
            
            # (新) 3. 通道填充 (如果需要)
            if original_channels < self.MAX_CHANNELS:
                h, w = img.shape[1], img.shape[2]
                padding = np.zeros((self.MAX_CHANNELS - original_channels, h, w), dtype=np.float32)
                img = np.concatenate((img, padding), axis=0)

            # (新) 4. 转换为张量 (现在总是 13 通道)
            img_tensor = torch.from_numpy(img)
            
            # 5. 中心裁剪 (不变)
            c, h, w = img_tensor.shape
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
            crop = img_tensor[:, top : top + self.crop_size, left : left + self.crop_size]
            
            # (新) 6. 返回填充后的裁切块、原始通道数、路径
            return crop, original_channels, filepath
            
        except Exception as e:
            print(f"警告：读取文件 {filepath} 失败: {e}")
            # (新) 返回三个 None
            return None, None, filepath
# --- (修改结束) ---


# --- (!!! 关键修改: Collate Fn !!!) ---
def eval_collate_fn(batch):
    # (新) 过滤 (img, ch, fp) 元组
    batch = [(img, ch, fp) for img, ch, fp in batch if img is not None]
    if not batch: 
        return None, None, None
        
    # (新) 分别打包
    images = torch.utils.data.dataloader.default_collate([item[0] for item in batch])
    original_channels = [item[1] for item in batch] # (新)
    filepaths = [item[2] for item in batch]
    
    return images, original_channels, filepaths # (新)
# --- (修改结束) ---


# --- 主评估函数 ---
def main(argv):
    parser = argparse.ArgumentParser(description="Evaluate a multi-spectral ELIC model using Center Crop (V5 - True Metrics).")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--crop-size", type=int, default=256, help="Center crop size (default: 256)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.checkpoint): sys.exit(f"错误：检查点不存在: {args.checkpoint}")
    if not os.path.isdir(args.dataset): sys.exit(f"错误：数据集目录不存在: {args.dataset}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1. 加载模型
    print("加载模型: Elic2022MultiSpectral (C_in_out=13)")
    # (!!! 关键修改: 实例化为 13 通道 !!!)
    net = Elic2022MultiSpectral(C_in_out=13) 
    net = net.to(device)
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    is_ddp = False
    for k, v in state_dict.items():
        if k.startswith('module.'): new_state_dict[k[7:]] = v; is_ddp = True
        else: new_state_dict[k] = v
    if is_ddp: print("检测到 DDP 检查点，已处理。")
    net.load_state_dict(new_state_dict, strict=True) 
    print("权重加载成功。")
    print("更新熵编码器...")
    net.update(force=True)
    print("更新完成。")
    net.eval()

    # 2. 准备数据集 (使用新的 collate_fn)
    eval_dataset = EvalDatasetCenterCrop(root=args.dataset, crop_size=args.crop_size)
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=1, # (!!! 关键: 必须为 1 !!!)
        num_workers=args.workers, 
        shuffle=False, 
        pin_memory=(args.cuda), 
        collate_fn=eval_collate_fn # (新)
    )

    # 3. 运行评估
    total_psnr = 0.0
    total_bpp = 0.0
    total_msssim_db = 0.0 
    num_images = 0
    skipped_count = 0 
    print("\n开始评估 (使用中心裁剪 - 真实指标)...")
    with torch.no_grad():
        pbar = tqdm(eval_dataloader, desc="评估中")
        
        # (!!! 关键修改: 循环 !!!)
        for images, original_channel_counts, filepaths in pbar:
            if images is None: continue 
            
            x = images.to(device) # x 是 (1, 13, H, W)
            
            # (不变) BPP 的 num_pixels 总是基于空间维度 H*W
            # BPP 是压缩 13 通道的真实成本
            num_pixels = x.size(0) * x.size(2) * x.size(3) # B*H*W
            
            try:
                # (不变) 模型在 13 通道上运行
                out_net = net(x) 
                x_hat = out_net['x_hat'] # x_hat 是 (1, 13, H, W)
                
                # (不变) BPP 在 13 通道上计算
                current_bpp = compute_bpp(out_net, num_pixels)
                
                # --- (!!! 关键修改: 切片 !!!) ---
                # 获取这张图像的原始通道数 (因为 batch_size=1)
                current_ch = original_channel_counts[0] 
                
                # 只选择 0 到 current_ch
                x_sliced = x[:, :current_ch, :, :]
                x_hat_sliced = x_hat[:, :current_ch, :, :]
                # --- (切片结束) ---
                
                # (新) 在切片后的张量上计算指标
                current_psnr = compute_psnr(x_sliced, x_hat_sliced.clamp(0, 1), data_range=1.0) 
                current_msssim_val = compute_msssim(x_sliced, x_hat_sliced.clamp(0, 1), data_range=1.0)
                current_msssim_db = compute_msssim_db(current_msssim_val)

                if math.isnan(current_bpp) or math.isinf(current_psnr) or math.isinf(current_msssim_db):
                    if math.isnan(current_bpp): reason = "BPP NaN"
                    elif math.isinf(current_psnr): reason = "PSNR Inf"
                    else: reason = "MS-SSIM Inf"
                    print(f"\n警告：图像 {filepaths[0]} 结果无效 ({reason})，跳过统计。")
                    skipped_count += 1
                    continue 

                total_bpp += current_bpp
                total_psnr += current_psnr
                total_msssim_db += current_msssim_db 
                num_images += 1
                
                # (不变) 更新进度条
                pbar.set_postfix(
                    avg_bpp=f'{total_bpp / num_images:.4f}',
                    avg_psnr=f'{total_psnr / num_images:.2f} dB',
                    avg_msssim=f'{total_msssim_db / num_images:.3f} dB' 
                )
                # --- (添加结束) ---

            except Exception as e:
                print(f"\n处理文件 {filepaths[0]} 时发生错误: ")
                print(f"  错误类型: {type(e)}")
                print(f"  错误信息: {e}")
                # print(traceback.format_exc()) 
                print("跳过此文件。")
                skipped_count += 1

    # 4. 打印结果 (不变)
    print("\n" + "=" * 30)
    print("--- 评估完成 (中心裁剪 - 真实指标) ---")
    if num_images == 0: print("错误：未能成功处理任何图像。")
    else:
        avg_bpp = total_bpp / num_images
        avg_psnr = total_psnr / num_images
        avg_msssim_db = total_msssim_db / num_images 
        
        print(f"在 {num_images} 张图像上评估完成。")
        if skipped_count > 0:
             print(f"  (另有 {skipped_count} 张图像因错误或尺寸过小而被跳过)")
        print(f"模型检查点: {args.checkpoint}")
        print(f"数据集: {args.dataset}")
        print(f"裁剪尺寸: {args.crop_size}x{args.crop_size}")
        print("-" * 30)
        print(f"平均 BPP : {avg_bpp:.6f} bpp")
        print(f"平均 PSNR: {avg_psnr:.4f} dB")
        print(f"平均 MS-SSIM (dB): {avg_msssim_db:.5f} dB")
    print("=" * 30)

if __name__ == "__main__":
    main(sys.argv[1:])