# visualize_enhanced.py
#
# V3 版：
# 1. 修复了中文乱码问题
# 2. (新功能) 添加了“百分位对比度拉伸”，以解决卫星图像太暗的问题

import argparse
import sys
import os
import torch
import torch.nn.functional as F
import rasterio
import numpy as np
from collections import OrderedDict 

# --- (关键修改) 导入 matplotlib 并设置中文字体 ---
try:
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'sans-serif'] 
    plt.rcParams['axes.unicode_minus'] = False 
    print("已尝试应用中文字体设置...")

except ImportError:
    print("错误：未找到 'matplotlib' 库。")
    print("请先运行: pip install matplotlib")
    sys.exit(1)
# --- (修改结束) ---


# --- 自定义模块导入 ---
try:
    from src.ms_models import Elic2022MultiSpectral
except ImportError:
    print("错误：无法导入 'src.ms_models.Elic2022MultiSpectral'。")
    print("请确保您在 'elic-rs/' 根目录下运行此脚本。")
    sys.exit(1)

# --- 辅助函数 ---

def pad_to_multiple(x, factor=64):
    h, w = x.size(2), x.size(3)
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x, h, w 

def select_true_color_bands(img_tensor_13_channel):
    true_color_indices = [3, 2, 1] 
    if img_tensor_13_channel.size(0) < 4:
        print(f"警告: 图像通道数 ({img_tensor_13_channel.size(0)}) 不足 13，将尝试使用 0,1,2")
        true_color_indices = [0, 1, 2] 
        if img_tensor_13_channel.size(0) < 3:
             return img_tensor_13_channel[[0,0,0], :, :] 
    return img_tensor_13_channel[true_color_indices, :, :]

def load_image_tensor(filepath):
    try:
        with rasterio.open(filepath) as src:
            img = src.read().astype(np.float32)
            # 注意：我们在这里不再除以 10000，保留原始值
            img_tensor = torch.from_numpy(img)
            return img_tensor, filepath
    except Exception as e:
        print(f"错误：读取文件 {filepath} 失败: {e}")
        return None, filepath

# --- (关键修改) 应用对比度拉伸 ---
def tensor_to_displayable_image(img_tensor_3channel):
    """
    将 (C, H, W) 的 PyTorch 张量转换为 (H, W, C) 的
    Numpy 数组，并应用 2%-98% 的百分位拉伸。
    """
    img_np = img_tensor_3channel.cpu().numpy() # (C, H, W)
    
    # 准备 (H, W, C) 格式用于显示
    img_display = np.zeros((img_np.shape[1], img_np.shape[2], 3), dtype=np.float32)

    # 对 R, G, B 三个通道分别进行拉伸
    for i in range(3):
        band = img_np[i, :, :]
        # 计算 2% 和 98% 的值
        p2, p98 = np.percentile(band, [2, 98])
        
        # 应用拉伸
        stretched_band = (band - p2) / (p98 - p2)
        
        # 裁剪到 [0, 1]
        stretched_band = np.clip(stretched_band, 0, 1)
        
        img_display[:, :, i] = stretched_band

    return img_display
# --- (修改结束) ---

# --- 主函数 ---
def main(argv):
    parser = argparse.ArgumentParser(description="Visualize Original vs. Reconstructed Multi-Spectral Image (Enhanced Contrast).")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the .tif image")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    args = parser.parse_args(argv)

    if not os.path.exists(args.checkpoint): sys.exit(f"错误：检查点不存在: {args.checkpoint}")
    if not os.path.exists(args.input): sys.exit(f"错误：输入图像不存在: {args.input}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1. 加载模型
    print("加载模型: Elic2022MultiSpectral...")
    net = Elic2022MultiSpectral()
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

    # 2. 加载图像 (保留原始值)
    print(f"加载原图: {args.input}")
    img_tensor, _ = load_image_tensor(args.input) 
    if img_tensor is None: sys.exit("无法加载图像。")
    
    # --- (关键修改) 将原始值归一化以输入模型 ---
    # 模型是用 [0, 1] (除以10000) 的数据训练的
    x_normalized = torch.clip(img_tensor / 10000.0, 0.0, 1.0).unsqueeze(0).to(device)
    # --- (修改结束) ---
    
    # 3. 运行压缩和解压缩
    print("正在对图像进行编码和解码...")
    with torch.no_grad():
        x_padded, original_h, original_w = pad_to_multiple(x_normalized, 64)
        out_net = net(x_padded)
        x_hat_padded = out_net['x_hat']
        x_hat_normalized = x_hat_padded[..., :original_h, :original_w] # (B, C, H, W)
    
    print("处理完成。")
    
    # 4. 准备可视化 (提取真彩色通道)
    print("提取真彩色(RGB)通道用于显示...")
    # (修改) 我们使用未归一化的原始张量进行可视化
    original_rgb_tensor = select_true_color_bands(img_tensor) 
    # (修改) 重建图需要乘以 10000，以匹配原始数据的动态范围
    reconstructed_rgb_tensor = select_true_color_bands(x_hat_normalized.squeeze(0)) * 10000.0

    # 5. 显示图像
    print("正在生成对比图...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"模型: {os.path.basename(args.checkpoint)}\n图像: {os.path.basename(args.input)}", fontsize=16)

    # (修改) 现在使用新的拉伸函数来显示
    ax1.imshow(tensor_to_displayable_image(original_rgb_tensor))
    ax1.set_title("原图 (对比度拉伸)", fontsize=14)
    ax1.axis('off')

    ax2.imshow(tensor_to_displayable_image(reconstructed_rgb_tensor))
    ax2.set_title("重建图 (对比度拉伸)", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    
    output_filename = "visualization_compare_enhanced.png" # 新的文件名
    plt.savefig(output_filename)
    print(f"对比图已保存到: {output_filename}")


if __name__ == "__main__":
    main(sys.argv[1:])