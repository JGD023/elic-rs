# visualize.py
#
# 描述:
#   加载一个训练好的 Elic2022MultiSpectral 模型，
#   读取一个 13 通道的 .tif 图像，
#   对其进行压缩和解压缩，
#   最后并排显示原图和重建图的“真彩色”(RGB)
#   视图，以便进行视觉比较。
#
# (V2 版：添加了中文字体支持)

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
    # 告知 matplotlib 在没有图形界面的服务器上也能运行
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt

    # 尝试设置一个通用的 CJK 字体列表
    # (请确保您已在服务器上安装了这些字体, e.g., sudo apt install fonts-noto-cjk)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'sans-serif'] 
    # 解决中文环境下负号显示为方块的问题
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

# --- 辅助函数 (pad_to_multiple, select_true_color_bands, load_image_tensor, tensor_to_displayable_image) ---

def pad_to_multiple(x, factor=64):
    """将图像 x (B, C, H, W) 填充到 H 和 W 是 factor 的倍数。"""
    h, w = x.size(2), x.size(3)
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x, h, w # 返回原始 H, W 用于裁剪

def select_true_color_bands(img_tensor_13_channel):
    """从 13 通道 fMoW/Sentinel-2 图像中提取真彩色 (RGB) 通道。"""
    true_color_indices = [3, 2, 1] 
    if img_tensor_13_channel.size(0) < 4:
        print(f"警告: 图像通道数 ({img_tensor_13_channel.size(0)}) 不足 13，将尝试使用 0,1,2")
        true_color_indices = [0, 1, 2] 
        if img_tensor_13_channel.size(0) < 3:
             return img_tensor_13_channel[[0,0,0], :, :] 
    return img_tensor_13_channel[true_color_indices, :, :]

def load_image_tensor(filepath):
    """从 .tif 文件加载并预处理图像。"""
    try:
        with rasterio.open(filepath) as src:
            img = src.read().astype(np.float32)
            img = np.clip(img / 10000.0, 0.0, 1.0)
            img_tensor = torch.from_numpy(img)
            return img_tensor, filepath
    except Exception as e:
        print(f"错误：读取文件 {filepath} 失败: {e}")
        return None, filepath

def tensor_to_displayable_image(img_tensor):
    """(C, H, W) -> (H, W, C) for matplotlib"""
    return img_tensor.cpu().numpy().transpose(1, 2, 0).clip(0, 1)

# --- 主函数 ---
def main(argv):
    parser = argparse.ArgumentParser(description="Visualize Original vs. Reconstructed Multi-Spectral Image.")
    parser.add_argument(
        "-c", "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint (.pth.tar)"
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="Path to the single .tif image file to visualize"
    )
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
        if k.startswith('module.'): 
            new_state_dict[k[7:]] = v; is_ddp = True
        else: 
            new_state_dict[k] = v
    if is_ddp: print("检测到 DDP 检查点，已处理。")
    net.load_state_dict(new_state_dict, strict=True)
    print("权重加载成功。")
    
    print("更新熵编码器...")
    net.update(force=True)
    print("更新完成。")
    net.eval()

    # 2. 加载并处理单张图像
    print(f"加载原图: {args.input}")
    img_tensor, _ = load_image_tensor(args.input) 
    if img_tensor is None:
        sys.exit("无法加载图像。")
    
    x = img_tensor.unsqueeze(0).to(device) # (B, C, H, W)
    
    # 3. 运行压缩和解压缩
    print("正在对图像进行编码和解码...")
    with torch.no_grad():
        x_padded, original_h, original_w = pad_to_multiple(x, 64)
        out_net = net(x_padded)
        x_hat_padded = out_net['x_hat']
        x_hat = x_hat_padded[..., :original_h, :original_w] # (B, C, H, W)
    
    print("处理完成。")
    
    # 4. 准备可视化 (提取真彩色通道)
    print("提取真彩色(RGB)通道用于显示...")
    original_rgb_tensor = select_true_color_bands(x.squeeze(0))
    reconstructed_rgb_tensor = select_true_color_bands(x_hat.squeeze(0))

    # 5. 显示图像
    print("正在生成对比图...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"模型: {os.path.basename(args.checkpoint)}\n图像: {os.path.basename(args.input)}", fontsize=16)

    # 显示原图
    ax1.imshow(tensor_to_displayable_image(original_rgb_tensor))
    ax1.set_title("原图 (真彩色)", fontsize=14)
    ax1.axis('off')

    # 显示重建图
    ax2.imshow(tensor_to_displayable_image(reconstructed_rgb_tensor))
    ax2.set_title("重建图 (真彩色)", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # (关键修改) 移除 plt.show()，它在服务器上会失败
    # plt.show() 
    
    output_filename = "visualization_compare.png"
    plt.savefig(output_filename)
    print(f"对比图已保存到: {output_filename}")


if __name__ == "__main__":
    main(sys.argv[1:])