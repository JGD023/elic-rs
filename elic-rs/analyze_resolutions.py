# analyze_resolutions.py
import os
import glob
import sys
import argparse
import rasterio
from tqdm import tqdm
from collections import Counter

def analyze_dataset(data_dir, patch_size, target_channels):
    """
    (只读) 递归扫描目录并分析所有 .tif 文件的元数据。
    """
    print(f"--- 数据集分析开始 ---")
    print(f"目标目录: {data_dir}")
    print(f"检查标准: {target_channels} 通道, 最小分辨率 {patch_size}x{patch_size}")
    print("-" * 20)
    
    # 1. 递归查找所有 .tif 文件
    search_path = os.path.join(data_dir, "**", "*.tif")
    all_files = sorted(glob.glob(search_path, recursive=True))
    
    if not all_files:
        print(f"在 {data_dir} 中未找到 .tif 文件。")
        return

    print(f"找到 {len(all_files)} 个 .tif 文件，开始分析 (这可能需要几分钟)...")

    # 统计变量
    corrupt_files = 0
    wrong_channel_files = 0
    small_files = 0
    
    # 分辨率追踪
    min_h, min_w = float('inf'), float('inf')
    max_h, max_w = 0, 0
    min_res_file = ""
    max_res_file = ""
    
    # 使用 Counter 统计最常见的分辨率
    resolution_counter = Counter()

    # 2. 循环分析所有文件
    for filepath in tqdm(all_files, desc="分析中"):
        try:
            # 只打开文件头，不读取整个图像，速度很快
            with rasterio.open(filepath) as src:
                channels = src.count
                height = src.height
                width = src.width
            
            # 统计分辨率
            resolution_counter[(height, width)] += 1
            
            # --- 检查问题 ---
            if channels != target_channels:
                wrong_channel_files += 1
            
            if height < patch_size or width < patch_size:
                small_files += 1

            # --- 追踪最大/最小值 ---
            if height * width < min_h * min_w:
                min_h, min_w = height, width
                min_res_file = filepath
            
            if height * width > max_h * max_w:
                max_h, max_w = height, width
                max_res_file = filepath
                
        except Exception as e:
            # 标记损坏的或无法打开的文件
            corrupt_files += 1

    # 3. 打印总结
    print("\n" + "=" * 30)
    print("--- 分析报告 ---")
    print(f"总共扫描文件: {len(all_files)}")
    print("=" * 30)
    
    print("\n[问题文件统计]")
    print(f"  文件损坏或无法读取: {corrupt_files}")
    print(f"  通道数不是 {target_channels} 的文件: {wrong_channel_files}")
    print(f"  分辨率小于 {patch_size}x{patch_size} 的文件: {small_files}")
    
    print("\n[分辨率极值]")
    print(f"  最小分辨率 (H x W): {min_h} x {min_w}")
    print(f"     (文件: {min_res_file})")
    print(f"  最大分辨率 (H x W): {max_h} x {max_w}")
    print(f"     (文件: {max_res_file})")

    print("\n[Top 10 最常见的分辨率 (H x W)]")
    for (h, w), count in resolution_counter.most_common(10):
        print(f"  - {h} x {w}: {count} 个文件")

    print("\n--- 分析完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 fMoW-S2 数据集的分辨率")
    parser.add_argument(
        "data_dir", 
        type=str, 
        help="要分析的数据目录 (例如: ./data/fmow-sentinel/train)"
    )
    parser.add_argument(
        "--patch_size", 
        type=int, 
        default=256, 
        help="用于对比的最小分辨率 (patch_size)"
    )
    parser.add_argument(
        "--channels", 
        type=int, 
        default=13, 
        help="用于对比的目标通道数"
    )
    
    args = parser.parse_args()
    
    analyze_dataset(args.data_dir, args.patch_size, args.channels)