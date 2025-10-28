# clean_dataset.py
import os
import glob
import sys
import argparse
import rasterio
from tqdm import tqdm

def clean_dataset(data_dir, min_res, target_channels, auto_yes=False):
    """
    递归扫描目录并删除不符合要求的文件。
    """
    print(f"--- 数据清理开始 ---")
    print(f"目标目录: {data_dir}")
    print(f"要求: {target_channels} 个通道, 最小分辨率 {min_res}x{min_res}")
    print("-" * 20)
    
    # 1. 递归查找所有 .tif 文件
    search_path = os.path.join(data_dir, "**", "*.tif")
    all_files = sorted(glob.glob(search_path, recursive=True))
    
    if not all_files:
        print(f"在 {data_dir} 中未找到 .tif 文件。")
        return

    print(f"找到 {len(all_files)} 个 .tif 文件，开始分析...")

    files_to_delete = []
    reasons = {
        "channel": [],
        "resolution": [],
        "corrupt": []
    }

    # 2. 循环分析所有文件
    for filepath in tqdm(all_files, desc="分析中"):
        try:
            with rasterio.open(filepath) as src:
                channels = src.count
                height = src.height
                width = src.width
            
            # 检查通道数
            if channels != target_channels:
                reasons["channel"].append(filepath)
                files_to_delete.append(filepath)
            # 检查分辨率
            elif height < min_res or width < min_res:
                reasons["resolution"].append(filepath)
                files_to_delete.append(filepath)
                
        except Exception as e:
            # 标记损坏的或无法打开的文件
            print(f"\n[警告] 文件损坏或无法读取，将删除: {filepath}\n  错误: {e}")
            reasons["corrupt"].append(filepath)
            files_to_delete.append(filepath)

    # 3. 打印总结并请求确认
    print("\n--- 分析完成 ---")
    
    if not files_to_delete:
        print("恭喜！您的数据集非常干净，无需删除任何文件。")
        return

    print(f"总共找到 {len(files_to_delete)} 个需要删除的文件：")
    print(f"  - {len(reasons['channel'])} 个 (原因: 通道数不是 {target_channels})")
    print(f"  - {len(reasons['resolution'])} 个 (原因: 分辨率小于 {min_res}x{min_res})")
    print(f"  - {len(reasons['corrupt'])} 个 (原因: 文件损坏或无法读取)")
    
    print("\n" + "=" * 30)
    print(" 警告：文件删除是永久性的，无法撤销！")
    print("=" * 30 + "\n")

    if auto_yes:
        print("--yes 标志已提供，自动开始删除...")
        response = 'yes'
    else:
        # 请求用户确认
        response = input(f"您确定要永久删除这 {len(files_to_delete)} 个文件吗？ [请输入 'yes' 确认]: ")

    # 4. 执行删除
    if response.lower() == 'yes':
        print(f"\n正在删除 {len(files_to_delete)} 个文件...")
        for filepath in tqdm(files_to_delete, desc="删除中"):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"\n删除 {filepath} 失败: {e}")
        print("--- 清理完成 ---")
    else:
        print("操作已取消。没有文件被删除。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理 fMoW-S2 数据集")
    parser.add_argument(
        "data_dir", 
        type=str, 
        help="要清理的数据目录 (例如: ./data/fmow-sentinel/train)"
    )
    parser.add_argument(
        "--patch_size", 
        type=int, 
        default=256, 
        help="最小允许的分辨率 (H 和 W)，应与训练 patch_size 匹配"
    )
    parser.add_argument(
        "--channels", 
        type=int, 
        default=13, 
        help="必需的通道数"
    )
    parser.add_argument(
        "--yes", 
        action="store_true", 
        help="跳过确认提示，立即删除文件"
    )
    
    args = parser.parse_args()
    
    clean_dataset(args.data_dir, args.patch_size, args.channels, args.yes)