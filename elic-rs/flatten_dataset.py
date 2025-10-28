import os
import shutil
from pathlib import Path

# --- 配置 ---

# 1. 设置数据集的根目录
# (假设此脚本与 'data' 文件夹在同一级)
# 否则，请在此处指定 'data/fmow-sentinel' 的完整路径
base_dir = Path("data/fmow-sentinel")

# 2. 设置要处理的数据集划分
splits_to_process = ["train", "val", "test_gt"]

# 3. 安全开关 (True = 只打印，不执行 | False = 实际执行)
DRY_RUN = False

# --- 脚本 ---

if DRY_RUN:
    print("=" * 40)
    print("      *** 正在以 DRY_RUN 模式运行 ***")
    print("  (不会移动或删除任何文件，只会打印计划)")
    print("=" * 40)

# 确保基础目录存在
if not base_dir.is_dir():
    print(f"错误: 基础目录 '{base_dir}' 未找到。")
    print("请确保 'base_dir' 路径设置正确。")
    exit(1)

total_files_moved = 0
total_dirs_removed = 0

# 1. 遍历所有 'split' 文件夹 (train, val, test_gt)
for split in splits_to_process:
    split_path = base_dir / split
    if not split_path.is_dir():
        print(f"\n[跳过] '{split}' 目录不存在于: {base_dir}")
        continue
    
    print(f"\n--- 正在处理: {split_path} ---")

    # 2. 遍历所有 'category' 文件夹 (e.g., airport)
    for category_path in split_path.iterdir():
        if not category_path.is_dir():
            continue

        print(f"  [类别: {category_path.name}]")
        
        # 3. 遍历所有 'instance_id' 文件夹 (e.g., airport_0)
        for instance_path in category_path.iterdir():
            # 我们只处理 'instance' 文件夹，如果文件夹内还有文件（非目录），则跳过
            if not instance_path.is_dir():
                continue

            print(f"    - 正在压平: {instance_path.name}")
            
            # 4. 遍历 'instance' 文件夹中的所有文件
            for file_path in instance_path.iterdir():
                if file_path.is_file():
                    # 目标路径 = 上一级的 'category' 文件夹
                    dest_path = category_path / file_path.name

                    # 检查目标位置是否已存在同名文件
                    if dest_path.exists():
                        print(f"      [警告] 跳过 {file_path.name}，目标位置已存在同名文件。")
                        continue

                    # 移动文件
                    print(f"      -> 计划移动 {file_path.name} 到 {category_path.name}/")
                    if not DRY_RUN:
                        try:
                            shutil.move(file_path, dest_path)
                            total_files_moved += 1
                        except Exception as e:
                            print(f"      [错误] 移动 {file_path.name} 失败: {e}")

            # 5. 移动完所有文件后，尝试删除空的 'instance' 文件夹
            if not DRY_RUN:
                try:
                    instance_path.rmdir() # 只有当文件夹为空时才会成功
                    print(f"    - [删除] 已删除空文件夹: {instance_path.name}")
                    total_dirs_removed += 1
                except OSError:
                    # 如果文件夹非空 (例如因为同名文件跳过导致)，则不会删除
                    print(f"    - [跳过删除] {instance_path.name} 非空，未删除。")

print("\n" + "=" * 40)
if DRY_RUN:
    print("      *** DRY_RUN 模式运行结束 ***")
    print("  (没有文件被真正移动或删除)")
    print("  (如果输出符合预期，请将 DRY_RUN 设置为 False 并重新运行)")
else:
    print("      *** 脚本执行完毕 ***")
    print(f"  总共移动的文件数: {total_files_moved}")
    print(f"  总共删除的目录数: {total_dirs_removed}")
print("=" * 40)