from huggingface_hub import snapshot_download
import os

# Hugging Face 数据集的 ID
repo_id = "embed2scale/SSL4EO-S12-v1.1"

# 您想要下载的特定目录
# 我们使用通配符 '*' 来匹配 'S2L2A' 目录下的所有文件
allow_pattern = "train/S2L2A/*"

# 您想将数据保存在本地的哪个文件夹
local_directory = "./my_ssl4eo_dataset"

print(f"正在从 {repo_id} 下载 {allow_pattern}...")
print(f"将保存到: {os.path.abspath(local_directory)}")

# 开始下载
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=allow_pattern,
    local_dir=local_directory,
    local_dir_use_symlinks=False  # 建议设为 False，以复制文件
)

print("下载完成！")