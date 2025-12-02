# elic-rs/check_model_stats.py
import torch
from torchinfo import summary
import sys

# 1. 导入您的自定义模型
try:
    from src.ms_models import Elic2022MultiSpectral
except ImportError:
    print("错误：无法导入 'src.ms_models.Elic2022MultiSpectral'。")
    print("请确保您在 'elic-rs/' 根目录下运行此脚本。")
    sys.exit(1)

# 2. 定义模型的输入形状
INPUT_CHANNELS = 13      # 您的最大通道数
PATCH_SIZE = 256       # 您的 patch_size
BATCH_SIZE = 1         # (用于分析的标准批大小)

# 3. 实例化模型
try:
    # (!!! 关键修改: 传入通道数 !!!)
    model = Elic2022MultiSpectral(C_in_out=INPUT_CHANNELS)
    model.eval() # 切换到评估模式

    # 4. 定义完整的输入尺寸
    input_size = (BATCH_SIZE, INPUT_CHANNELS, PATCH_SIZE, PATCH_SIZE)

    # 5. 运行并打印模型摘要
    print(f"--- Elic2022MultiSpectral ({INPUT_CHANNELS}通道) 模型分析 ---")
    print(f"--- 输入尺寸: {input_size} ---")
    
    summary(
        model, 
        input_size=input_size, 
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        verbose=1 
    )

except Exception as e:
    print(f"计算模型参数时发生错误: {e}")