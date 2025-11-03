import torch
from torchinfo import summary
import sys

# 1. 导入您的自定义模型
#    我们假设此脚本在项目的根目录下运行
try:
    from src.ms_models import Elic2022MultiSpectral
except ImportError:
    print("错误：无法导入 'src.ms_models.Elic2022MultiSpectral'。")
    print("请确保您在 'elic-rs/' 根目录下运行此脚本。")
    sys.exit(1)

# 2. 定义模型的输入形状
#    (基于您的训练脚本)
INPUT_CHANNELS = 13      # 您的通道数
PATCH_SIZE = 256       # 您的 patch_size
BATCH_SIZE = 1         # (用于分析的标准批大小)

# 3. 实例化模型
try:
    model = Elic2022MultiSpectral()
    model.eval() # 切换到评估模式

    # 4. 定义完整的输入尺寸
    #    (batch_size, channels, height, width)
    input_size = (BATCH_SIZE, INPUT_CHANNELS, PATCH_SIZE, PATCH_SIZE)

    # 5. 运行并打印模型摘要
    print(f"--- Elic2022MultiSpectral (13通道) 模型分析 ---")
    print(f"--- 输入尺寸: {input_size} ---")
    
    # 运行 summary
    # col_names 指定了我们想看的列：
    # "input_size": 输入尺寸
    # "output_size": 输出尺寸
    # "num_params": 参数量
    # "mult_adds": 乘加运算次数 (即 MACs, 通常等同于 FLOPs)
    summary(
        model, 
        input_size=input_size, 
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        verbose=1 # 0 只显示总结, 1 显示完整表格
    )

except Exception as e:
    print(f"计算模型参数时发生错误: {e}")
    print("这可能是因为模型中的某些自定义层（如 GDN）与 torchinfo 不兼容。")