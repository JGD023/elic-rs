import torch

# 您出问题的checkpoint路径
checkpoint_path = './checkpoints/elic2022_ms_lmbda_0.3_epoch_5.pth.tar'

try:
    data = torch.load(checkpoint_path, map_location='cpu')
    
    # 假设权重保存在 'state_dict' 键下
    # 如果不是，您可能需要调整 'state_dict' 为正确的键名，或者直接遍历 data
    if 'state_dict' in data:
        weights = data['state_dict']
    else:
        print("未找到 'state_dict'，将尝试检查整个文件...")
        weights = data

    has_nan = False
    for k, v in weights.items():
        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
            print(f"!!! 在键 '{k}' 中发现 NaN !!!")
            has_nan = True

    if has_nan:
        print("\n[结论]: 您的checkpoint文件已损坏，包含 NaN。")
    else:
        print("\n[结论]: 您的checkpoint文件中未直接发现 NaN。")

except Exception as e:
    print(f"加载或检查checkpoint时出错: {e}")