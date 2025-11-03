# src/ms_models.py
# (V0 原始版 - 因为 V9 禁用了 AMP，不再需要 V8 修复)

import torch.nn as nn
from compressai.models import Elic2022Official
from compressai.layers import GDN
from compressai.models.utils import conv, deconv

class Elic2022MultiSpectral(Elic2022Official):
    """
    继承 Elic2022Official，将其修改为支持 13 通道输入和 13 通道输出。

    我们必须覆盖 g_a (编码器) 和 g_s (解码器)。
    """
    def __init__(self, N=192, M=320, groups=None, **kwargs):
        # 1. 调用父类的构造函数
        super().__init__(N=N, M=M, groups=groups, **kwargs)
        
        # 2. 定义输入/输出通道数
        self.C_in_out = 13

        # 3. 覆盖编码器 (g_a)
        # 原始 g_a 的第一层是 conv(3, N, ...)，我们改为 conv(13, N, ...)
        # 我们必须完整复制 Elic2022Official (继承自 Cheng2020Attention) 的 g_a 结构
        self.g_a = nn.Sequential(
            conv(self.C_in_out, N, kernel_size=5, stride=2), # <-- 修改点
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        # 4. 覆盖解码器 (g_s)
        # 原始 g_s 的最后一层是 deconv(N, 3, ...)，我们改为 deconv(N, 13, ...)
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, self.C_in_out, kernel_size=5, stride=2), # <-- 修改点
        )

    # forward 函数、aux_loss 函数等均继承自父类，无需修改