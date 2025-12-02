# elic-rs/src/ms_models.py
# (最终版：支持任意通道数)

import torch.nn as nn
from compressai.models import Elic2022Official
from compressai.layers import GDN
from compressai.models.utils import conv, deconv

class Elic2022MultiSpectral(Elic2022Official):
    """
    继承 Elic2022Official，将其修改为支持任意通道数 (C_in_out) 的输入和输出。
    默认为 13。

    我们必须覆盖 g_a (编码器) 和 g_s (解码器)。
    """
    
    # --- (关键修改) ---
    # 将 C_in_out 添加到构造函数参数中，并设置默认值为 13
    def __init__(self, C_in_out=13, N=192, M=320, groups=None, **kwargs):
        # 1. 调用父类的构造函数
        super().__init__(N=N, M=M, groups=groups, **kwargs)
        
        # 2. 从参数中获取输入/输出通道数
        self.C_in_out = C_in_out # <-- 不再硬编码
        # --- (修改结束) ---

        # 3. 覆盖编码器 (g_a)
        # 原始 g_a 的第一层是 conv(3, N, ...)，我们改为 conv(self.C_in_out, N, ...)
        # (此处的 self.C_in_out 将使用构造函数传入的值)
        self.g_a = nn.Sequential(
            conv(self.C_in_out, N, kernel_size=5, stride=2), 
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        # 4. 覆盖解码器 (g_s)
        # 原始 g_s 的最后一层是 deconv(N, 3, ...)，我们改为 deconv(N, self.C_in_out, ...)
        # (此处的 self.C_in_out 将使用构造函数传入的值)
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, self.C_in_out, kernel_size=5, stride=2),
        )

    # forward 函数、aux_loss 函数等均继承自父类，无需修改