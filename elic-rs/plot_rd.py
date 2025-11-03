import matplotlib.pyplot as plt

# --- 您的数据 ---
# 这是从您的截图中提取的数据
# X 轴: BPP (Bits Per Pixel)
bpp_values = [0.114009, 0.334127, 0.439690, 0.487400, 0.894774, 1.126258]

# Y 轴 1: PSNR (dB)
psnr_values = [35.8327, 39.9366, 40.2640, 41.2560, 42.9070, 44.0640]

# Y 轴 2: MS-SSIM (dB)
msssim_db_values = [15.13045, 19.02051, 19.74394, 20.55219, 22.48093, 23.78509]

# --- 开始绘图 ---

# 创建一个包含两个子图的图窗 (1行, 2列)
# figsize=(12, 5) 设置图窗的整体大小
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- 子图 1: BPP vs PSNR ---
ax1.plot(bpp_values, psnr_values, marker='o', linestyle='-', color='b', label='ELIC 2022 (PSNR)')
ax1.set_title('Rate-Distortion Curve (BPP vs. PSNR)', fontsize=14)
ax1.set_xlabel('BPP (bits per pixel)', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# --- 子图 2: BPP vs MS-SSIM (dB) ---
ax2.plot(bpp_values, msssim_db_values, marker='s', linestyle='-', color='r', label='ELIC 2022 (MS-SSIM)')
ax2.set_title('Rate-Distortion Curve (BPP vs. MS-SSIM)', fontsize=14)
ax2.set_xlabel('BPP (bits per pixel)', fontsize=12)
ax2.set_ylabel('MS-SSIM (dB)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# 自动调整子图布局，防止标签重叠
plt.tight_layout()

# 显示图表
plt.show()

# 如果您想将图表保存为文件（例如 PNG）:
fig.savefig('rd_curves.png', dpi=300)