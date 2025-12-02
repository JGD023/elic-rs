import matplotlib.pyplot as plt

# --- 您的数据 ---
# (!!!) 重要 (!!!)
# 您必须在此处手动填入 evaluate.py 输出的 *bppbf* 值
# X 轴: BPPBF (Bits Per Pixel Per Band Frame)
bppbf_values = [0.008769, 0.025702, 0.037492, 0.068828, 0.086635] # <-- 示例数据 (即您的 bpp / 13)

# Y 轴 1: PSNR (dB)
psnr_values = [35.8327, 39.9366, 41.2560, 42.9070, 44.0640] # <-- 示例数据

# Y 轴 2: MS-SSIM (dB)
msssim_db_values = [15.13045, 19.02051, 20.55219, 22.48093, 23.78509] # <-- 示例数据

# --- 开始绘图 ---

# 创建一个包含两个子图的图窗 (1行, 2列)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- 子图 1: BPPBF vs PSNR ---
ax1.plot(bppbf_values, psnr_values, marker='o', linestyle='-', color='b', label='ELIC-MS (PSNR)')
ax1.set_title('Rate-Distortion Curve (BPPBF vs. PSNR)', fontsize=14)
ax1.set_xlabel('Bits-per-pixel-band-frame (bppbf)', fontsize=12) # <-- 修改
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# --- 子图 2: BPPBF vs MS-SSIM (dB) ---
ax2.plot(bppbf_values, msssim_db_values, marker='s', linestyle='-', color='r', label='ELIC-MS (MS-SSIM)')
ax2.set_title('Rate-Distortion Curve (BPPBF vs. MS-SSIM)', fontsize=14)
ax2.set_xlabel('BPPBF (bits per pixel per band)', fontsize=12) # <-- 修改
ax2.set_ylabel('MS-SSIM (dB)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()

# (可选) 在服务器上运行时，取消注释以下行以保存文件
output_filename = "rd_curves_bppbf.png"
fig.savefig(output_filename, dpi=300)
print(f"RD 曲线图已保存到: {output_filename}")

# 在本地运行时，使用 plt.show()
plt.show()