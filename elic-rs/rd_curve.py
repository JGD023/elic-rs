import matplotlib.pyplot as plt

# ==========================================
# 1. 数据录入 (来自于你提供的图片)
# ==========================================

# Dataset A: ELIC (Remote Sensing Version)
# 对应上方表格
elic_bpp = [
    0.114009, 0.334127, 0.43969, 0.4874, 
    0.894774, 1.126258, 1.615965, 2.525965, 5.493228
]
elic_psnr = [
    35.8327, 39.9366, 40.264, 41.256, 
    42.907, 44.064, 44.6495, 46.8189, 47.472
]

# Dataset B: JPEG2000 (JP2)
# 对应下方表格 (x轴取 actual_bpp, y轴取 psnr_10k_all_bands)
jp2_bpp = [
    0.103801, 0.203441, 0.330141, 
    0.496009, 0.691751, 0.987141
]
jp2_psnr = [
    32.5650, 36.4471, 37.7232, 
    38.7636, 39.6409, 40.6388
]

# ==========================================
# 2. 绘图设置
# ==========================================
plt.figure(figsize=(10, 7))

# 绘制 ELIC 曲线 (红色实线)
plt.plot(elic_bpp, elic_psnr, color='red', marker='o', 
         linestyle='-', linewidth=2, markersize=6, label='ELIC (Ours/Remote Sensing)')

# 绘制 JP2000 曲线 (蓝色虚线)
plt.plot(jp2_bpp, jp2_psnr, color='blue', marker='s', 
         linestyle='--', linewidth=2, markersize=6, label='JPEG2000')

# ==========================================
# 3. 美化图表 (论文风格)
# ==========================================

# 设置坐标轴标签
plt.xlabel('Bit Per Pixel (bpp)', fontsize=14)
plt.ylabel('PSNR (dB)', fontsize=14)
plt.title('Rate-Distortion Comparison: ELIC vs JPEG2000', fontsize=16)

# 开启网格，方便读数
plt.grid(True, which='both', linestyle='--', alpha=0.6)

# 添加图例
plt.legend(fontsize=12, loc='lower right')

# 限制一下 X 轴范围，让对比更明显
# 因为 ELIC 有一个很大的点 (5.49 bpp)，如果全部显示，左边的低码率区域会挤在一起。
# 建议：如果想看低码率对比细节，可以取消下面这行的注释
# plt.xlim(0, 1.5) 

# 保存与显示
plt.tight_layout()
plt.savefig('rd_curve_comparison.png', dpi=300) # 保存高清图
plt.show()

print("RD 曲线对比图已生成！")