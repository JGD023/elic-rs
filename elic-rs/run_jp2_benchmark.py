import os
import glob
import math
import warnings
import sys
import numpy as np
from osgeo import gdal
from tqdm import tqdm

# --- 1. 配置 ---

# 【重要】设置目标 BPP (所有通道的总和)
# 我们将精确匹配您的 ELIC 范围
BPP_TARGETS = [0.1, 0.2, 0.33, 0.5, 0.7, 1.0]

# 【重要】匹配 ELIC 的归一化
PSNR_DATA_RANGE = 10000.0 
NUM_BANDS = 13
DATASET_ROOT = "./data/fmow-sentinel/val"

# (可选) 设置要测试的子目录名称
SUBDIRS = ["port", "shipyard", "airport", "nuclear_powerplant"]

# (可选) 限制每个子目录测试的图像数量 (用于快速测试)
IMAGE_LIMIT = 0 # 0 = 无限制

RESULTS_FILE = "jp2_per_channel_results_FINAL.csv"

# --- 2. 辅助函数 (PSNR 计算) ---

def calculate_psnr(original_band, compressed_band):
    """
    计算单个波段的 PSNR。
    硬编码 max_pixel_value = 10000.0 以匹配 ELIC
    """
    original = original_band.astype(np.float64)
    compressed = compressed_band.astype(np.float64)
    
    mse = np.mean((original - compressed) ** 2)
    
    if mse == 0:
        return float('inf') # 图像完全相同
    
    max_pixel_value = PSNR_DATA_RANGE
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

# --- 3. 主函数 ---

def main():
    # 抑制 GDAL 警告 (例如 TIFFReadDirectory...)
    warnings.filterwarnings('ignore')
    gdal.UseExceptions() # 让 GDAL 在失败时抛出 Python 异常

    # 准备 JP2OpenJPEG 驱动程序
    jp2_driver = gdal.GetDriverByName("JP2OpenJPEG")
    if jp2_driver is None:
        print("[致命错误] 找不到 JP2OpenJPEG 驱动程序。请确保您的 GDAL 已正确安装。")
        sys.exit(1)

    # 准备 GTiff 驱动程序 (用于读取和临时文件)
    gtiff_driver = gdal.GetDriverByName("GTiff")

    # 查找所有图像文件
    all_files = []
    print(f"--- 正在 {DATASET_ROOT} 中搜索 .tif 文件... ---")
    for subdir in SUBDIRS:
        search_path = os.path.join(DATASET_ROOT, subdir, "**", "*.tif")
        all_files.extend(glob.glob(search_path, recursive=True))
    print(f"找到 {len(all_files)} 个图像文件。")

    # 写入 CSV 标头
    with open(RESULTS_FILE, 'w') as f:
        f.write("target_bpp,total_images,actual_bpp,psnr_10k_all_bands\n")

    # --- 4. 主循环：遍历 BPP ---
    for bpp_target in BPP_TARGETS:
        print(f"\n=================================================")
        print(f"测试目标 BPP (所有通道): {bpp_target}")
        
        # 计算每个通道的目标 BPP
        per_channel_rate = bpp_target / NUM_BANDS
        print(f" (-> 目标 BPP (每通道): {per_channel_rate:.6f})")
        print(f"=================================================")

        total_psnr_all_images = 0.0
        total_bpp_all_images = 0.0
        total_images_processed = 0

        # --- 5. 循环：遍历所有文件 ---
        pbar = tqdm(all_files, desc=f"BPP {bpp_target}")
        for tif_file in pbar:
            
            try:
                # 检查文件是否可读（处理损坏的链接）
                if not os.path.exists(tif_file) or not os.path.getsize(tif_file) > 0:
                    tqdm.write(f"  [警告] 文件 {os.path.basename(tif_file)} 不可读 (0字节或链接损坏)，跳过。")
                    continue
                
                ds_orig = gdal.Open(tif_file)
                if ds_orig is None:
                    tqdm.write(f"  [警告] GDAL 无法打开 {os.path.basename(tif_file)}，跳过。")
                    continue
                
                # 检查波段数
                if ds_orig.RasterCount != NUM_BANDS:
                    tqdm.write(f"  [警告] {os.path.basename(tif_file)} 波段数 ({ds_orig.RasterCount}) != 13，跳过。")
                    continue
                
                pixels = ds_orig.RasterXSize * ds_orig.RasterYSize
                
                total_bits_for_image = 0
                total_psnr_for_image = 0
                bands_processed_successfully = 0

                # --- 6. 核心：逐通道循环 ---
                for b in range(1, NUM_BANDS + 1):
                    # 1. 提取原始波段数据
                    band_orig_ds = gdal.Translate("", ds_orig, bandList=[b], format="VRT")
                    if band_orig_ds is None: continue
                    band_orig_data = band_orig_ds.ReadAsArray()

                    # 2. 压缩 (使用 Python 驱动程序)
                    temp_jp2_file = f"/tmp/band_{b}.jp2"
                    options = [
                        f"RATES={per_channel_rate}",
                        "REVERSIBLE=NO",
                        "GeoJP2=NO" # (不需要地理信息)
                    ]
                    # 使用 CreateCopy 将波段数据集压缩到临时文件
                    jp2_ds = jp2_driver.CreateCopy(temp_jp2_file, band_orig_ds, options=options)
                    jp2_ds = None # 关闭文件并写入磁盘
                    
                    # 3. 累加码率
                    total_bits_for_image += os.path.getsize(temp_jp2_file) * 8
                    
                    # 4. 解码
                    ds_recon = gdal.Open(temp_jp2_file)
                    if ds_recon is None: continue
                    band_recon_data = ds_recon.ReadAsArray()
                    
                    # 5. 评估 PSNR
                    psnr = calculate_psnr(band_orig_data, band_recon_data)
                    
                    if not math.isinf(psnr):
                        total_psnr_for_image += psnr
                        bands_processed_successfully += 1
                    
                    ds_recon = None
                    os.remove(temp_jp2_file) # 清理临时 .jp2 文件

                # --- 7. 计算此图像的平均值 ---
                if bands_processed_successfully == NUM_BANDS:
                    actual_bpp = total_bits_for_image / pixels
                    avg_psnr = total_psnr_for_image / NUM_BANDS
                    
                    total_bpp_all_images += actual_bpp
                    total_psnr_all_images += avg_psnr
                    total_images_processed += 1
                else:
                    tqdm.write(f"  [警告] 文件 {os.path.basename(tif_file)} 未能成功处理所有 13 个波段，已剔除。")
                
                ds_orig = None # 关闭 TIF 文件

            except Exception as e:
                tqdm.write(f"  [致命错误] 处理 {os.path.basename(tif_file)} 时出错: {e}。跳过。")
        
        # --- 8. 计算该 BPP 级别的总平均值 ---
        if total_images_processed > 0:
            avg_bpp = total_bpp_all_images / total_images_processed
            avg_psnr = total_psnr_all_images / total_images_processed
            
            print("-------------------------------------------------")
            print(f"目标 BPP {bpp_target} 结果 (共 {total_images_processed} 张图像):")
            print(f"  Avg Actual bpp:    {avg_bpp:.6f}")
            print(f"  Avg PSNR (10k):    {avg_psnr:.4f} dB")
            print("-------------------------------------------------")
            
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{bpp_target},{total_images_processed},{avg_bpp:.10f},{avg_psnr:.10f}\n")
        else:
            print(f"[警告] 目标 BPP {bpp_target} 未能处理任何图像。")
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{bpp_target},0,0,0\n")

    print(f"\n测试完成！结果已保存到 {RESULTS_FILE}")

if __name__ == "__main__":
    main()