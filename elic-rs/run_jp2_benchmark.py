import os
import glob
import math
import warnings
import sys
import numpy as np
from osgeo import gdal
from tqdm import tqdm

# --- 1. 配置 ---

# 【重要】设置目标 BPPBF (bits per pixel per band)
# 我们将精确匹配您的 ELIC 范围
BPPBF_TARGETS = [0.008, 0.025, 0.033, 0.037, 0.068, 0.086] # <-- (修改) 使用 bppbf 目标

# 【重要】匹配 ELIC 的归一化
PSNR_DATA_RANGE = 10000.0 
NUM_BANDS = 13
DATASET_ROOT = "./data/fmow-sentinel/val"

SUBDIRS = ["port", "shipyard", "airport", "nuclear_powerplant"]
IMAGE_LIMIT = 0 
RESULTS_FILE = "jp2_per_channel_results_BPPBF.csv" # <-- (修改) 新文件名

# --- 2. 辅助函数 (PSNR 计算) ---
def calculate_psnr(original_band, compressed_band):
    original = original_band.astype(np.float64)
    compressed = compressed_band.astype(np.float64)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = PSNR_DATA_RANGE
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

# --- 3. 主函数 ---
def main():
    warnings.filterwarnings('ignore')
    gdal.UseExceptions()

    jp2_driver = gdal.GetDriverByName("JP2OpenJPEG")
    if jp2_driver is None:
        print("[致命错误] 找不到 JP2OpenJPEG 驱动程序。")
        sys.exit(1)
    gtiff_driver = gdal.GetDriverByName("GTiff")

    all_files = []
    print(f"--- 正在 {DATASET_ROOT} 中搜索 .tif 文件... ---")
    for subdir in SUBDIRS:
        search_path = os.path.join(DATASET_ROOT, subdir, "**", "*.tif")
        all_files.extend(glob.glob(search_path, recursive=True))
    print(f"找到 {len(all_files)} 个图像文件。")

    # --- (修改) 写入 CSV 标头 ---
    with open(RESULTS_FILE, 'w') as f:
        f.write("target_bppbf,total_images,actual_bppbf,psnr_10k_all_bands\n")

    # --- 4. 主循环：遍历 BPPBF ---
    for bppbf_target in BPPBF_TARGETS: # <-- (修改)
        print(f"\n=================================================")
        print(f"测试目标 BPPBF (每通道): {bppbf_target:.6f}")
        print(f"=================================================")

        # (修改) per_channel_rate 现在就是 bppbf_target
        per_channel_rate = bppbf_target 

        total_psnr_all_images = 0.0
        total_bppbf_all_images = 0.0 # <-- (修改)
        total_images_processed = 0

        # --- 5. 循环：遍历所有文件 ---
        pbar = tqdm(all_files, desc=f"BPPBF {bppbf_target}") # <-- (修改)
        for tif_file in pbar:
            try:
                if not os.path.exists(tif_file) or not os.path.getsize(tif_file) > 0:
                    tqdm.write(f"  [警告] 文件 {os.path.basename(tif_file)} 不可读，跳过。")
                    continue
                
                ds_orig = gdal.Open(tif_file)
                if ds_orig is None:
                    tqdm.write(f"  [警告] GDAL 无法打开 {os.path.basename(tif_file)}，跳过。")
                    continue
                
                if ds_orig.RasterCount != NUM_BANDS:
                    tqdm.write(f"  [警告] {os.path.basename(tif_file)} 波段数 ({ds_orig.RasterCount}) != 13，跳过。")
                    continue
                
                pixels = ds_orig.RasterXSize * ds_orig.RasterYSize
                
                total_bits_for_image = 0
                total_psnr_for_image = 0
                bands_processed_successfully = 0

                # --- 6. 核心：逐通道循环 ---
                for b in range(1, NUM_BANDS + 1):
                    band_orig_ds = gdal.Translate("", ds_orig, bandList=[b], format="VRT")
                    if band_orig_ds is None: continue
                    band_orig_data = band_orig_ds.ReadAsArray()

                    temp_jp2_file = f"/tmp/band_{b}.jp2"
                    options = [
                        f"RATES={per_channel_rate}", # <-- (修改) 使用每通道的 bppbf
                        "REVERSIBLE=NO",
                        "GeoJP2=NO"
                    ]
                    jp2_ds = jp2_driver.CreateCopy(temp_jp2_file, band_orig_ds, options=options)
                    jp2_ds = None 
                    
                    total_bits_for_image += os.path.getsize(temp_jp2_file) * 8
                    
                    ds_recon = gdal.Open(temp_jp2_file)
                    if ds_recon is None: continue
                    band_recon_data = ds_recon.ReadAsArray()
                    
                    psnr = calculate_psnr(band_orig_data, band_recon_data)
                    
                    if not math.isinf(psnr):
                        total_psnr_for_image += psnr
                        bands_processed_successfully += 1
                    
                    ds_recon = None
                    os.remove(temp_jp2_file)

                # --- 7. 计算此图像的平均值 ---
                if bands_processed_successfully == NUM_BANDS:
                    # --- (关键修改) ---
                    actual_bpp = total_bits_for_image / pixels # 这是总 bpp
                    actual_bppbf = actual_bpp / NUM_BANDS # 这是 bppbf
                    avg_psnr = total_psnr_for_image / NUM_BANDS
                    
                    total_bppbf_all_images += actual_bppbf # <-- (修改)
                    total_psnr_all_images += avg_psnr
                    total_images_processed += 1
                    # --- (修改结束) ---
                else:
                    tqdm.write(f"  [警告] 文件 {os.path.basename(tif_file)} 未能成功处理所有 13 个波段，已剔除。")
                
                ds_orig = None

            except Exception as e:
                tqdm.write(f"  [致命错误] 处理 {os.path.basename(tif_file)} 时出错: {e}。跳过。")
        
        # --- 8. 计算该 BPPBF 级别的总平均值 ---
        if total_images_processed > 0:
            avg_bppbf = total_bppbf_all_images / total_images_processed # <-- (修改)
            avg_psnr = total_psnr_all_images / total_images_processed
            
            print("-------------------------------------------------")
            print(f"目标 BPPBF {bppbf_target} 结果 (共 {total_images_processed} 张图像):") # <-- (修改)
            print(f"  Avg Actual bppbf:  {avg_bppbf:.6f}") # <-- (修改)
            print(f"  Avg PSNR (10k):    {avg_psnr:.4f} dB")
            print("-------------------------------------------------")
            
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{bppbf_target},{total_images_processed},{avg_bppbf:.10f},{avg_psnr:.10f}\n") # <-- (修改)
        else:
            print(f"[警告] 目标 BPPBF {bppbf_target} 未能处理任何图像。") # <-- (修改)
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{bppbf_target},0,0,0\n") # <-- (修改)

    print(f"\n测试完成！结果已保存到 {RESULTS_FILE}")

if __name__ == "__main__":
    main()