import os
import cv2
import torch
import lpips
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from argparse import ArgumentParser

def main(args):
    # -------- 参数设置 --------
    raw_dir   = args.raw_dir
    mask_dir  = args.mask_dir
    rend_dir  = args.rend_dir
    out_txt   = args.out_txt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- 初始化 LPIPS --------
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    # -------- 结果存储 --------
    results = []

    # 遍历 renders 目录
    for fname in sorted(os.listdir(rend_dir)):
        # 忽略非图像文件
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        name_base = os.path.splitext(fname)[0]

        # 构造路径
        #raw_path  = os.path.join(raw_dir, fname)
        raw_path  = os.path.join(raw_dir, name_base + ".jpg")  # raw_frames 固定是 jpg
        #mask_path = os.path.join(mask_dir, fname)
        mask_path = os.path.join(mask_dir, name_base + ".png")  # mask 固定是 png
        rend_path = os.path.join(rend_dir, fname)

        # 输入文件检查
        if not (os.path.exists(raw_path) and os.path.exists(mask_path)):
            print(f"缺少对应 raw 或 mask for {name_base}, skip")
            continue

        # 读取图像
        raw  = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        rend = cv2.imread(rend_path, cv2.IMREAD_UNCHANGED)

        # raw * mask
        mask_norm = (mask.astype(np.float32) / 255.0)[:,:,None]  # 归一化
        raw_masked = (raw.astype(np.float32) * mask_norm).astype(np.uint8)
        rend_masked = (rend.astype(np.float32) * mask_norm).astype(np.uint8)

        # 确保尺寸一致
        # 如果 raw_masked 与 rend 尺寸不一致，则 resize raw_masked 到 rend 大小
        #h_r, w_r = rend.shape[:2]
        #raw_masked = cv2.resize(raw_masked, (w_r, h_r), interpolation=cv2.INTER_LINEAR)

        # ----- PSNR / SSIM -----
        psnr_val = peak_signal_noise_ratio(rend_masked, raw_masked, data_range=255)
        # SSIM 需灰度或多通道分别计算，目前 skimage 支持彩色
        #ssim_val = structural_similarity(rend, raw_masked, multichannel=True)
        ssim_val = structural_similarity(rend_masked, raw_masked, channel_axis=-1, data_range=255)

        # ----- LPIPS -----
        # 转换为 tensor [1,3,H,W] 归一化到 [-1,1]
        def to_tensor(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / 127.5 - 1.0
            img = img.permute(2,0,1).unsqueeze(0).to(device)
            return img

        t1 = to_tensor(rend_masked)
        t2 = to_tensor(raw_masked)
        with torch.no_grad():
            lpips_val = lpips_model(t1, t2).item()

        results.append((fname, psnr_val, ssim_val, lpips_val))

    # -------- 统计平均值 --------
    avg_psnr  = np.mean([r[1] for r in results])
    avg_ssim  = np.mean([r[2] for r in results])
    avg_lpips = np.mean([r[3] for r in results])

    # -------- 写入输出 --------
    with open(out_txt, "w") as f:
        f.write("img_name, PSNR, SSIM, LPIPS\n")
        for name, p, s, l in results:
            f.write(f"{name}, {p:.4f}, {s:.4f}, {l:.4f}\n")
        f.write("\nAVERAGE\n")
        f.write(f"PSNR: {avg_psnr:.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"LPIPS: {avg_lpips:.4f}\n")

    print(f"计算完毕，结果写入 {out_txt}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--raw_dir", type=str, default = None)
    parser.add_argument("--mask_dir", type=str, default = None)
    parser.add_argument("--rend_dir", type=str, default = None)
    parser.add_argument("--out_txt", type=str, default = None)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    main(args)