import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
import sys
from argparse import ArgumentParser
sys.path.append('../')
from utils.loss_utils import PerceptualLoss
from matplotlib import pyplot as plt


def main(args):
    # === 图像预处理 ===
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),  # VGG输入需要统一尺寸
        transforms.ToTensor()
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #std=[0.229, 0.224, 0.225])
    ])

    # === 设置路径 ===
    #image_folder = "./render_images"
    #save_folder = "./vgg_features"
    #image_folder = "/mnt/data/ljy/Li/GaussianHaircut/20250730_bomb_sideways_sun_CLIP_LIT_Ploss/images_2"
    #save_folder1 = "/mnt/data/ljy/Li/GaussianHaircut/20250730_bomb_sideways_sun_CLIP_LIT_Ploss/vgg_features1_pth"
    #save_folder2 = "/mnt/data/ljy/Li/GaussianHaircut/20250730_bomb_sideways_sun_CLIP_LIT_Ploss/vgg_features1_png"
    #mask_folder = "/mnt/data/ljy/Li/GaussianHaircut/20250730_bomb_sideways_sun_CLIP_LIT_Ploss/masks_2/body"
    image_folder = args.image_folder
    save_folder1 = args.save_folder1
    save_folder2 = args.save_folder2
    mask_folder = args.mask_folder
    os.makedirs(save_folder1, exist_ok=True)
    os.makedirs(save_folder2, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PerceptualLoss(layer='relu3_3', device=device)

    # === 遍历图像并提取 VGG 特征 ===
    for image_name in tqdm(os.listdir(image_folder)):
        if not image_name.endswith(".png"):
            continue

        basename = os.path.splitext(image_name)[0]

        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name)
        if not os.path.exists(mask_path):
            continue
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        #image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
        #import pdb;pdb.set_trace()
        image_tensor = transform(image)
        mask_tensor = transform(mask)
        image_tensor = image_tensor * mask_tensor

        #import pdb;pdb.set_trace()
        features = model.extract_features(image_tensor)  # [1, C, H', W']

        # 保存为 .pth
        torch.save(features.cpu(), os.path.join(save_folder1, f"{basename}.pth"))

        #import pdb;pdb.set_trace()
        # 如果你希望保存为可视化图像（仅可视化第一通道）
        #feat_map = features[0, 0].detach().cpu().numpy()
        feat_map = features[0].detach().cpu().numpy()
        
        plt.imsave(os.path.join(save_folder2, f"{basename}.png"), feat_map, cmap='viridis')

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--image_folder', default='', type=str)
    parser.add_argument('--save_folder1', default='', type=str)
    parser.add_argument('--save_folder2', default='', type=str)
    parser.add_argument('--mask_folder', default='', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)
