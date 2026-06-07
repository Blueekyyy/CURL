import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
import sys
from argparse import ArgumentParser
from torchvision.transforms import ToTensor
sys.path.append('../')
from utils.loss_utils import EdgeLoss


def main(args):
    """
    #=== Test ===
    image_folder = args.image_folder
    save_folder2 = args.save_folder2
    os.makedirs(save_folder2, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_loss_fn = EdgeLoss()

    for image_name in tqdm(os.listdir(image_folder)):
        if not image_name.endswith(".png"):
            continue

        basename = os.path.splitext(image_name)[0]

        image_path = os.path.join(image_folder, image_name)
        image = ToTensor()(Image.open(image_path).convert('RGB'))
        
        gt_edge = edge_loss_fn.get_edge(image)

        feat_map = gt_edge[0].detach().cpu().numpy()
        from matplotlib import pyplot as plt
        plt.imsave(os.path.join(save_folder2, f"{basename}.png"), feat_map, cmap='viridis')
    """

    
    image_folder = args.image_folder
    save_folder1 = args.save_folder1
    save_folder2 = args.save_folder2
    mask_folder = args.mask_folder
    os.makedirs(save_folder1, exist_ok=True)
    os.makedirs(save_folder2, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_loss_fn = EdgeLoss()

    for image_name in tqdm(os.listdir(image_folder)):
        if not image_name.endswith(".png"):
            continue

        basename = os.path.splitext(image_name)[0]

        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name)
        if not os.path.exists(mask_path):
            continue
        image = ToTensor()(Image.open(image_path).convert('RGB'))
        mask = ToTensor()(Image.open(mask_path).convert('RGB'))
        
        image = image * mask
        gt_edge = edge_loss_fn.get_edge(image)
        # 保存为 .pth
        torch.save(gt_edge.cpu(), os.path.join(save_folder1, f"{basename}.pth"))

        feat_map = gt_edge[0].detach().cpu().numpy()
        from matplotlib import pyplot as plt
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
