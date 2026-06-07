#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, log, pi
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import cv2


def l1_loss(network_output, gt, weight = None, mask = None):
    loss = torch.abs(network_output - gt)
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss.mean()

def ce_loss(network_output, gt):
    return F.binary_cross_entropy(network_output.clamp(1e-3, 1.0 - 1e-3), gt)

def or_loss(network_output, gt, confs = None, weight = None, mask = None):
    weight = torch.ones_like(gt[:1]) if weight is None else weight
    loss = torch.minimum(
        (network_output - gt).abs(),
        torch.minimum(
            (network_output - gt - 1).abs(), 
            (network_output - gt + 1).abs()
        ))
    loss = loss * pi
    if confs is not None:
        loss = loss * confs - (confs + 1e-7).log()    
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss * weight

def dp_loss(pred, gt, pred_mask, gt_mask, eps=0.1):
    filter_fg = torch.logical_and(gt_mask >= 1 - eps, pred_mask >= 1 - eps).detach()
    
    if (filter_fg.sum() == 0).all():
        return None, pred, gt
    
    pred_fg = pred[filter_fg]
    gt_fg = gt[filter_fg]

    with torch.no_grad():    
        # # Subsample points
        # idx_1 = torch.argsort(gt_fg).detach()
        # idx_2 = torch.randperm(gt_fg.shape[0], device='cuda').detach()
        # to_penalize = torch.logical_or(
        #     torch.logical_and(idx_1 < idx_2, pred_fg[idx_1] > pred_fg[idx_2]),
        #     torch.logical_and(idx_1 > idx_2, pred_fg[idx_1] < pred_fg[idx_2])
        # ).detach()

        pred_q2, pred_q98 = torch.quantile(pred_fg, torch.tensor([0.02, 0.98]).cuda())
        gt_q2, gt_q98 = torch.quantile(gt_fg, torch.tensor([0.02, 0.98]).cuda())

    pred_aligned = ((pred - pred_q2.detach()) / (pred_q98.detach() - gt_q2.detach())).clamp(0, 1)
    gt_aligned = ((gt - gt_q2) / (gt_q98 - gt_q2)).clamp(0, 1)

    mask = gt_mask * pred_mask.detach()
    pred_masked = pred_aligned * mask + (1 - mask)
    gt_masked = gt_aligned * mask + (1 - mask)

    loss = (pred_masked - gt_masked).abs().mean()

    return loss, pred_masked, gt_masked

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average=True)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


class PerceptualLoss(nn.Module):
    def __init__(self, layer='relu3_3', device='cuda'):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        assert layer in self.layer_map, "Unsupported layer name."

        self.vgg = nn.Sequential(*list(vgg.children())[:self.layer_map[layer] + 1])
        self.vgg.to(device)
        #self.criterion = nn.L1Loss().to(device)
        self.criterion = nn.MSELoss().to(device)
        self.device = device

    def extract_features(self, image_tensor):
        with torch.no_grad():
            return self.vgg(image_tensor.to(self.device))

    def forward(self, input, target, mask=None):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #size=(224, 224)
        #input = self.load_image(input, self.device)
        #target = self.load_image(target, self.device)
        #import pdb;pdb.set_trace()
        #=== Test ===
        input = input.to(self.device)
        target = target.to(self.device)

        #if input.dim() == 3:  # [C, H, W]
            #input = input.unsqueeze(0)  # -> [1, C, H, W]

        #if target.dim() == 3:
            #target = target.unsqueeze(0)  # -> [1, C, H, W]
        #input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
        #target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        # === Test ===
        with torch.no_grad():
            input_feat = self.vgg(input)
            #target_feat = self.vgg(target)
        return self.criterion(input_feat, target)
        #if mask is not None:
            # 假设 mask 是 [1, 1, H, W] 或 [B, 1, H, W]，需要 broadcast 到 feature 尺寸
            # 比如 relu3_3 输出是 [B, C, H/8, W/8]
            #if mask.ndim == 3:
               # mask = mask.unsqueeze(1)
            #mask = F.interpolate(mask, size=input_feat.shape[2:], mode='bilinear', align_corners=False)
            #return self.criterion(input_feat * mask, target_feat * mask)
        #else:
            #return self.criterion(input_feat, target_feat)


def load_image(image, size=(224, 224)):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),  # [0, 255] → [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                std=[0.229, 0.224, 0.225])
        ])
        #image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).cuda()   # shape: [1, 3, H, W]
        return image

class EdgeLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(EdgeLoss, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.l1 = nn.L1Loss().to(device)

        # Sobel kernels [1, 1, 3, 3]
        sobel_kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        sobel_kernel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        self.register_buffer('sobel_x', sobel_kernel_x)
        self.register_buffer('sobel_y', sobel_kernel_y)

    def get_edge(self, x):  # x: [3, H, W]
        if x.dim() != 3:
            raise ValueError("Input must be [C, H, W]")

        x = x.unsqueeze(0).to(self.device)  # -> [1, 3, H, W]
        B, C, H, W = x.shape

        sobel_x = self.sobel_x.expand(C, 1, 3, 3)
        sobel_y = self.sobel_y.expand(C, 1, 3, 3)

        edge_x = F.conv2d(x, sobel_x, groups=C, padding=1)
        edge_y = F.conv2d(x, sobel_y, groups=C, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        return edge.squeeze(0)  # -> [C, H, W]

    def forward(self, pred_img, gt_img):  # both: [3, H, W]
        pred_img = pred_img.to(self.device)
        gt_img = gt_img.to(self.device)

        pred_edge = self.get_edge(pred_img)
        gt_edge = self.get_edge(gt_img)
        return self.l1(pred_edge, gt_edge)

class CannyEdgeLoss(nn.Module):
    """
    OpenCV Canny 封装成 PyTorch 风格，支持 pred/gt 边缘计算及 L1 Loss
    """
    def __init__(self, low_threshold=50, high_threshold=250, device='cuda'):
        super(CannyEdgeLoss, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.l1 = nn.L1Loss()

    def _to_tensor(self, img):
        """
        将 PIL.Image 或 [C,H,W] tensor 转成 [C,H,W] tensor 范围 [0,1]
        """
        if torch.is_tensor(img):
            img_tensor = img.float()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
        elif isinstance(img, Image.Image):
            img_tensor = ToTensor()(img)
        else:
            raise TypeError("img must be torch.Tensor or PIL.Image")
        return img_tensor.to(self.device)

    def _canny_tensor(self, img_tensor):
        """
        输入 [C,H,W] tensor，返回边缘图 [C,H,W] tensor
        """
        img_np = (img_tensor.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)  # H,W,C
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        edges_3ch = np.stack([edges]*3, axis=-1)
        edges_tensor = torch.from_numpy(edges_3ch).permute(2,0,1).float() / 255.0
        return edges_tensor.to(self.device)

    def forward(self, pred_img, gt_img):
        """
        pred_img, gt_img: [C,H,W] tensor 或 PIL.Image
        save_prefix: str，如果不为 None 则保存边缘图
        返回 L1 边缘损失
        """
        pred_tensor = self._to_tensor(pred_img)
        #gt_tensor = self._to_tensor(gt_img)

        pred_edge = self._canny_tensor(pred_tensor)
        #gt_edge = self._canny_tensor(gt_tensor)

        loss = self.l1(pred_edge, gt_img)
        return loss