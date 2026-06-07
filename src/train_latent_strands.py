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

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_chamfer_utils import chamfer_distance
from utils.loss_utils import l1_loss, ce_loss, or_loss, ssim, PerceptualLoss, EdgeLoss, CannyEdgeLoss
from gaussian_renderer import render_hair, network_gui
import sys
import yaml
from scene import Scene, GaussianModel, GaussianModelHair
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, vis_orient
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import pickle as pkl
from utils.general_utils import build_rotation
import time
from kaolin.metrics.pointcloud import sided_distance

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import imageio
import numpy as np


#import pdb; pdb.set_trace()
def training(dataset, opt, opt_hair, pipe, testing_iterations, saving_iterations, checkpoint_iterations, model_path_hair, pointcloud_path_head, checkpoint, checkpoint_hair, debug_from, is_step2_2, checkpoint_hair_ljy):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, model_path_hair)
    #import pdb;pdb.set_trace()
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians_hair = GaussianModelHair(dataset.source_path, dataset.flame_mesh_dir, opt_hair, dataset.sh_degree)
    scene = Scene(dataset, gaussians, pointcloud_path=pointcloud_path_head, load_iteration=-1)
    gaussians.training_setup(opt)
    gaussians_hair.create_from_pcd(dataset.source_path, dataset.strand_scale)
    gaussians_hair.training_setup(opt, opt_hair)
    if checkpoint:
        model_params, _ = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if checkpoint_hair:
        model_params, first_iter = torch.load(checkpoint_hair)
        gaussians_hair.restore(model_params, opt_hair)
    if checkpoint_hair_ljy:
        model_params, first_iter = torch.load(checkpoint_hair_ljy)
        first_iter = 0
        gaussians_hair.restore_ljy(model_params, opt, opt_hair)
    if dataset.trainable_cameras:
        print(f'Loading optimized cameras from iter {scene.loaded_iter}')
        #第一步训练中用BARF训练出旋转、平移、视场角
        #BARF optimize the parameters of the training cameras and perform 3D lifting of the scene, including orientation maps
        #we employ a learnable 6-DoF camera parameterization from BARF [19] as a residual to the initial estimation produced by SfM and train it alongside the 3D Gaussians using gradient-based optimization.
        #sfm即为colmap稀疏重建结果
        params_cam_rotation, params_cam_translation, params_cam_fov = pkl.load(open(scene.model_path + "/cameras/" + str(scene.loaded_iter) + ".pkl", 'rb'))
        for k in scene.train_cameras.keys():
            for camera in scene.train_cameras[k]:
                if dataset.trainable_cameras:
                    camera._rotation_res.data = params_cam_rotation[camera.image_name]
                    camera._translation_res.data = params_cam_translation[camera.image_name]
                if dataset.trainable_intrinsics:
                    #fov有平移、垂直，视锥体（CG）水平、垂直度数拍摄
                    camera._fov_res.data = params_cam_fov[camera.image_name]

    with torch.no_grad():
        # Head gaussians data
        gaussians.mask_precomp = gaussians.get_label[..., 0] < 0.5
        gaussians.xyz_precomp = gaussians.get_xyz[gaussians.mask_precomp].detach()
        gaussians.opacity_precomp = gaussians.get_opacity[gaussians.mask_precomp].detach()
        gaussians.scaling_precomp = gaussians.get_scaling[gaussians.mask_precomp].detach()
        gaussians.rotation_precomp = gaussians.get_rotation[gaussians.mask_precomp].detach()
        gaussians.cov3D_precomp = gaussians.get_covariance(1.0)[gaussians.mask_precomp].detach()
        gaussians.shs_view = gaussians.get_features[gaussians.mask_precomp].detach().transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1)**2)

    bg_color = [1, 1, 1, 0, 0, 0, 0, 0, 0, 100] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    """
    # === Test ===
    #import pdb;pdb.set_trace()
    output_dir = f"{dataset.source_path}/train_latent_strands_imgs"  # 👈 替换成你要保存的目录
    os.makedirs(output_dir, exist_ok=True)
    """
    # === Test ===
    loss_fn = PerceptualLoss(layer='relu3_3').cuda()
    #edge_loss_fn = EdgeLoss()
    edge_loss_fn = CannyEdgeLoss(low_threshold=30, high_threshold=150)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_hair(custom_cam, gaussians, gaussians_hair, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        #import pdb; pdb.set_trace()
        gaussians_hair.initialize_gaussians_hair(iteration)  #设置颜色、生成坐标等等
        gaussians_hair.update_learning_rate(iteration)

        # Pick a random Camera
        #import pdb; pdb.set_trace()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_hair(viewpoint_cam, gaussians, gaussians_hair, pipe, background)
    
        image = render_pkg["render"]
        mask = render_pkg["mask"]
        orient_angle = render_pkg["orient_angle"]

        """
        # === Test ===
        #import pdb;pdb.set_trace()
        # Step 1: 转换为 NumPy，转为 HWC 格式
        image_np = image.detach().cpu().numpy()         # (3, H, W)
        image_np = np.transpose(image_np, (1, 2, 0))     # → (H, W, 3)
        # Step 2: 归一化 + 转 uint8
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0.0, 1.0)      # ensure within 0~1
            image_np = (image_np * 255).astype(np.uint8)
        # Step 3: 构建文件名并保存
        filename1 = f"image_{iteration:06d}.png"
        output_path = os.path.join(output_dir, filename1)
        imageio.imwrite(output_path, image_np)

        # Step 1: 转换为 NumPy，转为 HWC 格式
        mask_np = mask.detach().cpu().numpy()         # (3, H, W)
        mask_np = np.transpose(mask_np, (1, 2, 0))     # → (H, W, 3)
        # Step 2: 归一化 + 转 uint8
        if mask_np.dtype != np.uint8:
            mask_np = np.clip(mask_np, 0.0, 1.0)      # ensure within 0~1
            mask_np = (mask_np * 255).astype(np.uint8)
        # Step 3: 构建文件名并保存
        filename2 = f"mask_{iteration:06d}.png"
        output_path = os.path.join(output_dir, filename2)
        imageio.imwrite(output_path, mask_np)
        
        #import pdb;pdb.set_trace()
        # Step 1: 转换为 NumPy，转为 HWC 格式
        orient_angle_np = orient_angle.detach().cpu().numpy()         # (3, H, W)
        orient_angle_np = np.transpose(orient_angle_np, (1, 2, 0))     # → (H, W, 3)
        # Step 1: 如果是单通道图像 → 扩展为伪 RGB
        if orient_angle_np.shape[2] == 1:
            orient_angle_np = np.repeat(orient_angle_np, 3, axis=2)  # (H, W, 3)
        # Step 3: 归一化 + 转 uint8
        if orient_angle_np.dtype != np.uint8:
            orient_angle_np = np.clip(orient_angle_np, 0.0, 1.0)      # ensure within 0~1
            orient_angle_np = (orient_angle_np * 255).astype(np.uint8)
        # Step 4: 构建文件名并保存
        filename3 = f"orient_angle_{iteration:06d}.png"
        output_path = os.path.join(output_dir, filename3)
        imageio.imwrite(output_path, orient_angle_np)
        """

        orient_conf = render_pkg["orient_conf"]
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_mask = viewpoint_cam.original_mask.cuda()
        gt_orient_angle = viewpoint_cam.original_orient_angle.cuda()
        gt_orient_conf = viewpoint_cam.original_orient_conf.cuda()
        #=== Test ===
        gt_pct_loss = viewpoint_cam.original_pct_loss.cuda()
        gt_edge_loss = viewpoint_cam.original_edge_loss.cuda()

        LCE = l1_loss(mask[:1], gt_mask[:1]) #, mask=(gt_mask[:1] < 0.5).detach())
        Ll1 = l1_loss(image, gt_image)
    
        # These losses are only applied for hair
        orient_weight = torch.ones_like(gt_mask[:1])
        if opt.use_gt_orient_conf: orient_weight = orient_weight * gt_orient_conf
        if not opt.train_orient_conf: orient_conf = None
        LOR = or_loss(orient_angle, gt_orient_angle, orient_conf, weight=orient_weight, mask=gt_mask[:1])

        # Diffusion loss
        #LDF = gaussians_hair.LDiff if gaussians_hair.LDiff is not None else torch.zeros_like(LOR)
        #LDF = 0
        LDF = torch.tensor(0.0)

        #=== Perceptual loss ===
        #Lperceptual = loss_fn(image * gt_mask[1:], gt_image * gt_mask[1:], gt_mask[1:])
        Lperceptual = loss_fn(image * gt_mask[1:], gt_pct_loss, gt_mask[1:])   # input, target, mask
        #=== Edge loss===
        #Ledge = edge_loss_fn(image * gt_mask[1:], gt_image * gt_mask[1:])
        Ledge = edge_loss_fn(image * gt_mask[1:], gt_edge_loss) #pred_img, gt_img
        Lssim = (1.0 - ssim(image * gt_mask[1:], gt_image * gt_mask[1:]))

        if torch.isnan(Ll1).any(): Ll1 = torch.zeros_like(Ll1)
        if torch.isnan(LCE).any(): LCE = torch.zeros_like(Ll1)
        if torch.isnan(LOR).any(): LOR = torch.zeros_like(Ll1)
        #if torch.isnan(LDF).any(): LDF = torch.zeros_like(Ll1)
        if torch.isnan(Lperceptual).any(): Lperceptual = torch.zeros_like(Ll1)
        if torch.isnan(Ledge).any(): Ledge = torch.zeros_like(Ll1)
        if torch.isnan(Lssim).any(): Lssim = torch.zeros_like(Ll1)

        if is_step2_2:
            loss = (
                Ll1 * opt.lambda_dl1 + 
                LCE * opt.lambda_dmask +
                Ledge * opt.lambda_dedge_2 +
                Lssim * opt.lambda_dssim
            )
        else:
            loss = (
                Ll1 * opt.lambda_dl1 + 
                LCE * opt.lambda_dmask +
                LOR * opt.lambda_dorient +
                #LDF * opt.lambda_dsds +
                Lperceptual * opt.lambda_dperceptual +
                Ledge * opt.lambda_dedge +
                Lssim * opt.lambda_dssim
            )
        loss.backward()  # 组合多个损失项进行梯度计算。

        iter_end.record()

        # Optimizer step
        if iteration < opt.iterations:
            for param in gaussians_hair.optimizer.param_groups[0]['params']:
                if param.grad is not None and param.grad.isnan().any():
                    #gaussians_hair.optimizer.zero_grad()
                    #print('NaN during backprop was found, skipping iteration...')
                    #=== Test ===
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)
                    print('NaN during backprop was found, set to 0.0...')
            gaussians_hair.optimizer.step()
            gaussians_hair.optimizer.zero_grad(set_to_none = True)
            # 避免 NaN 问题，只有梯度正常时才进行优化器步进。

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, LCE, LOR, LDF, loss, l1_loss, ce_loss, or_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians_hair, render_hair, (pipe, background))

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(model_path_hair + "/checkpoints", exist_ok=True)
                #保存整个训练的模型
                torch.save((gaussians_hair.capture(), iteration), model_path_hair + "/checkpoints/" + str(iteration) + ".pth")
                #没保存颜色，监控diffusion的纹理，看函数结构
                # torch.save(gaussians_hair.strands_generator.texture_decoder.state_dict(), f'{model_path_hair}/checkpoints/texture_decoder.pth')

def prepare_output_and_logger(args, model_path_hair):  
    """
    确定输出目录（用户指定或自动生成）。
    创建输出文件夹，防止训练结果丢失。
    保存训练参数，确保实验可复现。
    初始化 TensorBoard，用于可视化训练日志。
    """  
    # 生成或使用指定的输出目录
    if not model_path_hair: # 如果 model_path_hair 为空
        if os.getenv('OAR_JOB_ID'): 
            unique_str=os.getenv('OAR_JOB_ID') # 从环境变量获取作业 ID ，尝试从 OAR_JOB_ID（任务 ID）获取唯一标识（可能是在高性能计算集群上运行）。
        else:
            unique_str = str(uuid.uuid4()) # 生成一个唯一的 UUID
        model_path_hair = os.path.join("./output/", unique_str[0:10]) # 取 UUID 的前 10 个字符作为路径，将其拼接成 ./output/xxxxxxx 形式
        
    # Set up output folder
    print("Output folder: {}".format(model_path_hair)) # 打印输出路径，方便用户查看日志存储位置。
    os.makedirs(model_path_hair, exist_ok = True)  # 创建目录，如果目录已存在，不会报错。
    # 保存训练参数到文件
    with open(os.path.join(model_path_hair, "cfg_args"), 'w') as cfg_log_f:  # 把 args（训练参数）保存到 cfg_args 文件，用于后续检查或复现实验。
        cfg_log_f.write(str(Namespace(**vars(args))))  # Namespace(**vars(args))：转换 args 为 Namespace，便于序列化。

    # Create Tensorboard writer
    # 初始化 TensorBoard 记录
    tb_writer = None
    if TENSORBOARD_FOUND:  # 判断是否安装了 TensorBoard。
        tb_writer = SummaryWriter(model_path_hair)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, LCE, LOR, LDF, loss, l1_loss, ce_loss, or_loss, elapsed, testing_iterations, scene : Scene, gaussians_hair, renderFunc, renderArgs):
    # 记录训练损失 记录不同损失值，并存入 TensorBoard，便于后续可视化分析。
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ce_loss', LCE.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/or_loss', LOR.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/df_loss', LDF.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        # 记录 gaussians_hair 的缩放参数
        tb_writer.add_scalar("scaling/0", gaussians_hair.get_scaling[:, 0].mean().item(), iteration)
        tb_writer.add_scalar("scaling/1", gaussians_hair.get_scaling[:, 1].mean(), iteration)
        tb_writer.add_scalar("scaling/2", gaussians_hair.get_scaling[:, 2].mean(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        gaussians_hair.initialize_gaussians_hair(iteration)
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                ce_test = 0.0
                or_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, gaussians_hair, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    mask = torch.clamp(render_pkg["mask"], 0.0, 1.0)
                    orient_angle = torch.clamp(render_pkg["orient_angle"], 0.0, 1.0)
                    orient_conf = render_pkg["orient_conf"]
                    orient_conf_vis = (1 - 1 / (orient_conf + 1)) * mask[:1]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_mask = torch.clamp(viewpoint.original_mask.to("cuda"), 0.0, 1.0)
                    gt_orient_angle = torch.clamp(viewpoint.original_orient_angle.to("cuda"), 0.0, 1.0)
                    gt_orient_conf = viewpoint.original_orient_conf.to("cuda")
                    gt_orient_conf_vis = (1 - 1 / (gt_orient_conf + 1)) * gt_mask[:1]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_mask".format(viewpoint.image_name), F.pad(mask, (0, 0, 0, 0, 0, 3-mask.shape[0]), 'constant', 0)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_orient".format(viewpoint.image_name), vis_orient(orient_angle, mask[:1])[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_orient_conf".format(viewpoint.image_name), vis_orient(orient_angle, orient_conf_vis)[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_mask".format(viewpoint.image_name), F.pad(gt_mask, (0, 0, 0, 0, 0, 3-gt_mask.shape[0]), 'constant', 0)[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_orient".format(viewpoint.image_name), vis_orient(gt_orient_angle, gt_mask[:1])[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_orient_conf".format(viewpoint.image_name), vis_orient(gt_orient_angle, gt_orient_conf_vis)[None], global_step=iteration)
                    # 计算 L1、交叉熵、方向损失、PSNR
                    l1_test += l1_loss(image * gt_mask[:1], gt_image * gt_mask[:1]).mean().double()  # L1 损失：衡量预测图像与真实图像的像素差距
                    ce_test += ce_loss(mask[:1], gt_mask[:1]).mean().double()  # 交叉熵 (CE) 损失：衡量预测的 mask 误差
                    or_test += or_loss(orient_angle, gt_orient_angle, mask=gt_mask[:1], weight=gt_orient_conf).mean().double()  # OR 损失：衡量方向角度预测误差
                    psnr_test += psnr(image * gt_mask[:1], gt_image * gt_mask[:1]).mean().double()  # PSNR：衡量图像质量
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                ce_test /= len(config['cameras'])
                or_test /= len(config['cameras'])
                # 记录 & 打印测试结果
                print("\n[ITER {}] Evaluating {}: L1 {} CE {} OR {} PSNR {}".format(iteration, config['name'], l1_test, ce_test, or_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ce_loss', ce_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - or_loss', or_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    # 初始化自定义的参数管理类
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 15_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 15_000, 20_000])
    #parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 2_000, 10_000, 15_000, 20_000])
    #parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 2_000, 10_000, 15_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 15_000, 20_000])
    #parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 2_000, 10_000, 15_000, 20_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)  #允许从特定的 checkpoint 开始训练，支持普通模型 (start_checkpoint) 和头发模型 (start_checkpoint_hair)。
    parser.add_argument("--start_checkpoint_hair", type=str, default = None)
    parser.add_argument("--hair_conf_path", type=str, default = None)
    parser.add_argument("--model_path_hair", type=str, default = None)
    parser.add_argument("--pointcloud_path_head", type=str, default = None)
    parser.add_argument("--is_step2_2", action='store_true', default=False)
    parser.add_argument("--checkpoint_hair_ljy", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])  # 解析所有参数并存入 args，然后确保 iterations 也加入 save_iterations
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path_hair)

    # Configuration of hair strands
    # 读取头发配置文件
    with open(args.hair_conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('DATASET_TYPE', 'monocular')
        opt_hair = yaml.load(replaced_conf, Loader=yaml.Loader)

    # Initialize system state (RNG)
    # 训练前的初始化
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly) # 启用 PyTorch 的 autograd 异常检测（如果 --detect_anomaly 被设置）。
    training(lp.extract(args), op.extract(args), opt_hair, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.model_path_hair, args.pointcloud_path_head, args.start_checkpoint, args.start_checkpoint_hair, args.debug_from, args.is_step2_2, args.checkpoint_hair_ljy)

    # All done
    print("\nTraining complete.")