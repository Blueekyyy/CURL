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
from scene import Scene, GaussianModel, GaussianModelCurves
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


"""
training主要用于训练头发的高斯建模 (Gaussian Model) 以及头部点云的渲染。整个训练过程涉及数据初始化、网络训练、渲染、损失计算、优化器更新、日志记录和模型保存。
"""
def training(dataset, opt, opt_hair, pipe, testing_iterations, saving_iterations, checkpoint_iterations, model_path_curves, pointcloud_path_head, checkpoint, checkpoint_hair, debug_from, is_feature_dc_rest, checkpoint_curve, sobel_loss):
    """
    1. 训练前的初始化
        初始化 TensorBoard 记录 (tb_writer) 。
        创建 gaussians (用于头部) 和 gaussians_hair (用于头发) 的高斯模型。
        scene 负责加载数据和点云 (pointcloud_path_head)
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, model_path_curves)
    #import pdb; pdb.set_trace()
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians_hair = GaussianModelCurves(dataset.source_path, dataset.flame_mesh_dir, opt_hair, dataset.sh_degree)
    scene = Scene(dataset, gaussians, pointcloud_path=pointcloud_path_head, load_iteration=-1)
    """
    2. 载入已有的模型或检查点
        gaussians.training_setup(opt): 配置头部的高斯点云模型的训练。
        载入 checkpoint_hair 进行头发的初始化。
        设定 gaussians_hair (高斯曲线) 的训练参数。
    """
    #=== Test ===
    if is_feature_dc_rest:
        gaussians.training_setup_ljy(opt)
    else:
        gaussians.training_setup(opt)
    #传第二步GaussianModelHair的训练结果到GaussianModelCurve（都包含球谐函数颜色）
    #第一步训练的GaussianModel的属性是怎么影响后面两个模型的？在render_hair里cat两个模型的张量到一起
    model_params, _ = torch.load(checkpoint_hair)
    gaussians_hair.create_from_pcd(dataset.source_path, model_params, 30_000, gaussians.spatial_lr_scale)
    #gaussians_hair.create_from_pcd(dataset.source_path, model_params, 50_000, gaussians.spatial_lr_scale)
    #=== Test ===
    if is_feature_dc_rest:
        gaussians_hair.training_setup_ljy(opt)
    else:
        gaussians_hair.training_setup(opt)
    
    if checkpoint:
        model_params, _ = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    #=== Test ===
    if(checkpoint_curve):
        model_params, _ = torch.load(checkpoint_curve)
        gaussians_hair.restore_ljy(model_params, opt)


    if dataset.trainable_cameras:
        print(f'Loading optimized cameras from iter {scene.loaded_iter}')
        params_cam_rotation, params_cam_translation, params_cam_fov = pkl.load(open(scene.model_path + "/cameras/" + str(scene.loaded_iter) + ".pkl", 'rb'))
        for k in scene.train_cameras.keys():
            for camera in scene.train_cameras[k]:
                if dataset.trainable_cameras:
                    camera._rotation_res.data = params_cam_rotation[camera.image_name]
                    camera._translation_res.data = params_cam_translation[camera.image_name]
                if dataset.trainable_intrinsics:
                    camera._fov_res.data = params_cam_fov[camera.image_name]

    """
    3. 预处理数据
        计算头部高斯模型的一些预处理数据 (位置、透明度、缩放、旋转等)。
        这些数据在后续训练中用作参考，以减少计算开销。
    """
    with torch.no_grad():
        # Head gaussians data
        gaussians.mask_precomp = gaussians.get_label[..., 0] < 0.5
        gaussians.xyz_precomp = gaussians.get_xyz[gaussians.mask_precomp].detach()
        gaussians.opacity_precomp = gaussians.get_opacity[gaussians.mask_precomp].detach()
        gaussians.scaling_precomp = gaussians.get_scaling[gaussians.mask_precomp].detach()
        gaussians.rotation_precomp = gaussians.get_rotation[gaussians.mask_precomp].detach()
        gaussians.cov3D_precomp = gaussians.get_covariance(1.0)[gaussians.mask_precomp].detach()
        gaussians.shs_view = gaussians.get_features[gaussians.mask_precomp].detach().transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1)**2)

    """
        设置背景颜色 (白色或黑色)。
    """
    bg_color = [1, 1, 1, 0, 0, 0, 0, 0, 0, 100] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    """
    4. 训练循环
        使用 tqdm 进度条显示训练进度。
        迭代 opt.iterations 次。
    """
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    """
    # === Test ===
    #import pdb;pdb.set_trace()
    output_dir = f"{dataset.source_path}/train_strands_imgs"  # 👈 替换成你要保存的目录
    os.makedirs(output_dir, exist_ok=True)
    """
    # === Test ===
    loss_fn = PerceptualLoss(layer='relu3_3').cuda()
    #edge_loss_fn = EdgeLoss()
    edge_loss_fn = CannyEdgeLoss(low_threshold=30, high_threshold=150)

    for iteration in range(first_iter, opt.iterations + 1):
        """
        4.1 处理网络通信
            代码检查是否有 network_gui 连接（可能用于远程控制训练）。
            如果连接存在，就会尝试从 network_gui.receive() 获取参数。
            net_image_bytes 可能用于可视化网络训练进程。
        """
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
        """
        4.2 训练步骤
            记录训练时间 (iter_start.record())。
            重新初始化 gaussians_hair 。
            更新 gaussians_hair 的学习率。
        """
        iter_start.record()

        gaussians_hair.initialize_gaussians_hair()
        gaussians_hair.update_learning_rate(iteration)

        # Pick a random Camera
        """
            随机选择一个相机视角进行训练。
        """
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        """
            使用 render_hair 渲染当前视角。
            提取渲染图像、掩码、方向角、方向置信度。
        """
        #import pdb;pdb.set_trace()
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
        """
        5. 计算损失
            载入 Ground Truth (GT) 影像、掩码、方向角和置信度。
        """
        gt_image = viewpoint_cam.original_image.cuda()
        gt_mask = viewpoint_cam.original_mask.cuda()
        gt_orient_angle = viewpoint_cam.original_orient_angle.cuda()
        gt_orient_conf = viewpoint_cam.original_orient_conf.cuda()
        #=== Test ===
        gt_pct_loss = viewpoint_cam.original_pct_loss.cuda()
        gt_edge_loss = viewpoint_cam.original_edge_loss.cuda()
        
        """
            计算 L1 损失、结构相似性损失 (SSIM)、掩码损失、方向损失 (Lorient) 以及额外的 Lsds 约束。
            计算总损失并进行反向传播 (loss.backward())。

        """
        Ll1 = l1_loss(image, gt_image)
        Lssim = (1.0 - ssim(image, gt_image))
        Lmask = l1_loss(mask, gt_mask)
    
        orient_weight = torch.ones_like(gt_mask[:1])
        if opt.use_gt_orient_conf: orient_weight = orient_weight * gt_orient_conf
        if not opt.train_orient_conf: orient_conf = None
        Lorient = or_loss(orient_angle, gt_orient_angle, orient_conf, weight=orient_weight, mask=gt_mask[:1])

        if torch.isnan(Lorient).any(): Lorient = torch.zeros_like(Ll1)

        #Lsds = gaussians_hair.Lsds if hasattr(gaussians_hair, 'Lsds') and gaussians_hair.Lsds is not None else torch.zeros_like(Ll1)
        Lsds = torch.tensor(0.0)
        #=== Perceptual loss ===
        #Lperceptual = loss_fn(image * gt_mask[1:], gt_image * gt_mask[1:], gt_mask[1:])
        Lperceptual = loss_fn(image * gt_mask[1:], gt_pct_loss, gt_mask[1:])   # input, target, mask
        #=== Edge loss===
        #Ledge = edge_loss_fn(image * gt_mask[1:], gt_image * gt_mask[1:])
        Ledge = edge_loss_fn(image * gt_mask[1:], gt_edge_loss) #pred_img, gt_img

        if torch.isnan(Lperceptual).any(): Lperceptual = torch.zeros_like(Ll1)
        if torch.isnan(Ledge).any(): Ledge = torch.zeros_like(Ll1)

        if sobel_loss:
            loss = (
                Ll1 * opt.lambda_dl1 + 
                Lssim * opt.lambda_dssim
            )
        else:
            loss = (
                Ll1 * opt.lambda_dl1 + 
                Lssim * opt.lambda_dssim + 
                Lmask * opt.lambda_dmask + 
                Lorient * opt.lambda_dorient +
                #Lsds * opt.lambda_dsds +
                Lperceptual * opt.lambda_dperceptual +
                Ledge * opt.lambda_dedge
            )
        loss.backward()

        iter_end.record()

        # Optimizer step
        """
        6. 参数更新
            如果梯度 NaN，则跳过该步。
            运行优化器 step() 进行参数更新。
        """
        if iteration < opt.iterations:
            for param in [gaussians_hair._dirs, gaussians_hair._features_dc, gaussians_hair._features_rest]:
                if param.grad is not None and param.grad.isnan().any():
                    #gaussians_hair.optimizer.zero_grad(set_to_none = True)
                    #print('NaN during backprop was found, skipping iteration...')
                    #=== Test ===
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)
                    print('NaN during backprop was found, set to 0.0...')

            gaussians_hair.optimizer.step()
            gaussians_hair.optimizer.zero_grad()
        
        """
        if is_feature_dc_rest:
            # Optimizer step
            if iteration < opt.iterations:
                for param in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._opacity, gaussians._label, gaussians._scaling, gaussians._rotation]:
                    if param.grad is not None and param.grad.isnan().any():
                        #gaussians.optimizer.zero_grad(set_to_none = True)
                        #print('NaN during backprop was found, skipping iteration...')
                        #=== Test ===
                        param.grad = torch.nan_to_num(param.grad, nan=0.0)
                        print('NaN during backprop was found, set to 0.0...')

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        """

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            """
            7. 记录训练进度
                记录训练日志到 TensorBoard。
            """
            training_report(tb_writer, iteration, Ll1, Lmask, Lorient, Lsds, loss, l1_loss, or_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians_hair, render_hair, (pipe, background))
            """
            8. 保存模型
                在指定的 saving_iterations 时保存高斯模型。
                在 checkpoint_iterations 时保存模型检查点 (.pth 文件)。
            """
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(model_path_curves + "/checkpoints", exist_ok=True)
                torch.save((gaussians_hair.capture(), iteration), model_path_curves + "/checkpoints/" + str(iteration) + ".pth")

"""
prepare_output_and_logger创建输出目录并初始化 TensorBoard 记录器，用于记录训练过程中的日志和超参数配置。
"""
def prepare_output_and_logger(args, model_path_curves):    
    """
    1. 确定 model_path_curves (输出目录)
        如果 model_path_curves 未指定，则创建一个唯一的路径：
        如果 环境变量 OAR_JOB_ID 存在（通常用于集群计算），使用该 ID 作为唯一标识。
        否则，生成一个随机的 UUID，并截取前 10 个字符作为目录名。
        最终，输出目录路径设定为 "./output/{unique_str}"。
    """
    if not model_path_curves:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        model_path_curves = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    """
    2. 创建输出文件夹
        打印 目录路径，方便用户查看日志存储位置。
        使用 os.makedirs() 创建目录：
            exist_ok=True 表示如果目录已存在，则不会报错。
    """
    print("Output folder: {}".format(model_path_curves))
    os.makedirs(model_path_curves, exist_ok = True)
    """
    3. 记录超参数
        将训练参数 (args) 以字符串格式保存到 "cfg_args" 文件：
            vars(args): 将 args 转换为字典形式。
            Namespace(**vars(args)): 重新转换为 Namespace 以确保格式正确。
            cfg_log_f.write(...): 记录到文件，方便实验复现。
    """
    with open(os.path.join(model_path_curves, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    """
    4. 创建 TensorBoard 记录器
        检查 TensorBoard 是否可用 (TENSORBOARD_FOUND 变量)：
            如果可用，创建 SummaryWriter，用于记录训练数据，可以在 TensorBoard 进行可视化。
            否则，打印提示信息，说明不进行日志记录。
    """
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path_curves)
    else:
        print("Tensorboard not available: not logging progress")
    """
    5. 返回 TensorBoard 记录器
        返回 tb_writer，用于在训练过程中记录损失、图像、参数等信息。
    """
    return tb_writer

"""
training_report主要用于 记录和评估训练进度，并将相关数据写入 TensorBoard 以便可视化。它的主要功能包括：
    记录训练损失（L1损失、掩码损失、方向损失等）
    在测试集和训练集中进行模型评估（计算 PSNR、L1、掩码、方向损失）
    保存渲染结果和 Ground Truth 到 TensorBoard
    记录高斯模型的统计信息（透明度、标签、点数）
"""
def training_report(tb_writer, iteration, Ll1, Lmask, Lorient, Lsds, loss, l1_loss, or_loss, elapsed, testing_iterations, scene : Scene, gaussians_hair, renderFunc, renderArgs):
    """
    1. 记录训练损失
        如果 tb_writer 存在（即 TensorBoard 可用），则记录：
            Ll1: L1 损失
            Lmask: 掩码损失
            Lorient: 方向损失
            Lsds: 方向场平滑损失
            loss: 总损失
            elapsed: 训练迭代的执行时间
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ce_loss', Lmask.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/or_loss', Lorient.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/df_loss', Lsds.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    """
    2. 进行测试评估
        如果当前 iteration 在 testing_iterations 之中：
            清空 GPU 缓存，防止显存溢出。
            定义两个评估集：
                test（测试集）
                train（从训练集中选取部分样本）
    """
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        """
        3. 遍历评估集，计算指标
            遍历测试集和训练集：
                初始化评估指标（L1损失、掩码损失、方向损失、PSNR）。
        """
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                Ll1_test = 0.0
                Lmask_test = 0.0
                Lorient_test = 0.0
                psnr_test = 0.0
                """
                    对每个相机视角进行渲染：
                        调用 renderFunc() 生成渲染结果。
                        裁剪 结果，使其范围在 [0,1] 之间，防止溢出。
                        计算方向置信度可视化 (orient_conf_vis)。
                """
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, gaussians_hair, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    mask = torch.clamp(render_pkg["mask"], 0.0, 1.0)
                    orient_angle = torch.clamp(render_pkg["orient_angle"], 0.0, 1.0)
                    orient_conf = render_pkg["orient_conf"]
                    orient_conf_vis = (1 - 1 / (orient_conf + 1)) * mask[:1]
                    """
                        获取 Ground Truth 数据：
                            gt_image: 真实图片
                            gt_mask: 真实掩码
                            gt_orient_angle: 真实方向
                            gt_orient_conf: 真实方向置信度
                    """
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_mask = torch.clamp(viewpoint.original_mask.to("cuda"), 0.0, 1.0)
                    gt_orient_angle = torch.clamp(viewpoint.original_orient_angle.to("cuda"), 0.0, 1.0)
                    gt_orient_conf = viewpoint.original_orient_conf.to("cuda")
                    gt_orient_conf_vis = (1 - 1 / (gt_orient_conf + 1)) * gt_mask[:1]
                    """
                    4. 记录渲染结果
                        在 TensorBoard 中保存渲染结果：
                            render：渲染的图片
                            render_mask：渲染的掩码
                            render_orient：渲染的方向角
                            render_orient_conf：方向置信度可视化
                    """
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_mask".format(viewpoint.image_name), F.pad(mask, (0, 0, 0, 0, 0, 3-mask.shape[0]), 'constant', 0)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_orient".format(viewpoint.image_name), vis_orient(orient_angle, mask[:1])[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_orient_conf".format(viewpoint.image_name), vis_orient(orient_angle, orient_conf_vis)[None], global_step=iteration)
                        """
                            在 TensorBoard 记录 Ground Truth（仅在第一次测试时）。
                        """
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_mask".format(viewpoint.image_name), F.pad(gt_mask, (0, 0, 0, 0, 0, 3-gt_mask.shape[0]), 'constant', 0)[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_orient".format(viewpoint.image_name), vis_orient(gt_orient_angle, gt_mask[:1])[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_orient_conf".format(viewpoint.image_name), vis_orient(gt_orient_angle, gt_orient_conf_vis)[None], global_step=iteration)
                    """
                    5. 计算损失和 PSNR
                        计算 L1 损失、掩码损失、方向损失、PSNR（峰值信噪比）。
                    """
                    Ll1_test += l1_loss(image, gt_image).mean().double()
                    Lmask_test += l1_loss(mask, gt_mask).mean().double()
                    Lorient_test += or_loss(orient_angle, gt_orient_angle, mask=gt_mask[:1], weight=gt_orient_conf).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                Ll1_test /= len(config['cameras'])
                Lmask_test /= len(config['cameras'])
                Lorient_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])
                """
                6. 记录评估指标
                    打印和记录评估指标。
                """
                print("\n[ITER {}] Evaluating {}: L1 {} CE {} OR {} PSNR {}".format(iteration, config['name'], Ll1_test, Lmask_test, Lorient_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', Ll1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ce_loss', Lmask_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - or_loss', Lorient_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        """
        7. 记录高斯模型信息
            记录高斯点的透明度、标签、数量。
        """
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/label_histogram", scene.gaussians.get_label, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

"""
训练脚本的主入口，主要执行 参数解析、配置加载、系统初始化 和 训练过程启动。
"""
"""
1. 解析命令行参数
    检查是否作为主脚本运行（__name__ == "__main__"）。
    创建参数解析器，用于从命令行获取参数。
"""
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    """
    添加模型、优化、流水线相关参数
        ModelParams(parser): 解析 模型参数。
        OptimizationParams(parser): 解析 优化参数。
        PipelineParams(parser): 解析 流水线参数。
    """
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    """
    添加额外的训练配置
    """
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    #parser.add_argument('--ip', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    """
        指定测试、保存、检查点存储的迭代步。
    """
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 1_000, 5_000, 10_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[100, 1_000, 5_000, 10_000, 15_000, 30_000])
    """
        加载 checkpoint（恢复训练）
        指定数据和模型路径 
    """
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_checkpoint_hair", type=str, default = None)
    parser.add_argument("--hair_conf_path", type=str, default = None)
    parser.add_argument("--model_path_curves", type=str, default = None)
    parser.add_argument("--pointcloud_path_head", type=str, default = None)
    parser.add_argument("--is_feature_dc_rest", action="store_true", default = False)
    parser.add_argument("--start_checkpoint_curve", type=str, default = None)
    parser.add_argument("--sobel_loss", action="store_true", default = False)
    """
        解析命令行参数
        追加 iterations 到 save_iterations（保存最终模型）。
    """
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    """
    2. 输出当前优化的模型路径
        打印优化的模型路径，便于跟踪。
    """
    print("Optimizing " + args.model_path_curves)

    # Configuration of hair strands
    """
    3. 加载头发的 YAML 配置
        读取 YAML 配置文件。
        替换 DATASET_TYPE 为 monocular，适用于单目数据。
    """
    with open(args.hair_conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('DATASET_TYPE', 'monocular')
        opt_hair = yaml.load(replaced_conf, Loader=yaml.Loader)

    # Initialize system state (RNG)
    """
    4. 初始化系统状态
        初始化随机数种子，保证可复现性。
    """
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    """
    5. 启动 GUI 服务器
        开启 GUI 服务器（可用于可视化训练进度）。
    """
    #import pdb;pdb.set_trace()
    network_gui.init(args.ip, args.port)
    """
    6. 配置 PyTorch 运行环境
        启用 PyTorch 异常检测，可帮助调试梯度计算问题。
    """
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    """
    7. 启动训练
        传入解析好的参数，启动 training() 训练函数：
            lp.extract(args): 提取 模型参数。
            op.extract(args): 提取 优化参数。
            opt_hair: 头发的 YAML 配置。
            pp.extract(args): 提取 流水线参数。
            args.test_iterations: 训练过程中的测试迭代点。
            args.save_iterations: 训练过程中的保存点。
            args.checkpoint_iterations: 训练过程中的 checkpoint 保存点。
            args.model_path_curves: 模型路径。
            args.pointcloud_path_head: 头部点云路径。
            args.start_checkpoint: 头部点云 checkpoint。
            args.start_checkpoint_hair: 头发点云 checkpoint。
            args.debug_from: 设定从哪一步开始调试。
    """
    training(lp.extract(args), op.extract(args), opt_hair, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.model_path_curves, args.pointcloud_path_head, args.start_checkpoint, args.start_checkpoint_hair, args.debug_from, args.is_feature_dc_rest, args.start_checkpoint_curve, args.sobel_loss)

    # All done
    """
    8. 训练完成
        训练结束后，打印信息。
    """
    print("\nTraining complete.")
