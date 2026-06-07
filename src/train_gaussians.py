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
from utils.loss_utils import l1_loss, ssim, or_loss, PerceptualLoss, EdgeLoss, CannyEdgeLoss
from utils.general_utils import get_expon_lr_func
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, vis_orient
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import pickle as pkl
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import imageio
import numpy as np


"""
一个基于 3D 高斯建模的神经渲染训练循环，主要用于 优化点云数据、调整摄像机参数，并支持 训练、测试、保存、检查点恢复 等功能。
整体流程：
    1.初始化
        设置日志
        加载 GaussianModel（高斯点云）
        初始化 Scene（场景）
        加载 checkpoint（断点恢复）
    2.设置可训练摄像机参数
        旋转、平移、焦距（可选）
    3.训练循环
        连接 GUI（用于实时可视化）
        更新学习率
        选择训练视角
        进行渲染
        计算损失
        反向传播 & 梯度更新
        记录训练日志
        进行密度调整（Densification）
        保存检查点 & 训练结果
"""
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, is_feature_dc_rest, sobel_loss, pointcloud_path_head=None, is_densify=True):
    """
    1. 初始化
            tb_writer = prepare_output_and_logger(dataset): 日志记录
            gaussians = GaussianModel(dataset.sh_degree): 初始化高斯点云模型
            scene = Scene(dataset, gaussians): 初始化 3D 场景
            gaussians.training_setup(opt): 设置训练参数
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    #scene = Scene(dataset, gaussians)
    scene = Scene(dataset, gaussians, pointcloud_path=pointcloud_path_head)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if is_feature_dc_rest:
            gaussians.restore_ljy(model_params, opt)
            first_iter = 0
        else:
            gaussians.restore(model_params, opt)
    """
    2. 训练摄像机参数
            如果摄像机是可训练的，则创建 rotation、translation、fov 的参数字典。
    """
    if dataset.trainable_cameras or dataset.trainable_intrinsics:
        params_cam_rotation = {}
        params_cam_translation = {}
        params_cam_fov = {}
        for k in scene.train_cameras.keys():
            for camera in scene.train_cameras[k]:
                if dataset.trainable_cameras:
                    params_cam_rotation[camera.image_name] = camera._rotation_res
                    params_cam_translation[camera.image_name] = camera._translation_res
                if dataset.trainable_intrinsics:
                    params_cam_fov[camera.image_name] = camera._fov_res
        """
            使用 Adam 优化器训练摄像机的旋转、平移、焦距参数。
        """
        params_cam = list(params_cam_rotation.values()) + list(params_cam_translation.values()) + list(params_cam_fov.values())
        l = [
            {'params': list(params_cam_rotation.values()), 'lr': opt.cam_rotation_lr, "name": "rotation"},
            {'params': list(params_cam_translation.values()), 'lr': opt.cam_translation_lr_init * gaussians.spatial_lr_scale, "name": "translation"},
            {'params': list(params_cam_fov.values()), 'lr': opt.cam_fov_lr, "name": "fov"}
        ]

        optimizer_cameras = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        translation_scheduler_args = get_expon_lr_func(lr_init=opt.cam_translation_lr_init * gaussians.spatial_lr_scale,
                                                       lr_final=opt.cam_translation_lr_final * gaussians.spatial_lr_scale,
                                                       max_steps=opt.cam_lr_max_steps)

    """
    3. 设置训练背景
        白色/黑色背景（根据数据集选择）。
    """
    #import pdb;pdb.set_trace()
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
    output_dir = f"{dataset.source_path}/train_gaussians_imgs"  # 👈 替换成你要保存的目录
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
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        """
        6. 选择训练视角
            从训练摄像机列表随机选择一个视角。
        """
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        """
        7. 渲染
            渲染当前视角的图像
            render_pkg 包含：
                image（渲染图像）
                mask（遮罩）
                orient_angle（方向角）
                orient_conf（方向置信度）
        """
        #import pdb;pdb.set_trace()
        #print("Res:", viewpoint_cam.image_width, viewpoint_cam.image_height)
        #print("Res:", viewpoint_cam.width, viewpoint_cam.height)
        #print("Gaussians:", gaussians.get_xyz.shape)
        #if iteration % 100 == 0:
        #    torch.cuda.empty_cache()
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    
        image = render_pkg["render"]  # torch.Size([3, 960, 540])
        mask = render_pkg["mask"] # torch.Size([2, 960, 540])
        orient_angle = render_pkg["orient_angle"]  # torch.Size([1, 960, 540])
        orient_conf = render_pkg["orient_conf"] # torch.Size([1, 960, 540])
        viewspace_point_tensor = render_pkg["viewspace_points"]  # torch.Size([40318, 3])
        visibility_filter = render_pkg["visibility_filter"] # torch.Size([40318])
        radii = render_pkg["radii"] # torch.Size([40318])
        

        """
        # === Test ===
        if iteration % 1000 == 0:
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


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()  # torch.Size([3, 960, 540])
        gt_mask = viewpoint_cam.original_mask.cuda()  # torch.Size([2, 960, 540])
        gt_orient_angle = viewpoint_cam.original_orient_angle.cuda()  # torch.Size([1, 960, 540])
        gt_orient_conf = viewpoint_cam.original_orient_conf.cuda()  # torch.Size([1, 960, 540])
        #=== Perceptual loss ===
        #gt_perceptual_feat = viewpoint_cam.original_perceptual_feat.cuda()
        #import pdb;pdb.set_trace()
        gt_pct_loss = viewpoint_cam.original_pct_loss.cuda()
        gt_edge_loss = viewpoint_cam.original_edge_loss.cuda()

        """
        8. 计算损失
            计算 L1 损失（Ll1）
            计算 结构相似性损失（SSIM）
            计算 遮罩损失
            计算 方向角损失
            综合所有损失，进行加权求和
        """
        #import pdb;pdb.set_trace()
        Ll1 = l1_loss(image, gt_image, mask=gt_mask[1:].detach())  #mask为1通道，与3通道rgb相乘3次，结果仍然是3通道
        Lssim = (1.0 - ssim(image * gt_mask[1:], gt_image * gt_mask[1:]))
        Lmask = l1_loss(mask, gt_mask)  #两部分mask都监督（body和hair）
        #=== Perceptual loss ===
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #import pdb;pdb.set_trace()
        #image = load_image(image)
        #gt_image = load_image(gt_image)
        #import pdb;pdb.set_trace()
        Lperceptual = loss_fn(image * gt_mask[1:], gt_pct_loss, gt_mask[1:])   # input, target, mask
        #Lperceptual = loss_fn(image * gt_mask[1:], gt_image * gt_mask[1:], gt_mask[1:])
        #=== Edge loss===
        #import pdb;pdb.set_trace()
        #Ledge = edge_loss_fn(image * gt_mask[1:], gt_image * gt_mask[1:])
        Ledge = edge_loss_fn(image * gt_mask[1:], gt_edge_loss) #pred_img, gt_img


        orient_weight = torch.ones_like(gt_mask[:1]) * gt_orient_conf
        Lorient = or_loss(orient_angle, gt_orient_angle, orient_conf, weight=orient_weight, mask=gt_mask[:1])
        
        if torch.isnan(Lorient).any(): Lorient = torch.zeros_like(Ll1)

        if sobel_loss:
            loss = (
                Ll1 * opt.lambda_dl1 + 
                Lssim * opt.lambda_dssim +
                Lmask * opt.lambda_dmask +
                Lorient * opt.lambda_dorient
            )
        else:
            loss = (
                Ll1 * opt.lambda_dl1 + 
                Lssim * opt.lambda_dssim + 
                Lmask * opt.lambda_dmask + 
                Lorient * opt.lambda_dorient + 
                Lperceptual * opt.lambda_dperceptual +
                Ledge * opt.lambda_dedge
            )
       
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            """
            10. 记录训练进度
                每 10 次迭代更新一次进度条
            """
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, Lmask, Lorient, loss, l1_loss, or_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                #import pdb;pdb.set_trace()
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            
            """
            # Densification
            #import pdb;pdb.set_trace()
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                #跟踪图像空间中的最大半径以进行修剪
                #IndexError: The shape of the mask [884376] at index 0 does not match the shape of the indexed tensor [59905] at index 0
                #import pdb;pdb.set_trace()
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #import pdb;pdb.set_trace()
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)  # viewspace_point_tensor: [points_num, xyz]; visibility_filter (bool) : [points_num]

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    #import pdb;pdb.set_trace()
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.000, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            """
            
            # Densification_ljy
            densify_iteration_ljy = 0  # densify nums
            densify_total_ljy = 4
            densify_start_iteration_ljy = 9999
            #densify_points = 500_000  (VLM)
            densify_points = 1_300_000  #（之前用的）
            #densify_points = 1_600_000
            #densify_points = 1_700_000
            #densify_points = 1_800_000
            #densify_points = 2_000_000
            #densify_points = 2_200_000
            
            if is_densify:
                #import pdb;pdb.set_trace()
                #if iteration > opt.densify_until_iter_ljy:
                if iteration > densify_start_iteration_ljy:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)  # viewspace_point_tensor: [points_num, xyz]; visibility_filter (bool) : [points_num]

                    if iteration > opt.densify_from_iter_ljy and iteration % opt.densification_interval_ljy == 0 and densify_iteration_ljy < 4:
                    #if densify_iteration_ljy < densify_total_ljy:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        for i in range(6):
                            label_sum_5 = sum(gaussians.get_label[..., 0] >= 0.55)
                            label_sum_6 = sum(gaussians.get_label[..., 0] >= 0.65)
                            label_sum_7 = sum(gaussians.get_label[..., 0] >= 0.75)
                            label_sum_8 = sum(gaussians.get_label[..., 0] >= 0.85)
                            label_sum_9 = sum(gaussians.get_label[..., 0] >= 0.95)
                            gaussian_num_xyz = gaussians._xyz.shape[0]

                            print(f"label_sum_5:{label_sum_5}")
                            print(f"label_sum_6:{label_sum_6}")
                            print(f"label_sum_7:{label_sum_7}")
                            print(f"label_sum_8:{label_sum_8}")
                            print(f"label_sum_9:{label_sum_9}")
                            print(f"gaussian_num_xyz:{gaussian_num_xyz}")

                            if(gaussian_num_xyz + label_sum_5 <= densify_points):
                                gaussians.densify_and_prune_ljy(opt.densify_grad_threshold, 0.000, scene.cameras_extent, size_threshold)
                                print("now label_sum_5")
                            elif(gaussian_num_xyz + label_sum_6 <= densify_points):
                                gaussians.densify_and_prune_ljy(opt.densify_grad_threshold, 0.000, scene.cameras_extent, size_threshold)
                                print("now label_sum_6")
                            elif(gaussian_num_xyz + label_sum_7 <= densify_points):
                                gaussians.densify_and_prune_ljy(opt.densify_grad_threshold, 0.000, scene.cameras_extent, size_threshold)
                                print("now label_sum_7")
                            elif(gaussian_num_xyz + label_sum_8 <= densify_points):
                                gaussians.densify_and_prune_ljy(opt.densify_grad_threshold, 0.000, scene.cameras_extent, size_threshold)
                                print("now label_sum_8")
                            elif(gaussian_num_xyz + label_sum_9 <= densify_points):
                                gaussians.densify_and_prune_ljy(opt.densify_grad_threshold, 0.000, scene.cameras_extent, size_threshold)
                                print("now label_sum_9")
        
                        densify_iteration_ljy += 1
                    
                    #if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    #    gaussians.reset_opacity()
                    if (iteration == 10000):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        os.makedirs(scene.model_path + "/checkpoints", exist_ok=True)
                        os.makedirs(scene.model_path + "/cameras", exist_ok=True)
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/checkpoints/" + str(iteration) + ".pth")
                        if dataset.trainable_cameras:
                            pkl.dump((params_cam_rotation, params_cam_translation, params_cam_fov), open(scene.model_path + "/cameras/" + str(iteration) + ".pkl", 'wb'))
                        projection_all = {}
                        for camera in scene.train_cameras[1.0]:
                            projection_all[camera.image_name] = camera.full_proj_transform.cpu()
                        pkl.dump(projection_all, open(scene.model_path + "/cameras/" + str(iteration) + "_matrices.pkl", 'wb'))
                        print(f"**********************Gaussian Ellipsoids:{gaussians._xyz.shape}")
                        print(f"**********************label >= 0.5:{sum(gaussians.get_label[..., 0] >= 0.5)}")
                        sys.exit(0)

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

            if iteration < opt.iterations_cam and dataset.trainable_cameras:
                ''' Learning rate scheduling per step '''
                for param_group in optimizer_cameras.param_groups:
                    if param_group["name"] == "translation":
                        lr = translation_scheduler_args(iteration)
                        param_group['lr'] = lr
                
                for param in params_cam:
                    if param.grad is not None and param.grad.isnan().any():
                        #optimizer_cameras.zero_grad(set_to_none = True)
                        #print('NaN during backprop was found, skipping iteration...')
                        #=== Test ===
                        param.grad = torch.nan_to_num(param.grad, nan=0.0)
                        print('NaN during backprop was found, set to 0.0...')

                optimizer_cameras.step()
                optimizer_cameras.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(scene.model_path + "/checkpoints", exist_ok=True)
                os.makedirs(scene.model_path + "/cameras", exist_ok=True)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/checkpoints/" + str(iteration) + ".pth")
                if dataset.trainable_cameras:
                    pkl.dump((params_cam_rotation, params_cam_translation, params_cam_fov), open(scene.model_path + "/cameras/" + str(iteration) + ".pkl", 'wb'))
                projection_all = {}
                for camera in scene.train_cameras[1.0]:
                    projection_all[camera.image_name] = camera.full_proj_transform.cpu()
                pkl.dump(projection_all, open(scene.model_path + "/cameras/" + str(iteration) + "_matrices.pkl", 'wb'))

"""
准备训练的输出目录，并初始化 TensorBoard 进行日志记录
"""
def prepare_output_and_logger(args): 
    """
    1. 设置 model_path（输出目录）
        如果 args.model_path 为空，则创建一个唯一的输出目录：
            如果环境变量 OAR_JOB_ID 存在（可能是集群任务 ID），使用它作为目录名称。
            否则，使用 uuid.uuid4() 生成一个随机唯一 ID（取前 10 位），作为 model_path。
            最终，模型输出路径为 ./output/xxxxxxx/（10 位唯一 ID）。
    """   
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    """
    2. 创建输出文件夹
        打印输出目录路径
        创建目录（如果不存在）
    """
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    """
    3. 记录训练配置
        将 args（训练参数）保存到 cfg_args 文件中，方便日后复现实验。
        使用 Namespace(**vars(args)) 确保 args 以可读的格式存储。
    """
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    """
    4. 初始化 TensorBoard
        检查是否安装了 TensorBoard：
            如果 TENSORBOARD_FOUND=True，则 创建 SummaryWriter 记录日志。
            否则，打印警告 "Tensorboard not available: not logging progress"，表示不会记录训练过程。
    """
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    """
    返回值
        返回 TensorBoard 的 SummaryWriter（如果 TensorBoard 可用）。
        如果 TensorBoard 不可用，则返回 None。
    """
    return tb_writer

"""
记录训练损失、执行测试评估并可视化结果:
    记录训练损失（L1、掩码、方向、总损失）
    进行测试（每隔一定 iteration 进行评估）
    渲染测试视角
    计算测试损失
    记录可视化结果到 TensorBoard
    记录场景统计信息
    释放显存，防止 OOM
"""
def training_report(tb_writer, iteration, Ll1, Lmask, Lorient, loss, l1_loss, or_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    """
    1. 记录训练损失
        如果 tb_writer 不为空（即 TensorBoard 可用），它会把当前 iteration（训练步数）的 L1损失、交叉熵损失、方向损失、总损失 以及 迭代时间 记录到 TensorBoard 里。
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ce_loss', Lmask.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/or_loss', Lorient.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    """
    2. 进行测试评估
        如果当前 iteration 在 testing_iterations 列表里，就清理 CUDA 缓存，并进行测试。
    """
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        """
        2.1 获取测试和训练视角
            test 视角：从 scene.getTestCameras() 获取
            train 视角：从 scene.getTrainCameras() 里每 5 个间隔选取一个
        """
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        """
        2.2 遍历不同视角进行测试
            遍历 test 和 train 视角进行测试。
        """
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                """
                初始化测试损失：
                """
                Ll1_test = 0.0
                Lmask_test = 0.0
                Lorient_test = 0.0
                psnr_test = 0.0
                """
                遍历 cameras，用 renderFunc 进行渲染：
                    renderFunc 可能是一个 神经辐射场（NeRF）或高斯渲染 的函数，它会根据 viewpoint（相机视角）和 scene.gaussians（3D 高斯对象）生成渲染结果。
                """
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    """
                    提取 渲染输出：
                    """
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    mask = torch.clamp(render_pkg["mask"], 0.0, 1.0)
                    orient_angle = torch.clamp(render_pkg["orient_angle"], 0.0, 1.0)
                    orient_conf = render_pkg["orient_conf"]
                    orient_conf_vis = (1 - 1 / (orient_conf + 1)) * mask[:1]
                    """
                    提取 真实（ground truth）数据：
                    """
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_mask = torch.clamp(viewpoint.original_mask.to("cuda"), 0.0, 1.0)
                    gt_orient_angle = torch.clamp(viewpoint.original_orient_angle.to("cuda"), 0.0, 1.0)
                    gt_orient_conf = viewpoint.original_orient_conf.to("cuda")
                    gt_orient_conf_vis = (1 - 1 / (gt_orient_conf + 1)) * gt_mask[:1]
                    """
                    3. 记录测试可视化结果
                        如果 tb_writer 存在，且 idx < 5（仅记录前 5 个视角），会把可视化结果存入 TensorBoard：
                    """
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_mask".format(viewpoint.image_name), F.pad(mask, (0, 0, 0, 0, 0, 3-mask.shape[0]), 'constant', 0)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_orient".format(viewpoint.image_name), vis_orient(orient_angle, mask[:1])[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_orient_conf".format(viewpoint.image_name), vis_orient(orient_angle, orient_conf_vis)[None], global_step=iteration)
                        """
                        如果 iteration 是 testing_iterations[0]，还会保存 Ground Truth 数据：
                        """
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_mask".format(viewpoint.image_name), F.pad(gt_mask, (0, 0, 0, 0, 0, 3-gt_mask.shape[0]), 'constant', 0)[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_orient".format(viewpoint.image_name), vis_orient(gt_orient_angle, gt_mask[:1])[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_orient_conf".format(viewpoint.image_name), vis_orient(gt_orient_angle, gt_orient_conf_vis)[None], global_step=iteration)
                    
                    Ll1_test += l1_loss(image, gt_image).mean().double()
                    Lmask_test += l1_loss(mask, gt_mask).mean().double()
                    Lorient_test += or_loss(orient_angle, gt_orient_angle, mask=gt_mask[:1], weight=gt_orient_conf).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                Ll1_test /= len(config['cameras'])
                Lmask_test /= len(config['cameras'])
                Lorient_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} CE {} OR {} PSNR {}".format(iteration, config['name'], Ll1_test, Lmask_test, Lorient_test, psnr_test))
                """
                并存入 TensorBoard：
                """
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', Ll1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ce_loss', Lmask_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - or_loss', Lorient_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        """
        6. 记录场景信息
            记录 3D 高斯 opacity（透明度）分布
            记录 3D 高斯类别分布
            记录 总点数
        """
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/label_histogram", scene.gaussians.get_label, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        """
        7. 释放显存
            清理 CUDA 缓存，避免显存溢出。
        """
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    #import pdb;pdb.set_trace()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    """
            --test_iterations 定义在哪些迭代数上进行模型评估
            --save_iterations 定义在哪些迭代数上保存模型
            --detect_anomaly 可用于 检测 PyTorch 计算异常
            --checkpoint_iterations 控制保存 checkpoint 的时机
    """
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    #parser.add_argument('--ip', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 1_000, 5_000, 10_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[100, 1_000, 5_000, 10_000, 15_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--pointcloud_path_head", type=str, default = None)
    parser.add_argument("--is_densify", action="store_true", default = False)
    parser.add_argument("--is_feature_dc_rest", action="store_true", default = False)
    parser.add_argument("--sobel_loss", action="store_true", default = False)
    """
    2. 解析参数
        parser.parse_args(sys.argv[1:]) 解析 终端输入的参数。
        args.save_iterations.append(args.iterations)：确保 最后一次迭代 也会被保存。
    """
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    """
    初始化随机数种子，确保实验 可复现（通常用于 torch.manual_seed()）。
    """
    safe_state(args.quiet)
    
    # Start GUI server, configure and run training
    """
    4. 启动 GUI 服务器
        初始化 GUI 服务器，用于 远程监控训练状态（可能基于 WebSocket 或 Flask）。
        监听 args.ip 和 args.port，默认 127.0.0.1:6009。
    """
    network_gui.init(args.ip, args.port)
    """
    5. 设置 PyTorch 自动梯度异常检测
        如果 --detect_anomaly 开启，则启用 PyTorch 自动异常检测，可用于调试 NaN 或无效梯度。
    """
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.is_feature_dc_rest, args.sobel_loss, args.pointcloud_path_head, args.is_densify)

    # All done
    print("\nTraining complete.")
