import os
import sys
from pyhocon import ConfigFactory
from pathlib import Path

from PIL import Image
from torchvision.transforms import ToTensor
import torch
import numpy as np
import cv2

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.cameras import  FoVPerspectiveCameras
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.io import load_ply, save_ply, save_obj, load_objs_as_meshes

sys.path.append('../../ext/NeuralHaircut')
from src.models.dataset import Dataset, MonocularDataset
from src.utils.geometry import face_vertices
from NeuS.models.dataset import load_K_Rt_from_P

sys.path.append('../')
sys.path.append('../../ext/NeuralHaircut/k-diffusion')
from utils.general_utils import build_scaling_rotation
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args

import argparse
import yaml
import pickle as pkl
from skimage.draw import polygon

from tqdm import tqdm


#import pdb; pdb.set_trace()
def create_scalp_mask(scalp_mesh, scalp_uvs):
    # 创建一个全黑的掩码图像
    img = np.zeros((256, 256, 1), 'uint8')  # 创建空白掩码
    
    # 遍历头皮网格的每个三角形面
    for i in range(scalp_mesh.faces_packed().shape[0]):  #scalp_mesh.faces_packed() 返回所有的 三角形面（faces），它是一个 N×3 的张量，每一行代表一个三角形的三个顶点索引。  这里的 for 循环遍历所有面片（triangle faces）
        text = scalp_uvs[0][scalp_mesh.faces_packed()[i]].reshape(-1, 2).cpu().numpy() # 获取当前面片（face）的UV坐标
        poly = 255/2 * (text + 1)  # 归一化 UV 坐标到 [0, 255] 像素坐标
        rr, cc = polygon(poly[:,0], poly[:,1], img.shape)  # 获取多边形的像素索引，并填充到掩码图像
        img[rr,cc, :] = (255)  # 在掩码图像上绘制三角形

    scalp_mask = np.flip(img.transpose(1, 0, 2), axis=0)  # 进行坐标变换（可视化对齐）
    return scalp_mask


# 计算网格（mesh）的可见性映射，即 哪些顶点和面片是可见的，以及在 带有头部遮挡（head_mask）的情况下，哪些顶点和面片仍然可见。
def create_visibility_map(camera, rasterizer, mesh, head_mask):
    fragments = rasterizer(mesh, cameras=camera)
    pix_to_face = fragments.pix_to_face
    packed_faces = mesh.faces_packed() 
    packed_verts = mesh.verts_packed() 
    vertex_visibility_map = torch.zeros(packed_verts.shape[0]) 
    vertex_visibility_map_head = torch.zeros(packed_verts.shape[0]) 
    faces_visibility_map = torch.zeros(packed_faces.shape[0])
    faces_visibility_map_head = torch.zeros(packed_faces.shape[0])
    visible_faces = pix_to_face.unique()[1:] # not take -1
    pix_to_face_head = torch.where(head_mask[None, ..., None], pix_to_face, -1 * torch.ones_like(pix_to_face))
    visible_faces_head = pix_to_face_head.unique()[1:] # not take -1
    visible_verts_idx = packed_faces[visible_faces] 
    visible_verts_head_idx = packed_faces[visible_faces_head] 
    unique_visible_verts_idx = torch.unique(visible_verts_idx)
    unique_visible_verts_head_idx = torch.unique(visible_verts_head_idx)
    vertex_visibility_map[unique_visible_verts_idx] = 1.0
    vertex_visibility_map_head[unique_visible_verts_head_idx] = 1.0
    faces_visibility_map[torch.unique(visible_faces)] = 1.0
    faces_visibility_map_head[torch.unique(visible_faces_head)] = 1.0
    pix_to_face_vis = pix_to_face_head.clone() >= 0
    return vertex_visibility_map, vertex_visibility_map_head, faces_visibility_map, faces_visibility_map_head, pix_to_face_vis


def check_visiblity_of_faces(cams, masks, meshRasterizer, full_mesh, flame_mesh_dir, prob_thr, n_views_thr):
    """
    计算 3D 头部网格（mesh）不同视角下的可见性，并生成一个可见性掩码（vis_mask），用于区分 可见头部区域和头发区域。
    """
    # 创建存储可见性数据的文件夹
    # collect visibility maps
    os.makedirs(f'{flame_mesh_dir}/scalp_data/vis', exist_ok=True)
    # 计算每个摄像机视角的可见性
    vis_maps = []
    vis_maps_head = []
    for cam in cams.keys():
        head_mask = masks[cam] >= 0.5
        # 调用 create_visibility_map 计算可见性
        v, vh, _, _, vis = create_visibility_map(cams[cam], meshRasterizer, full_mesh, head_mask)  # _：忽略不需要的返回值（可见面片）。
        # 保存该视角下的可见性图像
        Image.fromarray((vis[0, ..., 0].cpu().numpy() * 255).astype('uint8')).save(f'{flame_mesh_dir}/scalp_data/vis/{cam}.jpg')
        # 存储该视角下的可见性映射
        vis_maps.append(v)
        vis_maps_head.append(vh)
    # 计算所有视角的可见性综合结果
    vis_maps = torch.stack(vis_maps).sum(0).float()
    vis_maps_head = torch.stack(vis_maps_head).sum(0).float()

    #计算头部的可见性概率
    #计算每个面片属于头部的概率
    prob_vis_head = vis_maps_head / vis_maps # probability of faces belonging to visible head
    prob_hair = 1 - prob_vis_head
    #生成最终的可见性掩码
    vis_mask = torch.logical_or(prob_hair > prob_thr, vis_maps / len(cams.keys()) < n_views_thr)

    return vis_mask



def main(args):
    """
    主要功能
    从头部网格 (mesh_head) 提取头皮网格 (scalp_mesh)
    加载摄像机参数 (cams_all)，计算头部可见性
    基于遮挡检测 (vis_vertex_mask)，裁剪出可见头皮网格
    保存裁剪后的头皮 (scalp.obj)
    生成头皮掩码 (dif_mask.png)
    """
    """
    加载头部网格（mesh_head）"
    """
    mesh_head = load_objs_as_meshes([f'{args.flame_mesh_dir}/stage_3/mesh_final.obj'], device=args.device)  # 加载 .obj 文件，创建 PyTorch3D Meshes 物体（即 3D 头部网格），args.flame_mesh_dir 目录包含 FLAME 模型生成的头部网格。

    """
    加载头皮相关数据
    """
    scalp_vert_idx = torch.load(f'{args.project_dir}/data/new_scalp_vertex_idx.pth').long().cuda()  # 头皮 顶点索引 (scalp_vert_idx)：包含头皮区域的顶点索引。
    scalp_faces = torch.load(f'{args.project_dir}/data/new_scalp_faces.pth')[None].cuda()  # 头皮 面索引 (scalp_faces)：存储头皮部分的面片信息。
    scalp_uvs = torch.load(f'{args.project_dir}/data/improved_neural_haircut_uvmap.pth')[None].cuda()  # 头皮 UV 纹理坐标 (scalp_uvs)：用于纹理映射的 UV 坐标。

    """"
    从头部网格提取头皮网格"
    """
    # Convert the head mesh into a scalp mesh
    scalp_verts = mesh_head.verts_packed()[None, scalp_vert_idx]  # 通过 scalp_vert_idx 从完整头部网格中提取 头皮区域的顶点 (scalp_verts)。
    scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0]  # 通过 scalp_faces 提取 头皮区域的面片 (scalp_face_verts)，创建 scalp_mesh。
    scalp_mesh = Meshes(verts=scalp_verts, faces=scalp_faces).cuda()

    """
    读取摄像机参数并处理遮挡检测
    """
    cams_all = pkl.load(open(args.cams_path, 'rb'))  # 通过 pickle 读取 摄像机参数 (cams_all)。
    masks = {}
    cams = {}
    for k in cams_all.keys():
        mask_hair = cv2.dilate(cv2.imread(f'{args.data_dir}/masks_2/hair/{k}.png'), np.ones((5, 5))) / 255. >= 0.5  # 读取 头发和身体的二值掩码 (mask_hair, mask)，计算头部的可见区域 mask_head。
        mask = cv2.dilate(cv2.imread(f'{args.data_dir}/masks_2/body/{k}.png'), np.ones((5, 5))) / 255. >= 0.5  # cv2.dilate() 进行 膨胀操作，确保掩码覆盖更多区域。
        mask_head = np.clip(mask.astype('float32') - mask_hair.astype('float32'), 0, 1)
        masks[k] = torch.from_numpy(mask_head)[:, :, 0].cuda()  # 结果 mask_head 存储到 masks 字典。
        """
        计算摄像机投影矩阵
        """
        intrinsics, pose = load_K_Rt_from_P(None, cams_all[k].transpose(0, 1)[:3, :4].numpy())  # intrinsics：摄像机的内参矩阵 (K)。pose：摄像机的外参矩阵 (R | t)。
        pose_inv = np.linalg.inv(pose)  # 计算 pose 的逆矩阵 pose_inv，用于 3D 变换。
        intrinsics_modified = intrinsics.copy()

        intrinsics_modified[0, 0] /= 2  # Halve fx
        intrinsics_modified[1, 1] /= 2  # Halve fy
        intrinsics_modified[0, 2] /= 2  # Halve cx
        intrinsics_modified[1, 2] /= 2  # Halve cy

        intrinsics_modified[0, 2] += 0.5  # Adjust cx
        intrinsics_modified[1, 2] += 0.5  # Adjust cy

        scale_y, scale_x = masks[k].shape

        # Create a scaling matrix
        scaling_matrix = np.array([[scale_x, 0, 0, 0],
                                   [0, scale_y, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

        intrinsics_scaled = scaling_matrix @ intrinsics_modified

        size = torch.tensor([scale_y, scale_x]).to(args.device)

        raster_settings_mesh = RasterizationSettings(
            image_size=(scale_y, scale_x), 
            blur_radius=0.000, 
            faces_per_pixel=1)

        cams[k] = cameras_from_opencv_projection(
            camera_matrix=torch.from_numpy(intrinsics_scaled).float()[None].cuda(), 
            R=torch.from_numpy(pose_inv[:3, :3]).float()[None].cuda(),
            tvec=torch.from_numpy(pose_inv[:3, 3]).float()[None].cuda(),
            image_size=size[None].cuda()
        ).cuda()

    # init camera
    R = torch.ones(1, 3, 3)
    t = torch.ones(1, 3)
    cam_intr = torch.ones(1, 4, 4)
    size = torch.tensor([scale_y, scale_x]).to(args.device)

    cam = cameras_from_opencv_projection(
        camera_matrix=cam_intr.cuda(), 
        R=R.cuda(),
        tvec=t.cuda(),
        image_size=size[None].cuda()
    ).cuda()

    # init mesh rasterization
    meshRasterizer = MeshRasterizer(cam, raster_settings_mesh)

    mesh_head.textures = TexturesVertex(verts_features=torch.ones_like(mesh_head.verts_packed()).float().cuda()[None])

    # join hair and bust mesh to handle occlusions
    full_mesh = mesh_head
    
    """
    计算可见性 Mask"
    """
    vis_vertex_mask = check_visiblity_of_faces(cams, masks, meshRasterizer, full_mesh, args.flame_mesh_dir, prob_thr=0.5, n_views_thr=0.1).cuda()  # 通过 check_visiblity_of_faces() 计算头皮网格的可见性：cams：摄像机参数。masks：头部遮挡掩码。meshRasterizer：网格光栅化器（将 3D 网格投影到 2D）。prob_thr=0.5：可见性概率阈值。n_views_thr=0.1：最少可见视角。

    # sorted_idx = torch.where(vis_mask | vis_vertex_mask.bool()[scalp_vert_idx])[0]
    # sorted_idx = vis_mask
    vis_vertex_mask_scalp = vis_vertex_mask.bool()[scalp_vert_idx]
    for i, j in zip([[327, 304, 286, 264, 247, 235], 
                     [236, 251, 271, 294, 309, 329], 
                     [336, 315, 298, 277, 253, 237], 
                     [238, 255, 284, 301, 324, 343],
                     [354, 330, 305, 285, 258, 239]], 
                    [[ 94, 114, 140, 156, 184, 201], 
                     [197, 179, 155, 138, 112,  92],
                     [ 87, 111, 136, 154, 171, 194],
                     [191, 165, 152, 125, 108,  84],
                     [ 79,  99, 118, 144, 159, 189]]):
        tmp = min(vis_vertex_mask_scalp[i].amin(), vis_vertex_mask_scalp[j].amin())
        vis_vertex_mask_scalp[i] = tmp
        vis_vertex_mask_scalp[j] = tmp
    
    for i, j in zip([414, 419, 425, 426, 422, 424, 421,
                     412, 417, 428, 433, 434, 429, 420, 410, 402,
                     403, 409, 415, 432, 437, 435, 423, 411, 398, 393, 387],
                    [ 17,  15,  12,  10,  13,   8,   5,
                      19,  16,   9,   3,   4,  11,  18,  23,  31,
                      27,  24,  20,   7,   0,   1,  22,  28,  36,  43,  47]):
        tmp = min(vis_vertex_mask_scalp[i], vis_vertex_mask_scalp[j])
        vis_vertex_mask_scalp[i] = tmp
        vis_vertex_mask_scalp[j] = tmp

    """
    头皮裁剪
    """
    # vis_vertex_mask_scalp = torch.ones_like(vis_vertex_mask_scalp).bool()
    sorted_idx = torch.where(vis_vertex_mask_scalp)[0]  # #只保留可见的头皮顶点索引。

    # Cut new scalp
    a = np.array(sorted(sorted_idx.cpu()))
    b = np.arange(a.shape[0])
    d = dict(zip(a,b))

    full_scalp_list = sorted(sorted_idx)

    save_path = os.path.join(args.flame_mesh_dir, 'scalp_data')
    os.makedirs(save_path, exist_ok=True)
    
    """
    筛选可见面片：只保留 顶点全部属于裁剪后顶点集 (full_scalp_list) 的面片。"
    """
    faces_masked = []
    for face in scalp_mesh.faces_packed():
        if face[0] in full_scalp_list and face[1] in full_scalp_list and  face[2] in full_scalp_list:
            faces_masked.append(torch.tensor([d[int(face[0])], d[int(face[1])], d[int(face[2])]]))
    """
    将 裁剪后的头皮网格 保存为 .obj 文件。
    """
    save_obj(os.path.join(save_path, 'scalp.obj'), scalp_mesh.verts_packed()[full_scalp_list], torch.stack(faces_masked))

    with open(os.path.join(save_path, 'cut_scalp_verts.pickle'), 'wb') as f:
        pkl.dump(list(torch.tensor(sorted_idx).detach().cpu().numpy()), f)
    
    """
    生成头皮掩码
    """
    # Create scalp mask for diffusion
    scalp_uvs = scalp_uvs[:, full_scalp_list]    
    scalp_mesh = load_objs_as_meshes([os.path.join(save_path, 'scalp.obj')], device=args.device)
    
    scalp_mask = create_scalp_mask(scalp_mesh, scalp_uvs)  # create_scalp_mask() 生成 头皮掩码 (scalp_mask)。    # 画图
    cv2.imwrite(os.path.join(save_path, 'dif_mask.png'), scalp_mask)  # cv2.imwrite() 将掩码图像保存为 dif_mask.png。

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument('--project_dir', default="", type=str)
    parser.add_argument('--data_dir', default="", type=str)
    parser.add_argument('--flame_mesh_dir', default="", type=str)
    parser.add_argument('--cams_path', default="", type=str)
    parser.add_argument('--device', default='cuda', type=str)
    args = get_combined_args(parser)

    main(args)