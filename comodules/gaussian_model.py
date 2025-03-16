import numpy as np
import torch
import os
from torch import nn
from plyfile import PlyData, PlyElement
from utils.system_utils import mkdir_p
from sklearn.neighbors import KDTree
from comodules.kd_tree import split_kd_tree, farthest_point_sampling

class GaussianModel_c:
    def __init__(self):
        self.max_sh_degree = 3  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._significance = torch.empty(0)
        # self.max_radii2D = torch.empty(0)
        # self.xyz_gradient_accum = torch.empty(0)
        # self.denom = torch.empty(0)
        # self.optimizer = None
        # self.percent_dense = 0
        # self.spatial_lr_scale = 0

    def dequant(self, min_xyz, max_xyz, qp):
        precision = 2**qp

        xyz = self._xyz.detach().cpu().numpy()
        xyz_dequant = np.zeros_like(xyz, dtype=np.float64)

        for i, column in enumerate(xyz.T):
            xyz_dequant[:, i] = column / (precision - 1) * (max_xyz[i] - min_xyz[i]) + min_xyz[i]

        self._xyz = nn.Parameter(torch.tensor(xyz_dequant, dtype=torch.float64, device="cuda").requires_grad_(True))
                
    def load_gaussian(self,path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
                
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0],len(extra_f_names)))
        for idx,attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x:int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0],len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x:int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0],len(rot_names)))
        for idx,attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def construct_list_of_attributes(self):
        # l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_gaussian(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def save_xyz_to_ply(self, qp : int, ply_path):
        precision = 2**qp
        xyz_numpy = self._xyz.detach().cpu().numpy()
        print(xyz_numpy.shape)
        xyz = np.zeros_like(xyz_numpy, dtype=np.int32)
        # xyz_quant = xyz_quant.cpu().numpy()
        min_xyz = []
        max_xyz = []

        for i, column in enumerate(xyz_numpy.T):
            min_xyz.append(np.min(column))
            max_xyz.append(np.max(column))


            xyz_quant = np.floor((column - np.min(column)) / (np.max(column) - np.min(column)) * (precision - 1) + 0.5)
            xyz_quant = xyz_quant.astype(int)
            xyz[:, i] = xyz_quant
        
        dtype_full = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, xyz))
        el = PlyElement.describe(elements, 'vertex') 
        PlyData([el]).write(ply_path)
        return min_xyz, max_xyz

    def remove_below_threshold(self, threshold):
        # 获取需要保留的索引
        keep_indices = (self._significance >= threshold).nonzero(as_tuple=True)[0]

        # 根据保留的索引更新各个张量
        self._xyz = self._xyz[keep_indices]
        self._features_dc = self._features_dc[keep_indices]
        self._features_rest = self._features_rest[keep_indices]
        self._scaling = self._scaling[keep_indices]
        self._rotation = self._rotation[keep_indices]
        self._opacity = self._opacity[keep_indices]
        self._significance = self._significance[keep_indices]

    def add_significance(self, significances):
        self._significance = torch.tensor(significances, dtype=torch.float, device="cuda")
    
    def cal_significance(self):
        volumes = torch.prod(self._scaling, dim=1)
        volumes = torch.abs(volumes)  # 取绝对值

        # 计算体积的90%分位数
        V_max90 = torch.quantile(volumes, 0.8)

        # 归一化体积
        V_norm = torch.min(torch.max(volumes / V_max90, torch.tensor(0.0)), torch.tensor(1.0))

        # 计算体积的0.01次方
        volume_powers = V_norm ** 0.01

        # 更新显著性
        self._significance *= volume_powers

        return self._significance
    
    def SH_rgbtoyuv(self):
        # RGB to YUV conversion matrix
        rgb_to_yuv_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ], dtype=self._features_dc.dtype, device=self._features_dc.device)

        self._features_dc = self._features_dc.view(-1, 3)

        yuv_dc = torch.matmul(self._features_dc, rgb_to_yuv_matrix.T).view(-1, 1, 3)

        yuv_rest = torch.tensordot(self._features_rest, rgb_to_yuv_matrix.T, dims=([2], [0]))

        self._features_dc = yuv_dc
        self._features_rest = yuv_rest
        

    def SH_yuvtorgb(self):
        # YUV to RGB conversion matrix
        yuv_to_rgb_matrix = torch.tensor([
            [1, 0, 1.13983],
            [1, -0.39465, -0.58060],
            [1, 2.03211, 0]
        ], dtype=self._features_dc.dtype, device=self._features_dc.device)

        self._features_dc = self._features_dc.view(-1, 3)

        rgb_dc = torch.matmul(self._features_dc, yuv_to_rgb_matrix.T).view(-1, 1, 3)

        rgb_rest = torch.tensordot(self._features_rest, yuv_to_rgb_matrix.T, dims=([2], [0]))

        self._features_dc = rgb_dc
        self._features_rest = rgb_rest

    def anchor_sampling(self, max_points_per_block = 256, path: str = None):
        points = self._xyz.detach().cpu().numpy()
        leaf_nodes = split_kd_tree(points, max_points_per_block)
        sampled_points_indices = []

        for indices in leaf_nodes:
            # FPS采样
            num_samples = max(1, int(0.1 * len(indices)))  # 采样5%的点
            sampled_indices = farthest_point_sampling(points[indices], num_samples)
            sampled_points_indices.extend(indices[sampled_indices])
        
        #提取anchor信息
        print(f"sampled_points_indices: {len(sampled_points_indices)}")
        print(f"原始点数: {points.shape[0]}")
        anchor_points = points[sampled_points_indices]
        anchor_features_dc = self._features_dc[sampled_points_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        anchor_features_rest = self._features_rest[sampled_points_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        anchor_scaling = self._scaling[sampled_points_indices].detach().cpu().numpy()
        anchor_rotation = self._rotation[sampled_points_indices].detach().cpu().numpy()
        anchor_opacities = self._opacity[sampled_points_indices].detach().cpu().numpy()

        #保存anchor为ply
        mkdir_p(os.path.dirname(path))
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor_points.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((anchor_points, anchor_features_dc, anchor_features_rest, anchor_opacities, anchor_scaling, anchor_rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # 从self中移除这些点和属性
        mask = np.ones(points.shape[0], dtype=bool)
        mask[sampled_points_indices] = False
        
        self._xyz = self._xyz[mask].to(self._xyz.device)
        self._features_dc = self._features_dc[mask].to(self._features_dc.device)
        self._features_rest = self._features_rest[mask].to(self._features_rest.device)
        self._scaling = self._scaling[mask].to(self._scaling.device)
        self._rotation = self._rotation[mask].to(self._rotation.device)
        self._opacity = self._opacity[mask].to(self._opacity.device)

        print(f"移除anchor后点数: {self._xyz.shape[0]}")

    def LOD_sampling_cs(self, max_points_per_block=256):
        points = self._xyz.detach().cpu().numpy()
        leaf_nodes = split_kd_tree(points, max_points_per_block)
        sampled_points_indices = []

        for indices in leaf_nodes:
            num_samples = max(1, int(0.5 * len(indices)))
            sampled_indices = farthest_point_sampling(points[indices], num_samples)
            sampled_points_indices.extend(indices[sampled_indices])

        # 提取 anchor 数据
        anchor_points = points[sampled_points_indices]
        anchor_features_dc = self._features_dc[sampled_points_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        anchor_features_rest = self._features_rest[sampled_points_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        anchor_scaling = self._scaling[sampled_points_indices].detach().cpu().numpy()
        anchor_rotation = self._rotation[sampled_points_indices].detach().cpu().numpy()
        anchor_opacities = self._opacity[sampled_points_indices].detach().cpu().numpy()

        attributes = np.concatenate((anchor_features_dc, anchor_features_rest, anchor_opacities, anchor_scaling, anchor_rotation), axis=1)

        # 在这里打印检查原始和操作后的数据
        print("Before assignment:")
        print("Original features_dc:", self._opacity[sampled_points_indices][:20])
        print("Anchor features_dc:", anchor_opacities[:20])

        # 确保原数据未改变
        self._xyz[sampled_points_indices] = torch.tensor(anchor_points, device=self._xyz.device)
        self._features_dc[sampled_points_indices] = torch.tensor(anchor_features_dc, device=self._xyz.device).reshape(-1, 3, 1).transpose(1, 2)
        self._features_rest[sampled_points_indices] = torch.tensor(anchor_features_rest, device=self._features_dc.device).reshape(-1, 3, 15).transpose(1, 2)
        self._scaling[sampled_points_indices] = torch.tensor(anchor_scaling, device=self._xyz.device)
        self._rotation[sampled_points_indices] = torch.tensor(anchor_rotation, device=self._xyz.device)
        self._opacity[sampled_points_indices] = torch.tensor(anchor_opacities, device=self._xyz.device)

        # 再次检查
        print("After assignment:")
        print("New features_dc:", self._opacity[sampled_points_indices][:20])

        return anchor_points, attributes

    def sample_points_and_attributes(points, features_dc, features_rest, scaling, rotation, opacity, sampled_indices):
        sampled_points = points[sampled_indices]
        sampled_features_dc = features_dc[sampled_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        sampled_features_rest = features_rest[sampled_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        sampled_scaling = scaling[sampled_indices].detach().contiguous()
        sampled_rotation = rotation[sampled_indices].detach().contiguous()
        sampled_opacity = opacity[sampled_indices].detach().contiguous()

        sampled_attributes = torch.cat([sampled_features_dc, sampled_features_rest, sampled_opacity, sampled_scaling, sampled_rotation], dim=1).detach().cpu().numpy()
        return sampled_points, sampled_attributes

    def anchor_lod_sampling(self, max_points_per_block=256, path: str=None):
        points = self._xyz.detach().cpu().numpy()
        leaf_nodes = split_kd_tree(points, max_points_per_block)
        sampled_points_indices_anchor = set()
        sampled_points_indices_lod1 = set()

        # 首先进行Anchor采样
        for indices in leaf_nodes:
            num_samples_anchor = max(1, int(0.1 * len(indices)))  # 采样10%的点作为Anchor
            sampled_indices_anchor = farthest_point_sampling(points[indices], num_samples_anchor)
            sampled_points_indices_anchor.update(indices[idx] for idx in sampled_indices_anchor)

        print(f"Anchor 点数: {len(sampled_points_indices_anchor)}")

        # 接着进行LOD1采样，从未被选中的点中采样
        for indices in leaf_nodes:
            available_indices = list(set(indices) - sampled_points_indices_anchor)
            num_samples_lod1 = max(1, int(0.3 * len(available_indices)))  # 采样30%的点作为LOD1
            sampled_indices_lod1 = farthest_point_sampling(points[available_indices], num_samples_lod1)
            sampled_points_indices_lod1.update(available_indices[idx] for idx in sampled_indices_lod1)

        print(f"LOD1 点数: {len(sampled_points_indices_lod1)}")
        print(f"原始点数: {points.shape[0]}")

        # 提取Anchor信息
        anchor_points, anchor_attributes = self.sample_points_and_attributes(
            points, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity, list(sampled_points_indices_anchor)
        )

        # 提取LOD1信息
        LOD1_points, LOD1_attributes = self.sample_points_and_attributes(
            points, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity, list(sampled_points_indices_lod1)
        )

        # 从self中移除Anchor和LOD1采样的点和属性
        mask = np.ones(points.shape[0], dtype=bool)
        mask[list(sampled_points_indices_anchor)] = False
        mask[list(sampled_points_indices_lod1)] = False

        self._xyz = torch.from_numpy(points[mask]).to(self._xyz.device)
        self._features_dc = self._features_dc[mask].to(self._features_dc.device)
        self._features_rest = self._features_rest[mask].to(self._features_rest.device)
        self._scaling = self._scaling[mask].to(self._scaling.device)
        self._rotation = self._rotation[mask].to(self._rotation.device)
        self._opacity = self._opacity[mask].to(self._opacity.device)

        # 提取LOD2信息
        LOD2_points, LOD2_attributes = self.sample_points_and_attributes(
            points, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity, np.where(mask)[0]
        )

        # 保存Anchor为PLY
        if path:
            mkdir_p(os.path.dirname(path))
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
            elements = np.empty(anchor_points.shape[0], dtype=dtype_full)
            attributes = np.concatenate((anchor_points, anchor_attributes), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)

        print(f"LOD分配点数: LOD1={LOD1_points.shape[0]}, LOD2={LOD2_points.shape[0]}")
        print(f"移除采样点后的剩余点数: {self._xyz.shape[0]}")
        
        return LOD1_points, LOD1_attributes, LOD2_points, LOD2_attributes

    
    def LOD_sampling(self, max_points_per_block = 256):
        points = self._xyz.detach().cpu().numpy()
        leaf_nodes = split_kd_tree(points, max_points_per_block)
        sampled_points_indices = []

        for indices in leaf_nodes:
            # FPS采样
            num_samples = max(1, int(0.33 * len(indices)))  # 采样5%的点
            sampled_indices = farthest_point_sampling(points[indices], num_samples)
            sampled_points_indices.extend(indices[sampled_indices])
        
        #提取anchor信息
        print(f"sampled_points_indices: {len(sampled_points_indices)}")
        print(f"LOD点数: {points.shape[0]}")
        LOD1_points = points[sampled_points_indices]
        #tensor
        LOD1_features_dc = self._features_dc[sampled_points_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        LOD1_features_rest = self._features_rest[sampled_points_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        LOD1_scaling = self._scaling[sampled_points_indices].detach().contiguous()
        LOD1_rotation = self._rotation[sampled_points_indices].detach().contiguous()
        LOD1_opacities = self._opacity[sampled_points_indices].detach().contiguous()

        LOD1_attributes = torch.cat([LOD1_features_dc, LOD1_features_rest, LOD1_opacities, LOD1_scaling, LOD1_rotation], dim=1).detach().cpu().numpy()

        # 从self中移除这些点和属性
        mask = np.ones(points.shape[0], dtype=bool)
        mask[sampled_points_indices] = False

        self._xyz = torch.from_numpy(points[mask]).to(self._xyz.device)
        self._features_dc = self._features_dc[mask].to(self._features_dc.device)
        self._features_rest = self._features_rest[mask].to(self._features_rest.device)
        self._scaling = self._scaling[mask].to(self._scaling.device)
        self._rotation = self._rotation[mask].to(self._rotation.device)
        self._opacity = self._opacity[mask].to(self._opacity.device)

        LOD2_points = points[mask]
        LOD2_features_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        LOD2_features_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        LOD2_scaling = self._scaling.detach().cpu().numpy()
        LOD2_rotation = self._rotation.detach().cpu().numpy()
        LOD2_opacities = self._opacity.detach().cpu().numpy()   
        # LOD2_features_dc = self._features_dc[LOD2_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        # LOD2_features_rest = self._features_rest[LOD2_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        # LOD2_scaling = self._scaling[LOD2_indices].detach().contiguous()
        # LOD2_rotation = self._rotation[LOD2_indices].detach().contiguous()
        # LOD2_opacities = self._opacity[LOD2_indices].detach().contiguous()

        # 在 tensor 域内拼接所有属性
        LOD2_attributes = np.concatenate((LOD2_features_dc, LOD2_features_rest, LOD2_opacities, LOD2_scaling, LOD2_rotation), axis=1)

        print(f"LODfenbie点数:{LOD1_points.shape[0], LOD2_points.shape[0]}")
        return  LOD1_points, LOD1_attributes, LOD2_points, LOD2_attributes
    
    def rec_attributes(self, LOD1_attributes, LOD2_attributes, LOD1_indices, LOD2_indices):
        device = self._xyz.device
        
        # 提取 attributes 的前 3 列 (N, 3) -> 转置 -> reshape -> (N, 1, 3)
        LOD1_features_dc = torch.tensor(LOD1_attributes[:, :3], device=device)
        LOD1_features_dc = LOD1_features_dc.t().reshape(3, -1).t().reshape(-1, 1, 3)
        LOD2_features_dc = torch.tensor(LOD2_attributes[:, :3], device=device)
        LOD2_features_dc = LOD2_features_dc.t().reshape(3, -1).t().reshape(-1, 1, 3)
        
        # 提取 attributes 的第 4 到第 48 列 (N, 45) -> 转置 -> reshape -> (N, 15, 3)
        LOD1_features_rest = torch.tensor(LOD1_attributes[:, 3:48], device=device)
        LOD1_features_rest = LOD1_features_rest.t().reshape(3, 15, -1).transpose(1, 2).reshape(-1, 15, 3)
        LOD2_features_rest = torch.tensor(LOD2_attributes[:, 3:48], device=device)
        LOD2_features_rest = LOD2_features_rest.t().reshape(3, 15, -1).transpose(1, 2).reshape(-1, 15, 3)

        # 其他属性
        LOD1_opacity = torch.tensor(LOD1_attributes[:, 48:49], device=device).reshape(-1, 1)
        LOD1_scaling = torch.tensor(LOD1_attributes[:, 49:52], device=device).reshape(-1, 3)
        LOD1_rotation = torch.tensor(LOD1_attributes[:, 52:56], device=device).reshape(-1, 4)
        LOD2_opacity = torch.tensor(LOD2_attributes[:, 48:49], device=device).reshape(-1, 1)
        LOD2_scaling = torch.tensor(LOD2_attributes[:, 49:52], device=device).reshape(-1, 3)
        LOD2_rotation = torch.tensor(LOD2_attributes[:, 52:56], device=device).reshape(-1, 4)

        self._features_dc[LOD1_indices] = LOD1_features_dc
        self._features_dc[LOD2_indices] = LOD2_features_dc
        self._features_rest[LOD1_indices] = LOD1_features_rest
        self._features_rest[LOD2_indices] = LOD2_features_rest
        self._opacity[LOD1_indices] = LOD1_opacity
        self._opacity[LOD2_indices] = LOD2_opacity
        self._scaling[LOD1_indices] = LOD1_scaling
        self._scaling[LOD2_indices] = LOD2_scaling
        self._rotation[LOD1_indices] = LOD1_rotation
        self._rotation[LOD2_indices] = LOD2_rotation
    
    def compute_residuals(self, closest_anchor_indices, sampled_points_indices):
        # 提取所有属性
        features_dc = self._features_dc.detach().cpu().numpy()
        features_rest = self._features_rest.detach().cpu().numpy()
        scaling = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()

        attributes = {
            'features_dc': self._features_dc.detach().cpu().numpy(),
            'features_rest': self._features_rest.detach().cpu().numpy(),
            'scaling': self._scaling.detach().cpu().numpy(),
            'rotation': self._rotation.detach().cpu().numpy(),
            'opacity': self._opacity.detach().cpu().numpy()
        }
        file_path_prefix = 'attributes'
        for key, value in attributes.items():
            file_path = f"{file_path_prefix}_{key}.txt"
            value_reshaped = value.reshape(value.shape[0], -1)
            np.savetxt(file_path, value_reshaped, fmt='%f')
            print(f"{key} saved to {file_path}")


        # 移除 anchor 点的属性
        features_dc = features_dc[~np.isin(np.arange(len(features_dc)), sampled_points_indices)]
        print(f"feature{features_dc.shape}")
        features_rest = features_rest[~np.isin(np.arange(len(features_rest)), sampled_points_indices)]
        scaling = scaling[~np.isin(np.arange(len(scaling)), sampled_points_indices)]
        rotation = rotation[~np.isin(np.arange(len(rotation)), sampled_points_indices)]
        opacity = opacity[~np.isin(np.arange(len(opacity)), sampled_points_indices)]

        shapes = {
            'features_dc': features_dc.shape,
            'features_rest': features_rest.shape,
            'scaling': scaling.shape,
            'rotation': rotation.shape,
            'opacity': opacity.shape
        }
        
        # 初始化残差存储
        residuals = {
            'features_dc': [],
            'features_rest': [],
            'scaling': [],
            'rotation': [],
            'opacity': [],
        }

        for i, anchor_idx in enumerate(closest_anchor_indices):
            anchor_idx = sampled_points_indices[anchor_idx]  # 取出实际的 anchor 点索引
            residuals['features_dc'].append(features_dc[i] - self._features_dc[anchor_idx].detach().cpu().numpy())
            residuals['features_rest'].append(features_rest[i] - self._features_rest[anchor_idx].detach().cpu().numpy())
            residuals['scaling'].append(scaling[i] - self._scaling[anchor_idx].detach().cpu().numpy())
            residuals['rotation'].append(rotation[i] - self._rotation[anchor_idx].detach().cpu().numpy())
            residuals['opacity'].append(opacity[i] - self._opacity[anchor_idx].detach().cpu().numpy())


        # 转换为numpy数组
        for key in residuals:
            residuals[key] = np.array(residuals[key])

        return residuals, shapes

    def recolor(self, path, min_xyz, max_xyz, qp):
        precision = 2**qp

        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(float)
        xyz_dequant = np.zeros_like(xyz, dtype=np.float64)

        for i, column in enumerate(xyz.T):
            xyz_dequant[:, i] = column / (precision - 1) * (max_xyz[i] - min_xyz[i]) + min_xyz[i]

        tree = KDTree(self._xyz.detach().cpu().numpy())
        _, idx = tree.query(xyz_dequant, k=1)
        idx = idx.flatten()
        #recolour
        self._features_dc = self._features_dc[idx, :, :]
        self._features_rest = self._features_rest[idx, :, :]
        self._scaling = self._scaling[idx, :]
        self._rotation = self._rotation[idx, :]
        self._opacity = self._opacity[idx, :]
        self._significance = self._significance[idx]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

    def append_gaussian(self, coords, attributes):
        # 确认 coords 和 attributes 的尺寸是否匹配
        assert coords.shape[0] == attributes.shape[0], "Coords and attributes must have the same number of rows"
        assert coords.shape[1] == 3, "Coords should have 3 columns"
        assert attributes.shape[1] == 56, "Attributes should have 56 columns"
        
        # 将 coords 移动到目标设备
        coords = torch.tensor(coords, device=self._xyz.device)
        
        # 提取 attributes 的前 3 列 (N, 3) -> 转置 -> reshape -> (N, 1, 3)
        features_dc = torch.tensor(attributes[:, :3], device=self._features_dc.device)
        features_dc = features_dc.reshape(-1, 3, 1).transpose(1, 2)
        
        # 提取 attributes 的第 4 到第 48 列 (N, 45) -> 转置 -> reshape -> (N, 15, 3)
        features_rest = torch.tensor(attributes[:, 3:48], device=self._features_rest.device)
        features_rest = features_rest.reshape(-1, 3, 15).transpose(1, 2)

        # 其他属性
        opacity = torch.tensor(attributes[:, 48:49], device=self._opacity.device)
        scaling = torch.tensor(attributes[:, 49:52], device=self._scaling.device)
        rotation = torch.tensor(attributes[:, 52:56], device=self._rotation.device)

        #更新self属性
        self._xyz = coords
        self._features_dc = features_dc
        self._features_rest = features_rest
        self._opacity = opacity
        self._scaling = scaling
        self._rotation = rotation

                
