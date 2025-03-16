import glob
import numpy as np
import open3d
import torch
import torch.utils.data as data
from os.path import join
import os
# import MinkowskiEngine as ME
import random
from plyfile import PlyData
device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_point_data(filedir):
    p = np.asarray(open3d.io.read_point_cloud(filedir).points)
    return p


def read_ply_bin(path):
    # 读取 PLY 文件
    plydata = PlyData.read(path)
    
    # 获取顶点数据
    vertex = plydata['vertex']
    
    # 提取 'x', 'y', 'z' 列，组成 coord 数组
    coord = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
    
    # 提取除 'x', 'y', 'z' 以外的其他列，组成 feat 数组
    other_cols = [] 
    for col in vertex.data.dtype.names:
        if col not in ['x', 'y', 'z', 'nx', 'ny', 'nz']:
            other_cols.append(col)        
    
    feat = np.column_stack([vertex[col] for col in other_cols])
    
    return coord, feat


def read_ply_ascii(filedir, dtype="int32"):
    files = open(filedir, 'r')
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data

    return coords

def write_ply_data(filename, coords, feature=None, dtypes=None, names=None):
    print(f'writing {filename}')
    if os.path.exists(filename):
        os.remove(filename)
    if dtypes is None:
        cols = coords.shape[1]
        dtypes = ['float'] * cols
    if names is None:
        names = ['x', 'y', 'z']
    f_flag = feature is not None
    
    with open(filename, 'w', newline='\n') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {coords.shape[0]}\n')

        for i in range(len(dtypes)):
            f.write(f'property {dtypes[i]} {names[i]}\n')

        f.write('end_header\n')

        for point_i in range(coords.shape[0]):
            point = coords[point_i]
            feat = feature[point_i]
            if f_flag:
                f.write(' '.join([str(float(p)) for p in point]) +' '+' '.join([str(int(f)) for f in feat])+'\n')
            else:
                f.write(' '.join([str(float(p)) for p in point]))
    return


def get_point_sparse(filedir, device=device0):
    p = np.asarray(open3d.io.read_point_cloud(filedir).points)
    pc = torch.tensor(p[:, :3]).to(device)
    if p.shape[1] == 3:
        point = torch.ones((pc.shape[0], 1), dtype=torch.float32).to(device)
    else:
        point = torch.tensor(p[:, 3:]).to(device)
    
    frame_C, frame_F = ME.utils.sparse_collate([pc], [point])
    frame_C, frame_F = frame_C.to(device), frame_F.to(device)
    frame_data = ME.SparseTensor(features=frame_F, coordinates=frame_C,
                                     tensor_stride=1, device=device)
    return frame_data
    

class Dataset(data.Dataset):
    def __init__(self, root_dir,format='ply', scaling_factor=1):
        self.lookup = sorted(glob.glob(os.path.join(root_dir, '**', f'*.ply'), recursive=True))
        self.format = format
        self.scaling_factor = scaling_factor

    def __getitem__(self, item):
        file_dir = self.lookup[item]
        print('*****************file dir: ',file_dir,'*'*10)
        if self.format == 'npy':
            p = np.load(file_dir)
        elif self.format == 'ply':
            p = np.asarray(open3d.io.read_point_cloud(file_dir).points)
            # p1 = np.asarray(open3d.io.read_point_cloud(file_dir1).points)
        pc = torch.tensor(p[:, :3]).cuda()
        # pc1 = torch.tensor(p1[:, :3]).cuda()

        if self.scaling_factor != 1:
            pc = torch.unique(torch.floor(pc / self.scaling_factor), dim=0)
            # pc1 = torch.unique(torch.floor(pc1 / self.scaling_factor), dim=0)
        xyz, point = pc, torch.ones((pc.shape[0], 1), dtype=torch.float32).cuda()
        # xyz1, point1 = pc1, torch.ones_like(pc1[:, :1])

        return xyz, point

    def __len__(self):
        return len(self.lookup)
    
class Data_applier:
    def __init__(self, dataset, device=device0):
        self.dataset = dataset
        self.device = device
    
    def get_data(self, idx):
        device = self.device
        # 原始数据
        frame_C, frame_F = self.dataset[idx]

        frame_C, frame_F = ME.utils.sparse_collate([frame_C], [frame_F])
        frame_C, frame_F = frame_C.to(device), frame_F.to(device)
        frame_data = ME.SparseTensor(features=frame_F, coordinates=frame_C,
                                     tensor_stride=1, device=device)
        return frame_data
    
