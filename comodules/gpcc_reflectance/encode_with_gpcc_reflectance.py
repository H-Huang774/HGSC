import re
from comodules.gpcc_reflectance.use_gpcc import gpcc_att_encode, gpcc_att_decode, gpcc_att_encode, gpcc_att_decode


import numpy as np
from comodules.gpcc_reflectance.data_in_out import read_ply_bin, read_ply_ascii, write_ply_data

import os


file_dir = os.path.dirname(os.path.realpath(__file__))
cfg_dir = os.path.join(file_dir,'cfgs/usually_use/lossless-geom-lossless-attrs/ford_01_q1mm')


def get_raval_number(V):
    dim = V.shape[1]
    p_num = V.shape[0]
    max_num = np.ceil(np.max(V) + 1)
    min_num = np.floor(np.min(V))
    
    r_ = max_num - min_num
    V = V - min_num
    
    ret = np.zeros(p_num, dtype=np.int64)
    for i in range(dim):
        ret = ret*r_ + V[:,i]
    return ret


def sort_cloud(V, C=None, indices=False):
    r_number = get_raval_number(V)
    sort_indices = np.argsort(r_number)
    r_V = V[sort_indices]
    if C is not None:
        r_C = C[sort_indices]
        if indices:
            return r_V, r_C, sort_indices
        return r_V,r_C
    if indices:
        return r_V, sort_indices
    return r_V

def float_to_uint16(float_value, min_value, max_value, depth=16):
    
    precise = 1 << depth - 1
    # 归一化到 [0, 1]
    normalized_value = (float_value - min_value) / (max_value - min_value)
    # 转换到 [0, 65535] 并转为 uint16
    uint16_value = np.round(normalized_value * precise).astype(np.int)
    return uint16_value

def uint16_to_float(uint16_value, min_value, max_value, depth=16):
    
    precise = 1 << depth - 1
    # 恢复到 [0, 1]
    normalized_value = uint16_value / precise
    # 反归一化到 [min_value, max_value]
    float_value = normalized_value * (max_value - min_value) + min_value
    return float_value



def gpcc_enc(coord, feat, idx):
    # 将点云数据写入文件
    file_name = f'exp/{idx}.ply'
    bin_name = f'exp/{idx}.bin'
    assert feat.shape[1] == 1
    write_ply_data(file_name, coord, feat,  
                   dtypes=['float', 'float', 'float', 'uint16'], 
                   names=['x', 'y', 'z', 'reflectance'])
    info,_ = gpcc_att_encode(file_name, bin_name,tmc3dir='tmc3_v23',DBG=True, cfgdir=cfg_dir+'/encoder.cfg')
    
    
    pattern = r'positions bitstream size (\d+) B'
    matches = re.search(pattern, info, re.DOTALL)
    positions_size = matches.group(1)
    
    pattern = r'positions processing time \(user\): ([\d.]+) s'
    matches = re.search(pattern, info, re.DOTALL)
    positions_time = matches.group(1)
    
    

    pattern = r'reflectances bitstream size (\d+) B'
    matches = re.search(pattern, info, re.DOTALL)
    reflectances_size = matches.group(1)
    
    pattern = r'reflectances processing time \(user\): ([\d.]+) s'
    matches = re.search(pattern, info, re.DOTALL)
    reflectances_time = matches.group(1)
    
    
    

        
    return {'pos_size':int(positions_size), 'pos_time': float(positions_time),
            'refc_size':int(reflectances_size), 'refc_time': float(reflectances_time)}
    
    
    
def gpcc_dec(idx):
    bin_name = f'exp/{idx}.bin'
    file_name = f'exp/{idx}_dec.ply'
    info, _ = gpcc_att_decode(bin_name, file_name, tmc3dir='tmc3_v23', DBG=True, cfgdir=cfg_dir+'/decoder.cfg')
    data = read_ply_ascii(file_name)
    
    
    pattern = r'positions processing time \(user\): ([\d.]+) s'
    matches = re.search(pattern, info, re.DOTALL)
    positions_time = matches.group(1)
    
    
    pattern = r'reflectances processing time \(user\): ([\d.]+) s'
    matches = re.search(pattern, info, re.DOTALL)
    reflectances_time = matches.group(1)
    
    coord = data[:, :3]
    feat = data[:, 3:]
    return {'coord':coord, 'pos_time': float(positions_time),
            'feat':feat, 'refc_time': float(reflectances_time)}
    



def compress_use_reflectance(file_name=None, coord_depth=12, feat_depth=16):
    
    if not os.path.exists('exp'):
        # 创建exp文件夹
        os.makedirs('exp')
        
    # data = read_ply_ascii(file_name)
    # coord = data[:, :3]
    # feat_ori = data[:, 3:]
    
    coord, feat = read_ply_bin(file_name)
    # print(data)
    # 断言feat_ori中没有nan
    assert np.sum(np.isnan(feat)) == 0
    
    # coord_min = coord.min(axis=0)
    # coord_max = coord.max(axis=0)
    # coord_uint16 = float_to_uint16(coord, coord_min, coord_max, depth=coord_depth)
    
    
    # 按列独立量化feat
    feat_min = feat.min(axis=0)
    feat_max = feat.max(axis=0)
    # print(feat_min, feat_max)
    feat_uint16 = float_to_uint16(feat, feat_min, feat_max, depth=feat_depth)
    coord, feat_uint16,indices = sort_cloud(coord, feat_uint16, indices=True)
    
    # 重新排序, 方便后续比对
    feat = feat[indices,:]
    
    
    enc_positions_size = 0
    enc_positions_time = 0
    enc_feat_size = 0
    enc_feat_time = 0
    
    feat_num = feat.shape[1]
    # feat_num = 1
    
    for col in range(feat_num):
        col_feat = feat_uint16[:, col:col+1]
        enc_info = gpcc_enc(coord, col_feat, col)
        # enc_positions_size = enc_info['pos_size']
        # enc_positions_time = enc_info['pos_time']
        
        enc_feat_size += enc_info['refc_size']
        enc_feat_time += enc_info['refc_time']
        
    print(f'Positions bitstream size: {enc_positions_size}')
    print(f'Positions encoding time: {enc_positions_time}')
    
    print(f'Reflectances bitstream size: {enc_feat_size}')
    print(f'Reflectances encoding time: {enc_feat_time}')
    
    
    
    # 解码
    feat_list = []
    dec_position_time = 0
    dec_feat_time = 0
    for col in range(feat_num):
        dec_info = gpcc_dec(col)
        re_coord = dec_info['coord']
        col_feat = dec_info['feat']
        
        dec_position_time = dec_info['pos_time']
        dec_feat_time += dec_info['refc_time']
        
        re_coord, col_feat = sort_cloud(re_coord, col_feat)
        assert np.sum(re_coord != coord) == 0
        feat_list.append(col_feat)
    re_feat_uint16 = np.concatenate(feat_list, axis=1)
    
    # re_feat = uint16_to_float(re_feat_uint16, feat_min, feat_max)
    # assert np.sum(re_feat_uint16 != feat_uint16[:,:feat_num]) == 0
    err = np.sum(np.abs(feat_uint16[:,:feat_num] - re_feat_uint16))
    assert np.sum(re_coord != coord) == 0
    
    # 反归一化
    
    re_feat = uint16_to_float(re_feat_uint16, feat_min, feat_max)
    
    err2 = np.sum(np.abs(feat - re_feat))
    print(err, err2)
    return re_coord, re_feat, {'enc_positions_size':enc_positions_size, 'enc_positions_time':enc_positions_time,
            'enc_feat_size':enc_feat_size, 'enc_feat_time':enc_feat_time,
            'dec_position_time':dec_position_time, 'dec_feat_time':dec_feat_time,
            }
    
    



if __name__ == '__main__':
    # file_dir = '/home/old/huangwenjie/python_files/RAHT/std_data/longdress_vox10_1052.ply'
    # file_dir = '/home/old/huangwenjie/python_files/RAHT/gs_data/point_cloud.ply'
    file_dir = '/home/old/huanghe/GS_repository/gaussian-splatting/output/dunhuang/anchor.ply'
    re_coord, re_feat, ret = compress_use_reflectance(file_name = file_dir)
    print(ret)