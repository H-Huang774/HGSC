
#
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "5"

import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
import numpy as np
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from comodules.gaussian_model import GaussianModel_c
import matplotlib.pyplot as plt
import comodules.encode as encode
import time
from utils.camera_utils import JSON_to_camera


def render_sets(pipeline : PipelineParams, config, model_path):
    cameras = JSON_to_camera(config)
    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree)
        model_base = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split(".")[0]
        gaussians.load_ply(model_path)
        
        bg_color = [1,1,1] if args.white_background else [0, 0, 0]
        # bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        significances_all = 0
        for idx, view in enumerate(tqdm(cameras, desc="Rendering progress")):
            render_bkg = render(view, gaussians, pipeline, background)
            significance_o = render_bkg["significance_o"]
            significances_all += significance_o     
    return significances_all


def prune_gaussians(significances ,gaussians : GaussianModel_c, percentage : int):
    gaussians.add_significance(significances)
    
    # add volume
    significances =  gaussians.cal_significance()
    significances = significances.detach().cpu().numpy()

    threshold = np.percentile(significances, percentage)
    gaussians.remove_below_threshold(threshold)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--xyz_qp",  default=15, type=int)
    parser.add_argument("--camera_json", type=str, default="/home/old/yangkaifa/projects/GaussianCube/data_construction/output_dc_fitting/100715345ee54d7ae38b52b4ee9d36a3/cameras.json")
    parser.add_argument("--model", type=str, default="/home/old/yangkaifa/projects/GaussianCube/data_construction/output_dc_fitting/100715345ee54d7ae38b52b4ee9d36a3/100715345ee54d7ae38b52b4ee9d36a3.ply")
    parser.add_argument("--resolution", type=int, default="-1")
    parser.add_argument("--max_points_per_block", type=int, default=256)
    parser.add_argument("--percentage", type=int, default=60)
    parser.add_argument("--gpcc_path", type=str, default="/home/old/huanghe/GS_repository/HGSC/comodules/tmc/linux/tmc3")
    parser.add_argument("--lod_bit_depth", type=int, nargs='+', default=[4, 8])
    args = parser.parse_args()
    args.sh_degree=3
    args.data_device='cuda:0'
    # torch.cuda.set_device(0)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    encoding_time_all = 0
    decoding_time_all = 0
    size_all = 0

    significances_start_time = time.time()
    significances_all = render_sets(pipeline.extract(args), args, args.model)
    significances_end_time = time.time()
    gs_path = args.model
    
    gaussians = GaussianModel_c()
    gaussians.load_gaussian(gs_path)
    #prune gaussian
    prune_gaussians(significances_all, gaussians, args.percentage)
    #rgb2yuv
    gaussians.SH_rgbtoyuv()
    
    gs_name = gs_path.split('/')[-2]
    # 几何压缩
    os.makedirs(os.path.join(os.path.curdir, 'output', gs_name), exist_ok=True)
    gs_ply_path = os.path.join(os.path.curdir, 'output', gs_name, 'quant.ply')
    min_xyz, max_xyz = gaussians.save_xyz_to_ply(args.xyz_qp , gs_ply_path)
    gpcc_encode_time, gpcc_size = encode.gpcc_encoder(ply_path=gs_ply_path, tmc_path=args.gpcc_path)
    size_all += gpcc_size
    encoding_time_all += gpcc_encode_time
    gs_xyz_decoded_ply = os.path.splitext(gs_ply_path)[0] + '_rec.ply'
    gpcc_decode_time = encode.gpcc_decoder(ply_path=gs_ply_path, tmc_path=args.gpcc_path)
    decoding_time_all += gpcc_decode_time
    encoding_time_all += gpcc_decode_time
    gaussians.recolor(gs_xyz_decoded_ply, min_xyz, max_xyz, args.xyz_qp)

    #anchor 属性RAHT压缩
    os.makedirs(os.path.join(os.path.curdir, 'output', gs_name), exist_ok=True)
    anchor_ply_path = os.path.join(os.path.curdir, 'output', gs_name, 'anchor.ply')
    anchor_sampling_start_time = time.time()
    gaussians.anchor_sampling(path=anchor_ply_path,max_points_per_block=args.max_points_per_block)
    anchor_sampling_end_time = time.time()
    anchor_recoord, anchor_refeat, ret = encode.anchor_attribute_compressor(anchor_ply_path)
    size_all = ret['enc_feat_size']
    encoding_time_all += (ret['enc_feat_time'] + ret['dec_feat_time'])/56 + anchor_sampling_end_time - anchor_sampling_start_time
    decoding_time_all += ret['dec_feat_time']/56
    #LOD1 属性压缩
    LOD_sampling_start_time = time.time()
    LOD1_points, LOD1_attributes, LOD2_points, LOD2_attributes = gaussians.LOD_sampling()
    LOD_sampling_end_time = time.time()
    LOD1_re_attributes, LOD1_bitstream, LOD1_encoding_time, LOD1_decoding_time = encode.LOD_attribute_compressor_zlib(anchor_recoord, anchor_refeat, LOD1_points, LOD1_attributes, bit_depth = args.lod_bit_depth)
    anchor_recoord = np.concatenate([anchor_recoord, LOD1_points], axis=0)
    anchor_refeat = np.concatenate([anchor_refeat, LOD1_re_attributes], axis=0)
    #LOD2 属性压缩
    LOD2_re_attributes, LOD2_bitstream, LOD2_encoding_time, LOD2_decoding_time = encode.LOD_attribute_compressor_zlib(anchor_recoord, anchor_refeat, LOD2_points, LOD2_attributes, bit_depth = args.lod_bit_depth)
    #重建point与attr
    anchor_recoord = np.concatenate([anchor_recoord, LOD2_points], axis=0)
    anchor_refeat = np.concatenate([anchor_refeat, LOD2_re_attributes], axis=0)

    size_all += (LOD1_bitstream + LOD2_bitstream) * 8
    encoding_time_all += LOD1_encoding_time + LOD2_encoding_time
    decoding_time_all += LOD1_decoding_time + LOD2_decoding_time
    
    out_dir = gs_path.split('/point_cloud.ply')[0]
    output_str = (
    f"size_all: {size_all}\n"
    f"encoding_time_all: {encoding_time_all}\n"
    f"decoding_time_all: {decoding_time_all}\n"
)
    out_path = os.path.join(out_dir, 'output.txt')
    with open(out_path, "w") as file:
        file.write(output_str)
    gaussians.append_gaussian(anchor_recoord, anchor_refeat)

    gaussians.dequant(min_xyz, max_xyz, args.xyz_qp)
    gaussians.SH_yuvtorgb()

    rec_path = os.path.join(out_dir, 'rec.ply')
    gaussians.save_gaussian(rec_path)
