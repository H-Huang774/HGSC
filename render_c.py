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

import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from loadjson import getCameras
from utils.camera_utils import JSON_to_camera

def render_sets_gt(pipeline : PipelineParams, config, model_path):
    cameras = JSON_to_camera(config)
    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree)
        gaussians.load_ply(os.path.join(model_path,"point_cloud.ply"))
        
        bg_color = [1,1,1] if args.white_background else [0, 0, 0]
        # bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_path = os.path.join(model_path, 'gt')
        if not os.path.exists(render_path):
            makedirs(render_path, exist_ok=True)
            for idx, view in enumerate(tqdm(cameras, desc="Rendering progress")):
                # if idx in [0, 25, 28]:
                rendering = render(view, gaussians, pipeline, background, is_rendering=True)["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_path, "{:05d}.png".format(idx)))
            print("Finish gt rendering...\n")
def render_sets_rec(pipeline : PipelineParams, config, model_path):
    cameras = JSON_to_camera(config)
    
    with torch.no_grad():
        filelist = os.listdir(model_path)
        for file in filelist:
            if file.endswith(".ply") and not file.endswith("point_cloud.ply"):
                gaussians = GaussianModel(args.sh_degree)
                gaussians.load_ply(os.path.join(model_path,file))
                
                bg_color = [1,1,1] if args.white_background else [0, 0, 0]
                # bg_color = [1,1,1]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                render_path = os.path.join(model_path, 'rec')
                if not os.path.exists(render_path):
                    makedirs(render_path, exist_ok=True)

                    for idx, view in enumerate(tqdm(cameras, desc="Rendering progress")):
                        # if idx in [0, 25, 28]:
                        rendering = render(view, gaussians, pipeline, background, is_rendering=True)["render"]
                        torchvision.utils.save_image(rendering, os.path.join(render_path, "{:05d}.png".format(idx)))

               

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--compare",action="store_true")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--distortion_model_path",default=None, type=str, help="Path to the raw model folder (if different from the source path)")
    parser.add_argument("--camera_json",default=r"D:\yanjiusheng\project\Gaussian_Splatting\gaussian-splatting\test_selectpoints_5.10\cameras.json", type=str, help="Path to the camera json file")
    parser.add_argument("--good_model_path", type=str, default="/home/old/yangkaifa/projects/GaussianCube/data_construction/output_gaussiancube_debug/volume/100715345ee54d7ae38b52b4ee9d36a3.ply")
    parser.add_argument("--resolution", type=int, default="1")
    args = parser.parse_args()

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.sh_degree=3
    args.data_device='cuda:0'

    # camera_file_path = args.camera_path
     
    render_sets_gt(pipeline.extract(args), args, args.good_model_path)
            
    if args.distortion_model_path is not None:
        render_sets_rec(pipeline.extract(args), args, args.distortion_model_path)
        
        
