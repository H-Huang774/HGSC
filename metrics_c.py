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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        # render = Image.open(renders_dir / fname)
        render = Image.open(os.path.join(renders_dir, fname))
        # gt = Image.open(gt_dir / fname)
        gt = Image.open(os.path.join(gt_dir, fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    #lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"result_path:SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

def GS_compare(path1, path2):

    full_dict = {}
    per_view_dict = {}
    
    dir_path = os.path.abspath(os.path.dirname(path1))
    print("dir_path:", dir_path)
    result_path = os.path.join(dir_path, "results")
    # os.make_dirs(result_path, exist_ok=True)
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
        full_dict= {}
        per_view_dict = {}   
        renders_dir1 = os.path.join(path1 , "renders")
        # renders_dir2 = os.path.join(path2 , "renders-point_cloud_dec_0.1.ply")
        renders_dir2 = path2
        renders, gts, image_names = readImages(renders_dir1, renders_dir2)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")
        
        full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                                            "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

        with open(result_path + "/results.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(result_path + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
        

def GS_compare_multi(path1, path2,file):
    
    full_dict = {}
    per_view_dict = {}
    
    result_path = os.path.join(path1, "results")
    print("result_path:", result_path)
    # os.make_dirs(result_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    
    full_dict= {}
    per_view_dict = {}   
    renders_dir1 = os.path.join(path1 , "gt")
    # renders_dir2 = os.path.join(path2 , "renders-point_cloud_dec_0.1.ply")
    renders_dir2 = path2
    renders, gts, image_names = readImages(renders_dir1, renders_dir2)

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")
    
    full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
    per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

    with open(result_path + "/results.json", 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(result_path + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)
                               
if __name__ == "__main__":
    # device = torch.device("cuda:0")
    # torch.cuda.set_device(device)
    # good_model_path = r"D:\yanjiusheng\project\Gaussian_Splatting\gaussian-splatting\output_test2\4cbf12a6-2"
    # bad_model_path = r"D:\yanjiusheng\project\Gaussian_Splatting\gaussian-splatting\output_test2\4cbf12a6-2-bad"

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--path1',default=None, type=str)
    parser.add_argument('--path2',default=None, type=str)
    args = parser.parse_args()
    # evaluate(args.model_paths)
    # path1 = 
    
    # GS_compare(args.path1, args.path2)
    
    filelist = os.listdir(args.path2)
    for file in filelist:
        print(file)
        if file.find("rec") != -1 and not file.endswith('.ply'):
            GS_compare_multi(args.path1, os.path.join(args.path2, file),file)
            # pass
        
