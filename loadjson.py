from argparse import ArgumentParser
import json
import numpy as np
from scene.dataset_readers import CameraInfo
# from scene.cameras import Camera
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal,focal2fov
# from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from arguments import ModelParams, PipelineParams,get_combined_args
from PIL import Image

WARNED = False

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def camera_from_JSON(camera_entry):
    id = camera_entry['id']
    img_name = camera_entry['img_name']
    width = camera_entry['width']
    height = camera_entry['height']
    position = np.array(camera_entry['position'])
    rotation = np.array(camera_entry['rotation'])
    fovy = camera_entry['fy']
    fovx = camera_entry['fx']
    
    
    W2C = np.zeros((4, 4))
    W2C[:3, :3] = rotation
    W2C[:3, 3] = position
    W2C[3, 3] = 1.0

    Rt = np.linalg.inv(W2C)
    R = np.transpose(Rt[:3, :3])
    T = Rt[:3, 3]
    
    # fovy = focal2fov(fov2focal(fovx, width), height)
    # FovY = fovy 
    # FovX = fovx
    
    FovY = focal2fov(fovy, height)
    FovX = focal2fov(fovx, width)
    # 创建一个新的空白图像，大小为980x545
    image = Image.new('RGB', (width, height), color = 'red')
    # image = Image.open(img)

    
    camera = (CameraInfo(uid=id, R=R, T=T, FovY=FovY, FovX=FovX, image = image, image_path = None,
                         image_name=img_name, width=width, height=height))

    
    return camera
    
def load_cameras_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        camera_entries = json.load(file)
        
    # camlist = []
    cameraslist = []
    # for entry in enumerate(camera_entries):
    #     # camera = MiniCam(entry["image_width"], entry["image_height"], entry["FoVy"], entry["FoVx"], entry["znear"], entry["zfar"], entry["world_view_transform"], entry["full_proj_transform"])
    #     camlist.append(camera_from_JSON(entry))
    
    for id, cam in enumerate(camera_entries):
        cameraslist.append(camera_from_JSON(cam))
       
    return cameraslist

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

# Path: gaussian-splatting/output_test/4cbf12a6-2/loadjson.py
def getCameras(args):
    
    # cameraslist = load_cameras_from_json(r"D:\yanjiusheng\project\Gaussian_Splatting\gaussian-splatting\test_4.27\cameras.json")
    # cameraslist = load_cameras_from_json(r"D:\yanjiusheng\project\Gaussian_Splatting\gaussian-splatting\test_selectpoints_5.10\cameras_test.json")
    cameraslist = load_cameras_from_json(args.camera_path)
    resolution_scales=[1.0]

    cameras = {}
    for resolution_scale in resolution_scales:
        print("Loading Cameras json")
        cameras[resolution_scale] = cameraList_from_camInfos(cameraslist, resolution_scale, args)

    return cameras[resolution_scale]

# parser = ArgumentParser(description="render parameters")
# args = ModelParams(parser, sentinel=True)
# pipeline = PipelineParams(parser)
# getCameras(args) 