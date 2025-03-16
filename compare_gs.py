import subprocess
from argparse import ArgumentParser

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--model_path", type=str, default='')
parser.add_argument("--camera_json", type=str, default='')
args = parser.parse_args()

# good_model_path = r"/home/old/huanghe/GS_repository/gaussian-splatting/Results/big_scenes/train"
# bad_model_path = r"/home/old/huanghe/GS_repository/gaussian-splatting/Results/big_scenes/train"
# camera_path = r"/home/old/huanghe/GS_repository/gaussian-splatting/Results/big_scenes/train/cameras.json"
# 运行render_c.py
p1 = subprocess.Popen(["python", "render_c.py","--good_model_path",args.model_path,"--distortion_model_path",args.model_path, "--camera_json",args.camera_json])
p1.wait()

# 当render_c.py完成后，运行metric.py
p2 = subprocess.Popen(["python", "metrics_c.py","--path1",args.model_path,"--path2",args.model_path])
p2.wait()