import os
import subprocess

# 指定目录路径
directory_path = "/home/old/huanghe/GS_repository/HGSC/Results/small_scenes"

# 获取目录下的所有文件名
file_names = os.listdir(directory_path)
xyz_qp = 15
max_points_per_block = 256
percentage = 60

# 打印文件名
for file_name in file_names:
    model_dir = os.path.join(directory_path, file_name)
    model_path = os.path.join(directory_path, file_name, 'point_cloud.ply')
    json_path = os.path.join(directory_path, file_name, 'cameras.json')
    
    # 构建并执行第一个命令，挂在后台
    cmd_rec = (
        f'nohup python train.py '
        f'--model {model_path} '
        f'--camera_json {json_path} '
        f'--xyz_qp {xyz_qp} '
        f'--max_points_per_block {max_points_per_block} '
        f'--percentage {percentage}'
    )
    subprocess.run(cmd_rec, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='4'))  # 使用 shell=True 以字符串形式执行命令

