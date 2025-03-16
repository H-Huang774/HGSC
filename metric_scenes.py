# -*- coding: utf-8 -*-
import os
import subprocess


# 指定目录路径
directory_path_1 = "/home/old/huanghe/GS_repository/HGSC/Results/big_scenes"
directory_path_2 = "/home/old/huanghe/GS_repository/HGSC/Results/small_scenes"
# 获取目录下的所有文件名


dic_path = [directory_path_1]
# 打印文件名
for directory_path in dic_path:
    file_names = os.listdir(directory_path)
    for file_name in file_names:
        model_dir = os.path.join(directory_path, file_name)
        model_path = os.path.join(directory_path, file_name, 'point_cloud.ply')
        json_path = os.path.join(directory_path, file_name, 'cameras.json')

        cmd_metric = (
            f'nohup python compare_gs.py '
            f'--model_path {model_dir} '
            f'--camera_json {json_path}'
        )
        subprocess.run(cmd_metric, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='8'))
