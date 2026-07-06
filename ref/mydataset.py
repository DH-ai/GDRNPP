# encoding: utf-8
"""This file includes necessary params, info."""
import mmcv

import os.path as osp
import numpy as np


K = np.array([[2481.9412514178307, 0.0, 978.95936559694314],
        [0.0, 2482.3917472975795, 629.72289542481894],
            [0.0, 0.0, 1.0]],dtype=np.float64)  



cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath("/mnt/data/work/synthetic-data-yolo-training_and_pose_estimation/src")

data_root = osp.join(root_dir, "output")
bop_root = osp.join(data_root, "bop")

# dataset_root = osp.join(bop_root, "mydataset")

train_dir = osp.join(bop_root, "train_pbr")
model_dir = osp.join(bop_root, "models")


id2obj = {
    1: "heart_shape",
    2: "semi_circle",
    3: "triangle_shape",
}
objects = list(id2obj.values())
obj2id = {v: k for k, v in id2obj.items()}
obj_num = len(id2obj)



camera_matrix = K
width = 1920
height = 1200


def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info