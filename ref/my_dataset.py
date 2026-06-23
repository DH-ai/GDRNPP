import os.path as osp
import numpy as np

cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS")

dataset_root = osp.join(bop_root, "mydataset")

train_dir = osp.join(dataset_root, "train_pbr")
model_dir = osp.join(dataset_root, "models")

vertex_scale = 0.001

id2obj = {
    1: "obj1",
    2: "obj2",
    3: "obj3",
}

objects = list(id2obj.values())
obj_num = len(id2obj)
obj2id = {v: k for k, v in id2obj.items()}

width = 1920
height = 1200