import os.path as osp
import numpy as np


K = np.array([[2481.9412514178307, 0.0, 978.95936559694314],
        [0.0, 2482.3917472975795, 629.72289542481894],
            [0.0, 0.0, 1.0]],dtype=np.float64)  
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS")

dataset_root = osp.join(bop_root, "mydataset")

train_dir = osp.join(dataset_root, "train_pbr")
model_dir = osp.join(dataset_root, "models")


id2obj = {
    1: "obj1",
    2: "obj2",
    3: "obj3",
}
objects = list(id2obj.values())
obj2id = {v: k for k, v in id2obj.items()}
obj_num = len(id2obj)



camera_matrix = K
width = 1920
height = 1200