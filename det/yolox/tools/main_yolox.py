#!/usr/bin/env python3
import logging
from loguru import logger as loguru_logger
import os.path as osp
from setproctitle import setproctitle
from detectron2.engine import (
    default_argument_parser,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import LazyConfig, instantiate

import cv2

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

import sys

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../"))

from lib.utils.time_utils import get_time_str
import core.utils.my_comm as comm
from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.engine.yolox_setup import default_yolox_setup
from det.yolox.engine.yolox_trainer import YOLOX_DefaultTrainer
from det.yolox.utils import fuse_model
from det.yolox.data.datasets.dataset_factory import register_datasets_in_cfg
import os 
import torch
logger = logging.getLogger("detectron2")


print("MAIN START")
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA AVAILABLE =", torch.cuda.is_available())
print("CUDA COUNT =", torch.cuda.device_count())

# exit()
def setup(args):
    """Create configs and perform basic setups."""
    cfg = LazyConfig.load(args.config_file)

    default_yolox_setup(cfg, args) # setup basic dectron2 logger, logs basic info about config cmd line arguments 
    register_datasets_in_cfg(cfg) # actual datset registraion

    setproctitle("{}.{}".format(cfg.train.exp_name, get_time_str()))
    return cfg


@loguru_logger.catch
def main(args):
    
    cfg = setup(args)
    print("="*50)
    print(cfg)
    # exit()
    Trainer = YOLOX_DefaultTrainer


    print(args.eval_only)
    
    if args.eval_only:  # eval
        model = Trainer.build_model(cfg)
        MyCheckpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.init_checkpoint, resume=args.resume
        )
        if cfg.test.fuse_conv_bn:
            logger.info("\tFusing conv bn...")
            model = fuse_model(model)
        res = Trainer.test(cfg, model)
        # import ipdb; ipdb.set_trace()
        return res
    # train
    trainer = Trainer(cfg)
    print("trainer.model = ", trainer.model)
    print("trainer.optimizer = ", trainer.optimizer)
    print("trainer.")
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("="*50)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
