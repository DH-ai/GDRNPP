#!/usr/bin/env python3
"""YOLOX prediction entrypoint, structured like ``tools/main_yolox.py``.

Unlike training, prediction currently runs in one process.  Using Detectron2's
``launch`` without input sharding would make every worker process the same files
and overwrite the same outputs.
"""

import os.path as osp
import sys

import cv2
from loguru import logger as loguru_logger


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Make ``core``, ``det``, and ``lib`` imports work when this file is executed
# directly instead of through ``python -m``.
CUR_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(CUR_DIR, "../../../"))

from det.yolox.prediction.cli_predict import build_parser, pred_main  # noqa: E402


@loguru_logger.catch(reraise=True)
def main(args):
    """Run all prediction operations represented by parsed CLI arguments."""
    return pred_main(args)


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    loguru_logger.info("Command line arguments: {}", cli_args)
    main(cli_args)
