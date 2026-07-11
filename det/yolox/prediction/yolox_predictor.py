import cv2
import torch
import numpy as np

from detectron2.config import LazyConfig
from detectron2.engine.defaults import instantiate

from core.utils.my_checkpoint import MyCheckpointer

from det.yolox.utils import postprocess
from det.yolox.data.data_augment import preproc


class YOLOXDetector:

    def __init__(
        self,
        config_file,
        checkpoint,
        device="cuda",
        conf_thres=None,
        nms_thres=None,
    ):

        self.cfg = LazyConfig.load(config_file)

        if conf_thres is not None:
            self.cfg.test.conf_thr = conf_thres

        if nms_thres is not None:
            self.cfg.test.nms_thr = nms_thres

        self.device = device

        self.model = instantiate(self.cfg.model)
        self.model.to(device)
        self.model.eval()

        MyCheckpointer(self.model).load(checkpoint)

    @torch.no_grad()
    def predict(self, image):

        if isinstance(image, str):
            image = cv2.imread(image)

        orig = image.copy()
        h, w = image.shape[:2]

        img, ratio = preproc(
            image,
            self.cfg.test.test_size,
        )

        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        outputs = self.model(img)

        dets = postprocess(
            outputs["det_preds"],
            self.cfg.test.num_classes,
            self.cfg.test.conf_thr,
            self.cfg.test.nms_thr,
        )

        dets = dets[0]

        if dets is None:
            return []

        dets = dets.cpu().numpy()

        results = []

        for det in dets:

            x1, y1, x2, y2 = det[:4]

            x1 /= ratio
            y1 /= ratio
            x2 /= ratio
            y2 /= ratio

            results.append(
                {
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    "score": float(det[4] * det[5]),
                    "class_id": int(det[6]),
                }
            )

        return results