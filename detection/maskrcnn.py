"""Mask R-CNN (Detectron2) segmenter. Used as the 'expert' model."""
from typing import List, Optional

import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from .masks import Detection, mask_to_polygon

BASE_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def build_cfg(weights: Optional[str] = None, num_classes: Optional[int] = None,
              score_thresh: float = 0.5, device: str = "cpu"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = weights or model_zoo.get_checkpoint_url(BASE_CONFIG)
    if num_classes:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device
    return cfg


class MaskRCNNSegmenter:
    def __init__(self, weights: Optional[str] = None, score_thresh: float = 0.5, device: str = "cpu"):
        # Fine-tuned weights have a single 'car' class; COCO weights need filtering.
        self.finetuned = weights is not None
        cfg = build_cfg(weights, 1 if self.finetuned else None, score_thresh, device)
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, image: np.ndarray) -> List[Detection]:
        inst = self.predictor(image)["instances"].to("cpu")
        classes = inst.pred_classes.numpy()
        detections = []
        for mask, score, cls in zip(inst.pred_masks.numpy(), inst.scores.numpy(), classes):
            if not self.finetuned and cls not in COCO_VEHICLES:
                continue
            label = "car" if self.finetuned else COCO_VEHICLES[cls]
            detections.append(Detection(mask_to_polygon(mask), float(score), label))
        return detections
