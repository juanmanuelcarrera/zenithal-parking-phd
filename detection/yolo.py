"""YOLOv11-seg (Ultralytics) segmenter. Used as the fast 'sentinel' model."""
from typing import List

import numpy as np
from ultralytics import YOLO

from .masks import Detection

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloSegmenter:
    def __init__(self, weights: str = "yolo11n-seg.pt", conf: float = 0.5, device: str = "cpu"):
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.finetuned = not weights.startswith("yolo11")

    def __call__(self, image: np.ndarray) -> List[Detection]:
        result = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
        if result.masks is None:
            return []

        detections = []
        for poly, score, cls in zip(result.masks.xy, result.boxes.conf.numpy(), result.boxes.cls.numpy()):
            cls = int(cls)
            if not self.finetuned and cls not in COCO_VEHICLES:
                continue
            label = "car" if self.finetuned else COCO_VEHICLES[cls]
            detections.append(Detection(poly.astype(np.int32), float(score), label))
        return detections

    def has_vehicles(self, image: np.ndarray) -> bool:
        return len(self(image)) > 0
