"""Parking space mapping from an empty-lot zenithal image."""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .feature_extraction import apply_extraction, row_clustering
from .geometry import order_clockwise
from .preprocessing import apply_detection, apply_filter, apply_threshold

DEFAULT_CONFIG = {
    "filter": {"method": "combined", "methods": ["median", "morphological"]},
    "threshold": {"method": "otsu"},
    "edges": {"method": "canny", "low_threshold": 50, "high_threshold": 150},
    "vertices": {"method": "contours", "epsilon_factor": 0.005, "min_area": 500},
    "polygons": {"simplify_distance": 30, "y_threshold": 80, "min_area": 1500},
}


def _run(step, image, cfg):
    params = {k: v for k, v in cfg.items() if k != "method"}
    return step(image, cfg["method"], **params)


class ParkingSpaceDetector:
    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def detect(self, image: np.ndarray, roi: Optional[np.ndarray] = None) -> Dict[str, Any]:
        timings = {}

        t = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if roi is not None:
            gray = cv2.bitwise_and(gray, gray, mask=roi)
        timings["gray"] = time.perf_counter() - t

        t = time.perf_counter()
        filtered, _ = _run(apply_filter, gray, self.config["filter"])
        timings["filter"] = time.perf_counter() - t

        t = time.perf_counter()
        binary, _ = _run(apply_threshold, filtered, self.config["threshold"])
        timings["threshold"] = time.perf_counter() - t

        t = time.perf_counter()
        edges, _ = _run(apply_detection, binary, self.config["edges"])
        timings["edges"] = time.perf_counter() - t

        t = time.perf_counter()
        contours, _ = _run(apply_extraction, edges, self.config["vertices"])
        timings["contours"] = time.perf_counter() - t

        t = time.perf_counter()
        rectangles, _ = row_clustering(contours, **self.config["polygons"])
        timings["polygons"] = time.perf_counter() - t

        spaces = [self._to_space(i, r) for i, r in enumerate(rectangles)]
        timings = {k: round(v * 1000, 2) for k, v in timings.items()}
        timings["total"] = round(sum(timings.values()), 2)

        return {"parking_spaces": spaces, "timings_ms": timings}

    @staticmethod
    def _to_space(i: int, rect: np.ndarray) -> Dict[str, Any]:
        pts = order_clockwise(rect.reshape(-1, 2).astype(float))
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        return {
            "id": f"space_{i + 1:03d}",
            "vertices": pts.astype(int).tolist(),
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "area": float(cv2.contourArea(pts.astype(np.float32))),
        }


def detect_parking_spaces(image_path: str, output: Optional[str] = None,
                          config: Optional[dict] = None) -> List[Dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    result = ParkingSpaceDetector(config).detect(image)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2))
    return result["parking_spaces"]
