from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    polygon: np.ndarray  # (N, 2) in image pixels
    score: float
    label: str = "car"

    @property
    def area(self) -> float:
        return float(cv2.contourArea(self.polygon.astype(np.float32)))


def mask_to_polygon(mask: np.ndarray, epsilon: float = 0.002) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.int32)
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
    return contour.reshape(-1, 2)
