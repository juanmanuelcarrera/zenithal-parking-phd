"""Pixel to metre calibration and polygon metrics."""
from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Shoelace (Gauss) formula. Points in order, not closed."""
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def vehicle_dimensions(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Length and width (pixels) of the minimum rotated rectangle."""
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    return max(w, h), min(w, h)


@dataclass
class Calibration:
    """gamma = d_real / d_img, metres per pixel. One per camera."""
    gamma: float

    @classmethod
    def from_reference(cls, p1, p2, real_distance_m: float) -> "Calibration":
        # p1, p2: endpoints of a road marking with known length
        d_img = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        return cls(real_distance_m / d_img)

    def length(self, pixels: float) -> float:
        return self.gamma * pixels

    def area(self, pixels2: float) -> float:
        return self.gamma ** 2 * pixels2

    def polygon_area(self, points) -> float:
        return self.area(polygon_area(points))

    def vehicle_size(self, points) -> Tuple[float, float]:
        length, width = vehicle_dimensions(points)
        return self.length(length), self.length(width)
