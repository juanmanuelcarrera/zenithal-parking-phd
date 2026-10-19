"""Measurement region at the entrance: each vehicle is measured once while inside it."""
from typing import Dict, Optional

import numpy as np
from shapely.geometry import Polygon

from .calibration import Calibration


class MeasurementRegion:
    def __init__(self, polygon, calibration: Calibration, min_overlap: float = 0.95):
        self.region = Polygon(polygon)
        self.calibration = calibration
        self.min_overlap = min_overlap
        self.measured = set()

    def measure(self, track_id: int, vehicle_polygon: np.ndarray) -> Optional[Dict[str, float]]:
        if track_id in self.measured:
            return None

        vehicle = Polygon(vehicle_polygon).buffer(0)
        if vehicle.area == 0 or vehicle.intersection(self.region).area / vehicle.area < self.min_overlap:
            return None  # vehicle not fully inside yet

        length, width = self.calibration.vehicle_size(vehicle_polygon)
        self.measured.add(track_id)
        return {"length_m": length, "width_m": width, "area_m2": self.calibration.polygon_area(vehicle_polygon)}
