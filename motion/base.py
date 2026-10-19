"""Base class for motion detectors."""

import cv2
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any


class BaseMotionDetector(ABC):
    """Abstract base class for motion detectors."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = kwargs
        self.is_initialized = False
        self.frame_count = 0
        self.total_processing_time = 0.0

        self.metrics = {
            'processing_times': [],
            'memory_usage': [],
            'detections': [],
            'frame_count': 0
        }

    @abstractmethod
    def initialize(self, first_frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, int, np.ndarray]:
        pass

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        blur_size = self.parameters.get('blur_kernel_size', 5)
        if blur_size > 1:
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        return gray

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, int, np.ndarray, float]:
        start_time = time.perf_counter()

        if not self.is_initialized:
            self.initialize(frame)
            self.is_initialized = True
            processing_time = time.perf_counter() - start_time
            return False, 0, np.zeros(frame.shape[:2], dtype=np.uint8), processing_time

        has_motion, n_detections, mask = self.detect_motion(frame)

        processing_time = time.perf_counter() - start_time

        self.frame_count += 1
        self.total_processing_time += processing_time
        self.metrics['processing_times'].append(processing_time)
        self.metrics['detections'].append(n_detections)
        self.metrics['frame_count'] += 1

        return has_motion, n_detections, mask, processing_time

    def get_average_processing_time(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return self.total_processing_time / self.frame_count

    def get_fps(self) -> float:
        avg_time = self.get_average_processing_time()
        if avg_time == 0:
            return 0.0
        return 1.0 / avg_time

    def get_metrics_summary(self) -> Dict[str, Any]:
        processing_times = self.metrics['processing_times']

        return {
            'detector_name': self.name,
            'parameters': self.parameters,
            'total_frames': self.frame_count,
            'avg_processing_time': self.get_average_processing_time(),
            'fps': self.get_fps(),
            'min_processing_time': min(processing_times) if processing_times else 0,
            'max_processing_time': max(processing_times) if processing_times else 0,
            'total_detections': sum(self.metrics['detections']),
            'avg_detections_per_frame': np.mean(self.metrics['detections']) if self.metrics['detections'] else 0
        }

    def reset_metrics(self) -> None:
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.metrics = {
            'processing_times': [],
            'memory_usage': [],
            'detections': [],
            'frame_count': 0
        }
        self.is_initialized = False
