"""Metrics collection and analysis for motion detection algorithms."""

import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Single frame detection result."""
    has_motion: bool
    n_contours: int
    processing_time: float
    cpu_percent: float
    memory_mb: float
    mask: Optional[np.ndarray] = None


class MotionDetectionMetrics:
    """Comprehensive metrics collector for motion detection performance."""

    def __init__(self, detector_name: str):
        self.detector_name = detector_name
        self.results: List[DetectionResult] = []
        self.start_time = None

        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024

    def start_measurement(self) -> None:
        self.start_time = time.perf_counter()

    def record_result(self, has_motion: bool, n_contours: int, mask: Optional[np.ndarray] = None) -> DetectionResult:
        if self.start_time is None:
            raise ValueError("Must call start_measurement() before record_result()")

        processing_time = time.perf_counter() - self.start_time

        cpu_percent = self.process.cpu_percent()
        current_memory = self.process.memory_info().rss / 1024 / 1024

        result = DetectionResult(
            has_motion=has_motion,
            n_contours=n_contours,
            processing_time=processing_time,
            cpu_percent=cpu_percent,
            memory_mb=current_memory - self.initial_memory,
            mask=mask
        )

        self.results.append(result)
        self.start_time = None

        return result

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {'detector_name': self.detector_name, 'no_data': True}

        processing_times = [r.processing_time for r in self.results]
        cpu_usage = [r.cpu_percent for r in self.results]
        memory_usage = [r.memory_mb for r in self.results]
        motion_detections = [r.has_motion for r in self.results]
        contour_counts = [r.n_contours for r in self.results]

        return {
            'detector_name': self.detector_name,
            'total_frames': len(self.results),

            'avg_processing_time': np.mean(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'std_processing_time': np.std(processing_times),
            'avg_fps': 1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0,

            'avg_cpu_percent': np.mean(cpu_usage),
            'max_cpu_percent': np.max(cpu_usage),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),

            'motion_detection_rate': np.mean(motion_detections),
            'avg_contours_per_frame': np.mean(contour_counts),
            'max_contours_per_frame': np.max(contour_counts),

            'processing_times': processing_times,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage
        }

    def calculate_accuracy_metrics(self, ground_truth: List[bool]) -> Dict[str, float]:
        if len(ground_truth) != len(self.results):
            raise ValueError("Ground truth length must match number of results")

        predictions = [r.has_motion for r in self.results]

        tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
        tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
        fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
        fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)

        accuracy = (tp + tn) / len(ground_truth) if len(ground_truth) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
