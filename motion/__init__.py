"""Motion detection module for comparative analysis."""

from .base import BaseMotionDetector
from .mog2_detector import MOG2MotionDetector
from .mog_detector import MOGMotionDetector
from .gmg_detector import GMGMotionDetector
from .weighted_average_detector import WeightedAverageMotionDetector
from .threshold_difference_detector import ThresholdDifferenceMotionDetector
from .metrics import MotionDetectionMetrics
from .evaluator import MotionDetectionEvaluator

__all__ = [
    'BaseMotionDetector',
    'MOG2MotionDetector',
    'MOGMotionDetector',
    'GMGMotionDetector',
    'WeightedAverageMotionDetector',
    'ThresholdDifferenceMotionDetector',
    'MotionDetectionMetrics',
    'MotionDetectionEvaluator'
]
