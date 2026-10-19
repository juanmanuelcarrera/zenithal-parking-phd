"""Motion detector using threshold-based frame difference."""

import cv2
import numpy as np
from typing import Tuple, Optional
from .base import BaseMotionDetector


class ThresholdDifferenceMotionDetector(BaseMotionDetector):
    """Motion detector using simple frame difference with threshold."""

    def __init__(self, **kwargs):
        params = {
            'threshold': kwargs.get('threshold', 8),
            'min_area': kwargs.get('min_area', 100),
            'blur_kernel_size': kwargs.get('blur_kernel_size', 3),
            'dilate_iterations': kwargs.get('dilate_iterations', 4)
        }

        super().__init__('ThresholdDifference', **params)
        self.previous_frame = None
        self.target_size = None

    def initialize(self, first_frame: np.ndarray) -> None:
        self.previous_frame = self.preprocess_frame(first_frame)

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, int, np.ndarray]:
        current_gray = self.preprocess_frame(frame)

        if self.previous_frame is not None:
            diff = cv2.absdiff(self.previous_frame, current_gray)
        else:
            diff = np.zeros_like(current_gray)

        threshold_val = self.parameters['threshold']
        thresholded = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)[1]

        dilate_iterations = self.parameters['dilate_iterations']
        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresholded = cv2.dilate(thresholded, kernel, iterations=dilate_iterations)

        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = self.parameters['min_area']
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        has_motion = len(valid_contours) > 0
        n_contours = len(valid_contours)

        self.previous_frame = current_gray.copy()

        return has_motion, n_contours, thresholded
