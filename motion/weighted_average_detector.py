"""Motion detector using weighted average with cv2.accumulateWeighted."""

import cv2
import numpy as np
from typing import Tuple
from .base import BaseMotionDetector


class WeightedAverageMotionDetector(BaseMotionDetector):
    """Motion detector using weighted average background subtraction."""

    def __init__(self, **kwargs):
        params = {
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'threshold': kwargs.get('threshold', 10),
            'min_area': kwargs.get('min_area', 100),
            'blur_kernel_size': kwargs.get('blur_kernel_size', 5),
            'dilate_iterations': kwargs.get('dilate_iterations', 3)
        }

        super().__init__('WeightedAverage', **params)
        self.avg_image = None

    def initialize(self, first_frame: np.ndarray) -> None:
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        blur_size = self.parameters['blur_kernel_size']
        blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        self.avg_image = blur.copy().astype("float")

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, int, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_size = self.parameters['blur_kernel_size']
        blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        cv2.accumulateWeighted(blur, self.avg_image, self.parameters['learning_rate'])
        diff = cv2.absdiff(blur, cv2.convertScaleAbs(self.avg_image))

        threshold_val = self.parameters['threshold']
        thresholded = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)[1]

        dilate_iterations = self.parameters['dilate_iterations']
        thresholded = cv2.dilate(thresholded, None, iterations=dilate_iterations)

        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = self.parameters['min_area']
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        has_motion = len(valid_contours) > 0
        n_contours = len(valid_contours)

        return has_motion, n_contours, thresholded
