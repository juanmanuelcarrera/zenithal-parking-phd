"""Motion detector using BackgroundSubtractorGMG."""

import cv2
import numpy as np
from typing import Tuple
from .base import BaseMotionDetector


class GMGMotionDetector(BaseMotionDetector):
    """Motion detector using BackgroundSubtractorGMG."""

    def __init__(self, **kwargs):
        params = {
            'initializationFrames': kwargs.get('initializationFrames', 120),
            'decision_threshold': kwargs.get('decision_threshold', 0.8),
            'min_area': kwargs.get('min_area', 500),
            'blur_kernel_size': kwargs.get('blur_kernel_size', 5)
        }

        super().__init__('GMG', **params)
        self.bg_subtractor = None

    def initialize(self, first_frame: np.ndarray) -> None:
        if not hasattr(cv2, 'bgsegm'):
            raise ImportError("GMG requires opencv-contrib-python (cv2.bgsegm)")
        self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
            initializationFrames=self.parameters['initializationFrames'],
            decisionThreshold=self.parameters['decision_threshold']
        )

        gray = self.preprocess_frame(first_frame)
        self.bg_subtractor.apply(gray)

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, int, np.ndarray]:
        gray = self.preprocess_frame(frame)

        fg_mask = self.bg_subtractor.apply(gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = self.parameters['min_area']
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        has_motion = len(valid_contours) > 0
        n_contours = len(valid_contours)

        return has_motion, n_contours, fg_mask
