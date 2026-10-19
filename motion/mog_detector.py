"""Motion detector using BackgroundSubtractorMOG."""

import cv2
import numpy as np
from typing import Tuple
from .base import BaseMotionDetector


class MOGMotionDetector(BaseMotionDetector):
    """Motion detector using BackgroundSubtractorMOG."""

    def __init__(self, **kwargs):
        params = {
            'history': kwargs.get('history', 200),
            'n_mixtures': kwargs.get('n_mixtures', 5),
            'background_ratio': kwargs.get('background_ratio', 0.7),
            'noise_sigma': kwargs.get('noise_sigma', 0),
            'min_area': kwargs.get('min_area', 500),
            'blur_kernel_size': kwargs.get('blur_kernel_size', 5)
        }

        super().__init__('MOG', **params)
        self.bg_subtractor = None

    def initialize(self, first_frame: np.ndarray) -> None:
        if not hasattr(cv2, 'bgsegm'):
            raise ImportError("MOG requires opencv-contrib-python (cv2.bgsegm)")
        self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(
            history=self.parameters['history'],
            nmixtures=self.parameters['n_mixtures'],
            backgroundRatio=self.parameters['background_ratio'],
            noiseSigma=self.parameters['noise_sigma']
        )

        gray = self.preprocess_frame(first_frame)
        self.bg_subtractor.apply(gray)

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, int, np.ndarray]:
        gray = self.preprocess_frame(frame)

        fg_mask = self.bg_subtractor.apply(gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = self.parameters['min_area']
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        has_motion = len(valid_contours) > 0
        n_contours = len(valid_contours)

        return has_motion, n_contours, fg_mask
