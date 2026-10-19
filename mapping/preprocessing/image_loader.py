"""Image loading and preprocessing utilities for parking space detection."""
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os


def load_image(image_path: str, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    if roi is not None:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]

    return image


def get_default_roi(image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    height, width = image_shape[:2]
    return (0, 0, width, height)


def validate_roi(image_shape: Tuple[int, int, int], roi: Tuple[int, int, int, int]) -> bool:
    height, width = image_shape[:2]
    x, y, w, h = roi

    return (x >= 0 and y >= 0 and
            x + w <= width and y + h <= height and
            w > 0 and h > 0)


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
