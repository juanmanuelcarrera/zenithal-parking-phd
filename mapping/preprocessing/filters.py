"""Pre-processing filters module optimized for parking space detection."""
import cv2
import numpy as np
from typing import Tuple, Dict, Any


def median_blur(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict[str, Any]]:
    filtered = cv2.medianBlur(image, kernel_size)
    metadata = {
        'method': 'median_blur',
        'kernel_size': kernel_size,
        'description': 'Median filter for noise reduction'
    }
    return filtered, metadata


def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75,
                    sigma_space: float = 75) -> Tuple[np.ndarray, Dict[str, Any]]:
    filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    metadata = {
        'method': 'bilateral_filter',
        'd': d,
        'sigma_color': sigma_color,
        'sigma_space': sigma_space,
        'description': 'Bilateral filter that preserves edges'
    }
    return filtered, metadata


def morphological_filter(image: np.ndarray, operation: str = 'opening',
                        kernel_size: int = 5) -> Tuple[np.ndarray, Dict[str, Any]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == 'opening':
        filtered = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        filtered = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'gradient':
        filtered = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    else:
        raise ValueError(f"Unsupported morphological operation: {operation}")

    metadata = {
        'method': 'morphological_filter',
        'operation': operation,
        'kernel_size': kernel_size,
        'description': f'Morphological operation: {operation}'
    }
    return filtered, metadata


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma_x: float = 1.0,
                 sigma_y: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x, sigma_y)
    metadata = {
        'method': 'gaussian_blur',
        'kernel_size': kernel_size,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'description': 'Gaussian blur for smooth noise reduction'
    }
    return filtered, metadata


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0,
                     tile_grid_size: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, Dict[str, Any]]:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)

    metadata = {
        'method': 'clahe_enhancement',
        'clip_limit': clip_limit,
        'tile_grid_size': tile_grid_size,
        'description': 'Contrast Limited Adaptive Histogram Equalization'
    }
    return enhanced, metadata


def gradient_filter(image: np.ndarray, gradient_type: str = 'sobel_combined',
                   ksize: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    if gradient_type == 'sobel_x':
        filtered = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        filtered = np.absolute(filtered)
    elif gradient_type == 'sobel_y':
        filtered = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        filtered = np.absolute(filtered)
    elif gradient_type == 'sobel_combined':
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        filtered = np.sqrt(grad_x**2 + grad_y**2)
    elif gradient_type == 'scharr':
        grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        filtered = np.sqrt(grad_x**2 + grad_y**2)
    else:
        raise ValueError(f"Unsupported gradient type: {gradient_type}")

    filtered = np.uint8(np.clip(filtered, 0, 255))

    metadata = {
        'method': 'gradient_filter',
        'gradient_type': gradient_type,
        'ksize': ksize,
        'description': f'Gradient filter: {gradient_type}'
    }
    return filtered, metadata


def laplacian_filter(image: np.ndarray, kernel_size: int = 3, scale: float = 1.0,
                    delta: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)

    filtered = np.absolute(laplacian)
    filtered = np.uint8(np.clip(filtered, 0, 255))

    metadata = {
        'method': 'laplacian_filter',
        'kernel_size': kernel_size,
        'scale': scale,
        'delta': delta,
        'description': 'Laplacian filter for edge detection'
    }
    return filtered, metadata


def combined_filter(image: np.ndarray, methods: list = ['gaussian', 'bilateral']) -> Tuple[np.ndarray, Dict[str, Any]]:
    result = image.copy()
    applied_methods = []

    for method in methods:
        if method == 'gaussian':
            result, meta = gaussian_blur(result, kernel_size=5, sigma_x=1.0, sigma_y=1.0)
            applied_methods.append(meta)
        elif method == 'bilateral':
            result, meta = bilateral_filter(result, d=9, sigma_color=75, sigma_space=75)
            applied_methods.append(meta)
        elif method == 'median':
            result, meta = median_blur(result, kernel_size=5)
            applied_methods.append(meta)
        elif method == 'morphological':
            result, meta = morphological_filter(result, operation='opening', kernel_size=3)
            applied_methods.append(meta)
        elif method == 'morphological_closing':
            result, meta = morphological_filter(result, operation='closing', kernel_size=3)
            applied_methods.append(meta)
        elif method == 'clahe':
            result, meta = clahe_enhancement(result)
            applied_methods.append(meta)
        elif method == 'laplacian':
            result, meta = laplacian_filter(result, kernel_size=3, scale=1, delta=0)
            applied_methods.append(meta)

    metadata = {
        'method': 'combined_filter',
        'applied_methods': applied_methods,
        'description': f'Noise reduction combination: {" -> ".join(methods)}'
    }
    return result, metadata


AVAILABLE_FILTERS = {
    'median': median_blur,
    'bilateral': bilateral_filter,
    'gaussian': gaussian_blur,
    'morphological': morphological_filter,
    'gradient': gradient_filter,
    'laplacian': laplacian_filter,
    'clahe': clahe_enhancement,
    'combined': combined_filter
}


def apply_filter(image: np.ndarray, method: str, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method not in AVAILABLE_FILTERS:
        raise ValueError(f"Method not available: {method}. Available: {list(AVAILABLE_FILTERS.keys())}")

    return AVAILABLE_FILTERS[method](image, **kwargs)
