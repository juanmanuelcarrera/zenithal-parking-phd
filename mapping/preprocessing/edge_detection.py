"""Line and edge detection methods for parking space detection."""
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List


def canny_edge_detection(image: np.ndarray, low_threshold: int = 50,
                        high_threshold: int = 150, aperture_size: int = 3,
                        l2_gradient: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    edges = cv2.Canny(image, low_threshold, high_threshold,
                      apertureSize=aperture_size, L2gradient=l2_gradient)

    metadata = {
        'method': 'canny_edge_detection',
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'aperture_size': aperture_size,
        'l2_gradient': l2_gradient,
        'description': f'Canny edges ({low_threshold}-{high_threshold})'
    }
    return edges, metadata


def hough_lines_detection(image: np.ndarray, rho: float = 1, theta: float = np.pi/180,
                         threshold: int = 100, min_line_length: int = 50,
                         max_line_gap: int = 10) -> Tuple[np.ndarray, Dict[str, Any]]:
    line_image = np.zeros_like(image)

    lines = cv2.HoughLinesP(image, rho, theta, threshold,
                           minLineLength=min_line_length, maxLineGap=max_line_gap)

    lines_count = 0
    if lines is not None:
        lines_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)

    metadata = {
        'method': 'hough_lines_detection',
        'rho': rho,
        'theta': theta,
        'threshold': threshold,
        'min_line_length': min_line_length,
        'max_line_gap': max_line_gap,
        'lines_detected': lines_count,
        'description': f'Hough lines ({lines_count} detected)'
    }
    return line_image, metadata


def sobel_edge_detection(image: np.ndarray, ksize: int = 3, dx: int = 1, dy: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
    grad_x = cv2.Sobel(image, cv2.CV_64F, dx, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, dy, ksize=ksize)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    metadata = {
        'method': 'sobel_edge_detection',
        'ksize': ksize,
        'dx': dx,
        'dy': dy,
        'description': f'Sobel edges (kernel={ksize})'
    }
    return magnitude, metadata


def laplacian_edge_detection(image: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    laplacian = np.uint8(np.absolute(laplacian))

    metadata = {
        'method': 'laplacian_edge_detection',
        'ksize': ksize,
        'description': f'Laplacian edges (kernel={ksize})'
    }
    return laplacian, metadata


def gabor_filter(image: np.ndarray, ksize: int = 31, sigma: float = 4,
                theta: float = 0, lambd: float = 10, gamma: float = 0.5,
                psi: float = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)

    metadata = {
        'method': 'gabor_filter',
        'ksize': ksize,
        'sigma': sigma,
        'theta': theta,
        'lambda': lambd,
        'gamma': gamma,
        'psi': psi,
        'description': f'Gabor filter (theta={theta:.2f}, lambda={lambd})'
    }
    return filtered, metadata


def directional_gradient(image: np.ndarray, direction: str = 'horizontal') -> Tuple[np.ndarray, Dict[str, Any]]:
    if direction == 'horizontal':
        kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        gradient = cv2.filter2D(image, -1, kernel)
    elif direction == 'vertical':
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        gradient = cv2.filter2D(image, -1, kernel)
    elif direction == 'both':
        kernel_h = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_h = cv2.filter2D(image, -1, kernel_h)

        kernel_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        grad_v = cv2.filter2D(image, -1, kernel_v)

        gradient = np.sqrt(grad_h**2 + grad_v**2)
    else:
        raise ValueError("Direction must be 'horizontal', 'vertical', or 'both'")

    gradient = np.uint8(np.absolute(gradient))

    metadata = {
        'method': 'directional_gradient',
        'direction': direction,
        'description': f'Directional gradient ({direction})'
    }
    return gradient, metadata


def morphological_gradient(image: np.ndarray, kernel_size: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    metadata = {
        'method': 'morphological_gradient',
        'kernel_size': kernel_size,
        'description': f'Morphological gradient (kernel={kernel_size})'
    }
    return gradient, metadata


AVAILABLE_DETECTORS = {
    'canny': canny_edge_detection,
    'hough_lines': hough_lines_detection,
    'sobel': sobel_edge_detection,
    'laplacian': laplacian_edge_detection,
    'gabor': gabor_filter,
    'directional': directional_gradient,
    'morphological': morphological_gradient
}


def apply_detection(image: np.ndarray, method: str, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method not in AVAILABLE_DETECTORS:
        raise ValueError(f"Method not available: {method}. Available: {list(AVAILABLE_DETECTORS.keys())}")

    return AVAILABLE_DETECTORS[method](image, **kwargs)
