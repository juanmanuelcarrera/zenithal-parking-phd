"""Thresholding methods for parking space detection."""
import cv2
import numpy as np
import time
from typing import Tuple, Dict, Any
from scipy.signal import find_peaks


def otsu_threshold(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    threshold_value, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    metadata = {
        'method': 'otsu_threshold',
        'threshold_value': float(threshold_value),
        'description': 'Otsu automatic thresholding'
    }
    return thresholded, metadata


def adaptive_threshold(image: np.ndarray, max_value: float = 255,
                      adaptive_method: str = 'gaussian', threshold_type: str = 'binary',
                      block_size: int = 11, C: float = 2) -> Tuple[np.ndarray, Dict[str, Any]]:
    adaptive_methods = {
        'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
        'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    }

    threshold_types = {
        'binary': cv2.THRESH_BINARY,
        'binary_inv': cv2.THRESH_BINARY_INV
    }

    thresholded = cv2.adaptiveThreshold(
        image, max_value,
        adaptive_methods[adaptive_method],
        threshold_types[threshold_type],
        block_size, C
    )

    metadata = {
        'method': 'adaptive_threshold',
        'max_value': max_value,
        'adaptive_method': adaptive_method,
        'threshold_type': threshold_type,
        'block_size': block_size,
        'C': C,
        'description': f'Adaptive thresholding ({adaptive_method})'
    }
    return thresholded, metadata


def multi_threshold(image: np.ndarray, thresholds: list = [85, 170]) -> Tuple[np.ndarray, Dict[str, Any]]:
    result = np.zeros_like(image)

    sorted_thresholds = sorted(thresholds)

    for i, thresh in enumerate(sorted_thresholds):
        if i == 0:
            mask = image <= thresh
            result[mask] = 0
        else:
            mask = (image > sorted_thresholds[i-1]) & (image <= thresh)
            result[mask] = int(255 * (i + 1) / (len(sorted_thresholds) + 1))

    mask = image > sorted_thresholds[-1]
    result[mask] = 255

    metadata = {
        'method': 'multi_threshold',
        'thresholds': sorted_thresholds,
        'description': f'Multiple thresholds: {sorted_thresholds}'
    }
    return result, metadata


def niblack_threshold(image: np.ndarray, window_size: int = 15, k: float = -0.2) -> Tuple[np.ndarray, Dict[str, Any]]:
    img_float = image.astype(np.float32)

    kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
    local_mean = cv2.filter2D(img_float, -1, kernel)

    local_var = cv2.filter2D(img_float * img_float, -1, kernel) - local_mean * local_mean
    local_std = np.sqrt(np.maximum(local_var, 0))

    threshold = local_mean + k * local_std

    thresholded = np.where(img_float > threshold, 255, 0).astype(np.uint8)

    metadata = {
        'method': 'niblack_threshold',
        'window_size': window_size,
        'k': k,
        'description': f'Niblack local thresholding (k={k})'
    }
    return thresholded, metadata


def sauvola_threshold(image: np.ndarray, window_size: int = 15, k: float = 0.5, R: float = 128) -> Tuple[np.ndarray, Dict[str, Any]]:
    img_float = image.astype(np.float32)

    kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
    local_mean = cv2.filter2D(img_float, -1, kernel)

    local_var = cv2.filter2D(img_float * img_float, -1, kernel) - local_mean * local_mean
    local_std = np.sqrt(np.maximum(local_var, 0))

    threshold = local_mean * (1 + k * ((local_std / R) - 1))

    thresholded = np.where(img_float > threshold, 255, 0).astype(np.uint8)

    metadata = {
        'method': 'sauvola_threshold',
        'window_size': window_size,
        'k': k,
        'R': R,
        'description': f'Sauvola local thresholding (k={k}, R={R})'
    }
    return thresholded, metadata


def kapur_entropy_threshold(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    start_time = time.time()

    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()

    max_entropy = -1
    optimal_threshold = 0

    for t in range(1, 255):
        w0 = np.sum(hist[:t])
        if w0 == 0:
            continue

        w1 = np.sum(hist[t:])
        if w1 == 0:
            continue

        h0 = -np.sum([p/w0 * np.log(p/w0) for p in hist[:t] if p > 0])
        h1 = -np.sum([p/w1 * np.log(p/w1) for p in hist[t:] if p > 0])

        total_entropy = h0 + h1

        if total_entropy > max_entropy:
            max_entropy = total_entropy
            optimal_threshold = t

    _, thresholded = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)
    processing_time = time.time() - start_time

    metadata = {
        'method': 'kapur_entropy_threshold',
        'threshold_value': optimal_threshold,
        'max_entropy': max_entropy,
        'processing_time': processing_time,
        'description': f'Kapur entropy method (threshold={optimal_threshold})'
    }
    return thresholded, metadata


def kmeans_threshold(image: np.ndarray, k: int = 2, random_state: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return otsu_threshold(image)

    start_time = time.time()

    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_

    if k == 2:
        if centers[0] < centers[1]:
            bg_label, fg_label = 0, 1
        else:
            bg_label, fg_label = 1, 0

        thresholded = np.zeros_like(image)
        labels = labels.reshape(image.shape)
        thresholded[labels == fg_label] = 255
    else:
        sorted_indices = np.argsort(centers.flatten())
        thresholded = np.zeros_like(image)
        labels = labels.reshape(image.shape)

        for i, cluster_idx in enumerate(sorted_indices):
            intensity = int(255 * i / (k - 1))
            thresholded[labels == cluster_idx] = intensity

    processing_time = time.time() - start_time

    metadata = {
        'method': 'kmeans_threshold',
        'k_clusters': k,
        'bg_center': float(centers[bg_label][0]) if k == 2 else None,
        'fg_center': float(centers[fg_label][0]) if k == 2 else None,
        'all_centers': [float(c[0]) for c in centers],
        'inertia': float(kmeans.inertia_),
        'processing_time': processing_time,
        'description': f'K-means clustering threshold (k={k})'
    }
    return thresholded, metadata


def local_minima_threshold(image: np.ndarray, sigma: float = 2.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        return otsu_threshold(image)

    start_time = time.time()

    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    smoothed_hist = gaussian_filter1d(hist, sigma=sigma)

    peaks, _ = find_peaks(smoothed_hist, height=np.max(smoothed_hist) * 0.1)
    valleys, _ = find_peaks(-smoothed_hist, height=-np.max(smoothed_hist) * 0.1)

    if len(valleys) > 0:
        depths = smoothed_hist[valleys]
        min_valley_idx = np.argmin(depths)
        optimal_threshold = valleys[min_valley_idx]
    else:
        optimal_threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, thresholded = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)
    processing_time = time.time() - start_time

    metadata = {
        'method': 'local_minima_threshold',
        'threshold_value': int(optimal_threshold),
        'num_peaks': len(peaks),
        'num_valleys': len(valleys),
        'sigma': sigma,
        'processing_time': processing_time,
        'description': f'Local minima threshold (threshold={optimal_threshold}, {len(valleys)} valleys)'
    }
    return thresholded, metadata


def grabcut_threshold(image: np.ndarray, iterations: int = 5) -> Tuple[np.ndarray, Dict[str, Any]]:
    start_time = time.time()

    if len(image.shape) == 2:
        img_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_3ch = image.copy()

    mask = np.zeros(image.shape[:2], np.uint8)

    height, width = image.shape[:2]
    margin_x, margin_y = width // 6, height // 6
    rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_3ch, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        thresholded = mask2 * 255

        success = True

    except Exception as e:
        print(f"Warning: GrabCut failed ({e}), using Otsu fallback")
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        success = False

    processing_time = time.time() - start_time

    metadata = {
        'method': 'grabcut_threshold',
        'iterations': iterations,
        'rect': rect,
        'success': success,
        'processing_time': processing_time,
        'description': f'GrabCut segmentation ({iterations} iterations)'
    }
    return thresholded, metadata


AVAILABLE_THRESHOLDS = {
    'otsu': otsu_threshold,
    'adaptive': adaptive_threshold,
    'multi': multi_threshold,
    'niblack': niblack_threshold,
    'sauvola': sauvola_threshold,
    'kapur_entropy': kapur_entropy_threshold,
    'kmeans': kmeans_threshold,
    'local_minima': local_minima_threshold,
    'grabcut': grabcut_threshold
}


def apply_threshold(image: np.ndarray, method: str, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method not in AVAILABLE_THRESHOLDS:
        raise ValueError(f"Method not available: {method}. Available: {list(AVAILABLE_THRESHOLDS.keys())}")

    return AVAILABLE_THRESHOLDS[method](image, **kwargs)
