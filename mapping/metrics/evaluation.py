"""Evaluation metrics for parking space detection algorithms."""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time


def calculate_contour_metrics(contours: List[np.ndarray]) -> Dict[str, Any]:
    if not contours:
        return {
            'count': 0,
            'total_area': 0,
            'mean_area': 0,
            'std_area': 0,
            'min_area': 0,
            'max_area': 0,
            'total_perimeter': 0,
            'mean_perimeter': 0
        }

    areas = [cv2.contourArea(contour) for contour in contours]
    perimeters = [cv2.arcLength(contour, True) for contour in contours]

    return {
        'count': len(contours),
        'total_area': sum(areas),
        'mean_area': np.mean(areas),
        'std_area': np.std(areas),
        'min_area': min(areas),
        'max_area': max(areas),
        'total_perimeter': sum(perimeters),
        'mean_perimeter': np.mean(perimeters)
    }


def calculate_clustering_metrics(clustered_contours: List[List[np.ndarray]]) -> Dict[str, Any]:
    if not clustered_contours:
        return {
            'num_clusters': 0,
            'total_contours': 0,
            'mean_cluster_size': 0,
            'std_cluster_size': 0,
            'min_cluster_size': 0,
            'max_cluster_size': 0,
            'cluster_sizes': []
        }

    cluster_sizes = [len(cluster) for cluster in clustered_contours]
    total_contours = sum(cluster_sizes)

    return {
        'num_clusters': len(clustered_contours),
        'total_contours': total_contours,
        'mean_cluster_size': np.mean(cluster_sizes),
        'std_cluster_size': np.std(cluster_sizes),
        'min_cluster_size': min(cluster_sizes),
        'max_cluster_size': max(cluster_sizes),
        'cluster_sizes': cluster_sizes
    }


def calculate_edge_density(edge_image: np.ndarray) -> Dict[str, float]:
    total_pixels = edge_image.shape[0] * edge_image.shape[1]
    edge_pixels = np.sum(edge_image > 0)

    return {
        'edge_density': edge_pixels / total_pixels,
        'edge_pixels': int(edge_pixels),
        'total_pixels': total_pixels,
        'non_edge_pixels': total_pixels - int(edge_pixels)
    }


def calculate_processing_time(func, *args, **kwargs) -> Tuple[Any, float]:
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    processing_time = end_time - start_time

    return result, processing_time


def evaluate_threshold_quality(original_image: np.ndarray, thresholded_image: np.ndarray) -> Dict[str, float]:
    hist_original = cv2.calcHist([original_image], [0], None, [256], [0, 256])

    _, otsu_thresh = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    otsu_similarity = np.mean(thresholded_image == otsu_thresh)

    foreground_pixels = np.sum(thresholded_image == 255)
    background_pixels = np.sum(thresholded_image == 0)
    total_pixels = thresholded_image.shape[0] * thresholded_image.shape[1]

    fg_bg_ratio = foreground_pixels / background_pixels if background_pixels > 0 else float('inf')

    return {
        'otsu_similarity': otsu_similarity,
        'foreground_ratio': foreground_pixels / total_pixels,
        'background_ratio': background_pixels / total_pixels,
        'fg_bg_ratio': fg_bg_ratio,
        'foreground_pixels': int(foreground_pixels),
        'background_pixels': int(background_pixels)
    }


def calculate_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    contrast = std_intensity

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'contrast': contrast,
        'sharpness': sharpness,
        'entropy': entropy,
        'dynamic_range': int(gray.max() - gray.min())
    }


def compare_methods(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {'comparison': 'No results to compare'}

    comparison = {
        'methods_compared': list(results.keys()),
        'total_methods': len(results)
    }

    metrics_to_compare = ['processing_time', 'count', 'num_clusters', 'edge_density']

    for metric in metrics_to_compare:
        values = []
        for method, result in results.items():
            if metric in result:
                values.append((method, result[metric]))

        if values:
            values.sort(key=lambda x: x[1])
            comparison[f'best_{metric}'] = values[0][0]
            comparison[f'worst_{metric}'] = values[-1][0]
            comparison[f'{metric}_range'] = values[-1][1] - values[0][1]

    return comparison


def create_evaluation_report(method_name: str, result: Any, metadata: Dict[str, Any],
                           processing_time: float, input_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
    report = {
        'method': method_name,
        'processing_time': processing_time,
        'metadata': metadata
    }

    if isinstance(result, np.ndarray):
        if len(result.shape) == 2:
            if np.array_equal(result, result.astype(bool) * 255):
                report['edge_metrics'] = calculate_edge_density(result)
            else:
                report['image_quality'] = calculate_image_quality_metrics(result)

        if input_image is not None and len(result.shape) == 2:
            if np.array_equal(result, result.astype(bool) * 255):
                report['threshold_quality'] = evaluate_threshold_quality(input_image, result)

    elif isinstance(result, list):
        if result and isinstance(result[0], np.ndarray):
            report['contour_metrics'] = calculate_contour_metrics(result)
        elif result and isinstance(result[0], list):
            report['clustering_metrics'] = calculate_clustering_metrics(result)
            all_contours = [contour for cluster in result for contour in cluster]
            report['contour_metrics'] = calculate_contour_metrics(all_contours)

    return report


def benchmark_methods(methods: Dict[str, callable], input_data: Any, **kwargs) -> Dict[str, Dict[str, Any]]:
    results = {}

    for method_name, method_func in methods.items():
        try:
            result, processing_time = calculate_processing_time(method_func, input_data, **kwargs)

            if isinstance(result, tuple) and len(result) == 2:
                actual_result, metadata = result
            else:
                actual_result, metadata = result, {}

            report = create_evaluation_report(
                method_name, actual_result, metadata, processing_time, input_data
            )

            results[method_name] = report

        except Exception as e:
            results[method_name] = {
                'method': method_name,
                'error': str(e),
                'processing_time': 0,
                'metadata': {}
            }

    results['comparison'] = compare_methods({k: v for k, v in results.items() if 'error' not in v})

    return results
