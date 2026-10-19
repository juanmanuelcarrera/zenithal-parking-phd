"""Vertex and contour extraction methods for parking space detection."""
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.cluster import DBSCAN


def contour_approximation(image: np.ndarray, min_area: int = 1000,
                         epsilon_factor: float = 0.005,
                         retrieval_mode: int = cv2.RETR_LIST,
                         approximation_method: int = cv2.CHAIN_APPROX_SIMPLE) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(image, retrieval_mode, approximation_method)

    vertices_list = []
    contour_stats = []

    total_contours_found = len(contours)
    areas = [cv2.contourArea(cnt) for cnt in contours] if contours else []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area > min_area:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if approx is not None and len(approx) > 0:
                vertices = approx.squeeze()
                if len(vertices.shape) == 1 and len(vertices) == 2:
                    vertices = vertices.reshape(1, 2)
                elif len(vertices.shape) == 2 and vertices.shape[1] == 2:
                    pass
                else:
                    vertices = vertices.reshape(-1, 2)

                vertices_list.append(vertices)

                contour_stats.append({
                    'area': area,
                    'perimeter': perimeter,
                    'vertices_count': len(vertices),
                    'epsilon_used': epsilon,
                    'compactness': 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0,
                    'bounding_rect': cv2.boundingRect(contour),
                    'original_contour_points': len(contour)
                })

    metadata = {
        'method': 'contour_approximation',
        'total_contours': total_contours_found,
        'filtered_contours': len(vertices_list),
        'min_area': min_area,
        'epsilon_factor': epsilon_factor,
        'retrieval_mode': retrieval_mode,
        'contour_stats': contour_stats,
        'total_vertices': sum(len(v) for v in vertices_list),
        'area_range': {'min': min(areas) if areas else 0, 'max': max(areas) if areas else 0},
        'areas_above_threshold': sum(1 for a in areas if a > min_area),
        'description': f'Contour approximation ({len(vertices_list)} shapes, {sum(len(v) for v in vertices_list)} vertices from {total_contours_found} total contours)'
    }

    return vertices_list, metadata


def corner_detection(image: np.ndarray, max_corners: int = 100,
                    quality_level: float = 0.3, min_distance: float = 10,
                    block_size: int = 3, use_harris: bool = False,
                    k: float = 0.04) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    corners = cv2.goodFeaturesToTrack(
        image, maxCorners=max_corners, qualityLevel=quality_level,
        minDistance=min_distance, blockSize=block_size,
        useHarrisDetector=use_harris, k=k
    )

    vertices_list = []
    if corners is not None:
        corners = np.int32(corners).reshape(-1, 2)
        vertices_list = [corners]

    metadata = {
        'method': 'corner_detection',
        'detector_type': 'Harris' if use_harris else 'Shi-Tomasi',
        'corners_detected': len(corners) if corners is not None else 0,
        'max_corners': max_corners,
        'quality_level': quality_level,
        'min_distance': min_distance,
        'total_vertices': len(corners) if corners is not None else 0,
        'description': f'Corner detection ({len(corners) if corners is not None else 0} corners)'
    }

    return vertices_list, metadata


def bounding_rectangles(image: np.ndarray, min_area: int = 1000,
                       use_min_area_rect: bool = True) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vertices_list = []
    rectangle_stats = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_area:
            if use_min_area_rect:
                rect = cv2.minAreaRect(contour)
                vertices = cv2.boxPoints(rect)
                vertices = np.int32(vertices)

                width, height = rect[1]
                angle = rect[2]
                rect_area = width * height

                rectangle_stats.append({
                    'contour_area': area,
                    'rectangle_area': rect_area,
                    'width': width,
                    'height': height,
                    'angle': angle,
                    'aspect_ratio': max(width, height) / min(width, height) if min(width, height) > 0 else 0,
                    'fill_ratio': area / rect_area if rect_area > 0 else 0
                })
            else:
                x, y, w, h = cv2.boundingRect(contour)
                vertices = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])

                rect_area = w * h
                rectangle_stats.append({
                    'contour_area': area,
                    'rectangle_area': rect_area,
                    'width': w,
                    'height': h,
                    'angle': 0,
                    'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 0,
                    'fill_ratio': area / rect_area if rect_area > 0 else 0
                })

            vertices_list.append(vertices)

    metadata = {
        'method': 'bounding_rectangles',
        'rectangle_type': 'minimum_area' if use_min_area_rect else 'axis_aligned',
        'rectangles_found': len(vertices_list),
        'min_area': min_area,
        'rectangle_stats': rectangle_stats,
        'total_vertices': len(vertices_list) * 4,
        'description': f'Bounding rectangles ({len(vertices_list)} rectangles, {len(vertices_list) * 4} vertices)'
    }

    return vertices_list, metadata


def line_intersections(image: np.ndarray, rho: float = 1, theta: float = np.pi/180,
                      threshold: int = 100, min_line_length: int = 50,
                      max_line_gap: int = 10, intersection_threshold: float = 5.0) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    lines = cv2.HoughLinesP(image, rho, theta, threshold,
                           minLineLength=min_line_length, maxLineGap=max_line_gap)

    intersections = []

    if lines is not None and len(lines) > 1:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]

                intersection = find_line_intersection(line1, line2)

                if intersection is not None:
                    x, y = intersection
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        if (point_to_line_distance(intersection, line1) < intersection_threshold and
                            point_to_line_distance(intersection, line2) < intersection_threshold):
                            intersections.append(intersection)

    if intersections:
        intersections = np.array(intersections)
        if len(intersections) > 1:
            clustering = DBSCAN(eps=intersection_threshold, min_samples=1).fit(intersections)

            unique_intersections = []
            for cluster_id in set(clustering.labels_):
                cluster_points = intersections[clustering.labels_ == cluster_id]
                centroid = np.mean(cluster_points, axis=0)
                unique_intersections.append(centroid.astype(int))

            intersections = unique_intersections
        else:
            intersections = [intersections[0].astype(int)]

    vertices_list = [np.array(intersections)] if intersections else []

    metadata = {
        'method': 'line_intersections',
        'lines_detected': len(lines) if lines is not None else 0,
        'intersections_found': len(intersections),
        'threshold': threshold,
        'min_line_length': min_line_length,
        'intersection_threshold': intersection_threshold,
        'total_vertices': len(intersections),
        'description': f'Line intersections ({len(intersections)} intersections from {len(lines) if lines is not None else 0} lines)'
    }

    return vertices_list, metadata


def clustered_contours(image: np.ndarray, min_area: int = 1000,
                      epsilon_factor: float = 0.005, cluster_distance: float = 50.0) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    vertices_list, contour_metadata = contour_approximation(image, min_area, epsilon_factor)

    if not vertices_list:
        return vertices_list, {**contour_metadata, 'method': 'clustered_contours'}

    all_vertices = np.vstack(vertices_list)

    clustering = DBSCAN(eps=cluster_distance, min_samples=2).fit(all_vertices)

    clustered_vertices = []
    cluster_stats = []

    for cluster_id in set(clustering.labels_):
        if cluster_id != -1:
            cluster_points = all_vertices[clustering.labels_ == cluster_id]
            clustered_vertices.append(cluster_points)

            cluster_stats.append({
                'cluster_id': cluster_id,
                'points_count': len(cluster_points),
                'centroid': np.mean(cluster_points, axis=0),
                'spread': np.std(cluster_points, axis=0)
            })

    metadata = {
        'method': 'clustered_contours',
        'original_vertices': len(all_vertices),
        'clusters_found': len(clustered_vertices),
        'cluster_distance': cluster_distance,
        'cluster_stats': cluster_stats,
        'total_vertices': len(all_vertices),
        'description': f'Clustered contours ({len(clustered_vertices)} clusters from {len(all_vertices)} vertices)'
    }

    return clustered_vertices, metadata


def find_line_intersection(line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return np.array([x, y])


def point_to_line_distance(point: np.ndarray, line: np.ndarray) -> float:
    x0, y0 = point
    x1, y1, x2, y2 = line

    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if line_length < 1e-10:
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

    distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_length

    return distance


AVAILABLE_EXTRACTORS = {
    'contours': contour_approximation,
    'corners': corner_detection,
    'rectangles': bounding_rectangles,
    'intersections': line_intersections,
    'clustered': clustered_contours
}


def apply_extraction(image: np.ndarray, method: str, **kwargs) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    if method not in AVAILABLE_EXTRACTORS:
        raise ValueError(f"Method not available: {method}. Available: {list(AVAILABLE_EXTRACTORS.keys())}")

    return AVAILABLE_EXTRACTORS[method](image, **kwargs)
