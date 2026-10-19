"""Clustering algorithms for parking space detection."""
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler


def dbscan_clustering(contours: List[np.ndarray], eps: float = 50,
                     min_samples: int = 2, feature_type: str = 'center') -> Tuple[List[List[np.ndarray]], Dict[str, Any]]:
    if not contours:
        return [], {'method': 'dbscan_clustering', 'clusters': 0, 'noise_points': 0}

    features = extract_contour_features(contours, feature_type)

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(features)

    clustered_contours = []
    unique_labels = set(labels)
    noise_points = np.sum(labels == -1)

    for label in unique_labels:
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)[0]
        cluster = [contours[i] for i in cluster_indices]
        clustered_contours.append(cluster)

    metadata = {
        'method': 'dbscan_clustering',
        'eps': eps,
        'min_samples': min_samples,
        'feature_type': feature_type,
        'clusters': len(clustered_contours),
        'noise_points': int(noise_points),
        'description': f'DBSCAN clustering ({len(clustered_contours)} clusters)'
    }
    return clustered_contours, metadata


def kmeans_clustering(contours: List[np.ndarray], n_clusters: int = 3,
                     feature_type: str = 'center', random_state: int = 42) -> Tuple[List[List[np.ndarray]], Dict[str, Any]]:
    if not contours or len(contours) < n_clusters:
        return [contours] if contours else [], {'method': 'kmeans_clustering', 'clusters': 0}

    features = extract_contour_features(contours, feature_type)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)

    clustered_contours = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster = [contours[j] for j in cluster_indices]
        if cluster:
            clustered_contours.append(cluster)

    metadata = {
        'method': 'kmeans_clustering',
        'n_clusters': n_clusters,
        'feature_type': feature_type,
        'clusters': len(clustered_contours),
        'inertia': float(kmeans.inertia_),
        'description': f'K-means clustering ({len(clustered_contours)} clusters)'
    }
    return clustered_contours, metadata


def meanshift_clustering(contours: List[np.ndarray], bandwidth: Optional[float] = None,
                        feature_type: str = 'center') -> Tuple[List[List[np.ndarray]], Dict[str, Any]]:
    if not contours:
        return [], {'method': 'meanshift_clustering', 'clusters': 0}

    features = extract_contour_features(contours, feature_type)

    if bandwidth is None:
        bandwidth = estimate_bandwidth(features, quantile=0.2, n_samples=min(500, len(features)))
        if bandwidth <= 0:
            bandwidth = 50

    meanshift = MeanShift(bandwidth=bandwidth)
    labels = meanshift.fit_predict(features)

    clustered_contours = []
    unique_labels = set(labels)

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster = [contours[i] for i in cluster_indices]
        clustered_contours.append(cluster)

    metadata = {
        'method': 'meanshift_clustering',
        'bandwidth': float(bandwidth),
        'feature_type': feature_type,
        'clusters': len(clustered_contours),
        'description': f'Mean Shift clustering ({len(clustered_contours)} clusters)'
    }
    return clustered_contours, metadata


def proximity_clustering(contours: List[np.ndarray], max_distance: float = 100) -> Tuple[List[List[np.ndarray]], Dict[str, Any]]:
    if not contours:
        return [], {'method': 'proximity_clustering', 'clusters': 0}

    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        else:
            centers.append((0, 0))

    clustered_contours = []
    used = [False] * len(contours)

    for i, center1 in enumerate(centers):
        if used[i]:
            continue

        cluster = [contours[i]]
        used[i] = True

        for j, center2 in enumerate(centers):
            if used[j]:
                continue

            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            if distance <= max_distance:
                cluster.append(contours[j])
                used[j] = True

        clustered_contours.append(cluster)

    metadata = {
        'method': 'proximity_clustering',
        'max_distance': max_distance,
        'clusters': len(clustered_contours),
        'description': f'Proximity clustering ({len(clustered_contours)} clusters)'
    }
    return clustered_contours, metadata


def hierarchical_clustering(contours: List[np.ndarray], n_clusters: int = 3,
                           feature_type: str = 'center', linkage: str = 'ward') -> Tuple[List[List[np.ndarray]], Dict[str, Any]]:
    if not contours or len(contours) < n_clusters:
        return [contours] if contours else [], {'method': 'hierarchical_clustering', 'clusters': 0}

    try:
        from sklearn.cluster import AgglomerativeClustering

        features = extract_contour_features(contours, feature_type)

        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = clustering.fit_predict(features)

        clustered_contours = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster = [contours[j] for j in cluster_indices]
            if cluster:
                clustered_contours.append(cluster)

        metadata = {
            'method': 'hierarchical_clustering',
            'n_clusters': n_clusters,
            'feature_type': feature_type,
            'linkage': linkage,
            'clusters': len(clustered_contours),
            'description': f'Hierarchical clustering ({len(clustered_contours)} clusters)'
        }
        return clustered_contours, metadata

    except ImportError:
        return proximity_clustering(contours, max_distance=100)


def extract_contour_features(contours: List[np.ndarray], feature_type: str = 'center') -> np.ndarray:
    features = []

    for contour in contours:
        if feature_type == 'center':
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = 0, 0
            features.append([cx, cy])

        elif feature_type == 'bbox':
            x, y, w, h = cv2.boundingRect(contour)
            features.append([x, y, w, h, x + w/2, y + h/2])

        elif feature_type == 'moments':
            M = cv2.moments(contour)
            hu_moments = cv2.HuMoments(M).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            features.append(hu_moments)

    features = np.array(features)

    if len(features) > 1:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    return features


def tfm_levels_method(contours_data: List[List[List[int]]], original_image: np.ndarray = None,
                     y_threshold: float = 100, x_threshold: float = 100,
                     min_area: float = 2000) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    import random

    if not contours_data:
        return [], {'method': 'tfm_levels_method', 'rectangles': 0}

    def group_points_by_proximity_tfm(points, y_threshold=50, x_threshold=100):
        if len(points) < 4:
            return []

        y_coords = points[:, 1]

        sorted_y = np.sort(y_coords)
        y_diffs = np.diff(sorted_y)

        large_gaps = np.where(y_diffs > y_threshold)[0]
        n_rows = len(large_gaps) + 1

        if n_rows < 2:
            return []

        rows = []
        y_min = sorted_y[0]

        for gap_idx in large_gaps:
            y_max = sorted_y[gap_idx]
            mask = (y_coords >= y_min) & (y_coords <= y_max)
            row_points = points[mask]
            if len(row_points) > 0:
                row_points = row_points[np.argsort(row_points[:, 0])]
                rows.append(row_points)
            y_min = sorted_y[gap_idx + 1]

        mask = y_coords >= y_min
        row_points = points[mask]
        if len(row_points) > 0:
            row_points = row_points[np.argsort(row_points[:, 0])]
            rows.append(row_points)

        return rows

    def create_rectangles_from_rows_tfm(rows):
        rectangles = []

        for i in range(len(rows) - 1):
            top_row = rows[i]
            bottom_row = rows[i + 1]

            for j in range(min(len(top_row), len(bottom_row)) - 1):
                if j < len(top_row) - 1 and j < len(bottom_row) - 1:
                    rect = np.array([
                        top_row[j],
                        top_row[j + 1],
                        bottom_row[j + 1],
                        bottom_row[j]
                    ])

                    area = cv2.contourArea(rect)
                    if area > min_area:
                        rectangles.append(rect)

        return rectangles

    rectangles = []
    total_processed_contours = 0
    total_points_processed = 0

    for contour_vertices in contours_data:
        if len(contour_vertices) < 4:
            continue

        points = np.array(contour_vertices)
        total_points_processed += len(points)

        grouped_rows = group_points_by_proximity_tfm(points, y_threshold, x_threshold)

        if len(grouped_rows) >= 2:
            contour_rectangles = create_rectangles_from_rows_tfm(grouped_rows)
            rectangles.extend(contour_rectangles)
            total_processed_contours += 1

    metadata = {
        'method': 'tfm_levels_method',
        'original_contours': len(contours_data),
        'processed_contours': total_processed_contours,
        'points_processed': total_points_processed,
        'rectangles': len(rectangles),
        'y_threshold': y_threshold,
        'x_threshold': x_threshold,
        'min_area': min_area,
        'description': f'TFM Levels Method ({len(rectangles)} rectangles)'
    }

    return rectangles, metadata


def row_clustering(contours_data: List[List[List[int]]], original_image: np.ndarray = None,
                            simplify_distance: float = 40, y_threshold: float = 100,
                            min_area: float = 2000) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    import random

    if not contours_data:
        return [], {'method': 'row_clustering', 'rectangles': 0}

    simplified_contours = []
    total_original_points = 0
    total_simplified_points = 0

    for contour_vertices in contours_data:
        if len(contour_vertices) < 4:
            continue

        contour_points = np.array(contour_vertices)
        total_original_points += len(contour_points)

        simplified = simplify_contour_notebook(contour_points, simplify_distance)
        if len(simplified) >= 4:
            simplified_contours.append(simplified)
            total_simplified_points += len(simplified)

    rectangles = []
    total_processed_contours = 0

    for simplified_contour in simplified_contours:
        grouped_rows = group_points_by_proximity(simplified_contour, y_threshold)

        if len(grouped_rows) >= 2:
            contour_rectangles = create_rectangles_from_rows(grouped_rows, min_area)
            rectangles.extend(contour_rectangles)
            total_processed_contours += 1

    reduction_percentage = 0
    if total_original_points > 0:
        reduction_percentage = ((total_original_points - total_simplified_points) / total_original_points) * 100

    metadata = {
        'method': 'row_clustering',
        'original_contours': len(contours_data),
        'simplified_contours': len(simplified_contours),
        'original_points': total_original_points,
        'simplified_points': total_simplified_points,
        'point_reduction_percentage': reduction_percentage,
        'processed_contours': total_processed_contours,
        'rectangles': len(rectangles),
        'simplify_distance': simplify_distance,
        'y_threshold': y_threshold,
        'min_area': min_area,
        'description': f'Notebook complete method ({len(rectangles)} rectangles, {reduction_percentage:.1f}% point reduction)'
    }

    return rectangles, metadata


def simplify_contour_notebook(contour: np.ndarray, distance: float = 40) -> np.ndarray:
    def nearest_points(points, point, distance=40):
        distances = np.sum((points - point)**2, axis=1)
        distances = np.absolute(np.sqrt(distances))
        return [i for i, dist in enumerate(distances) if dist < distance]

    def medium_point(points):
        return np.around(np.mean(points, axis=0)).astype(int)

    processed_points = []
    simple_contour = np.array([])

    for i, point in enumerate(contour):
        if i not in processed_points:
            npoints = nearest_points(contour, point, distance)
            mp = np.array([medium_point(np.take(contour, npoints, 0))])
            simple_contour = mp if simple_contour.size == 0 else np.append(simple_contour, mp, 0)
            processed_points += npoints

    return simple_contour


def parking_proximity_clustering(points: List[np.ndarray], y_threshold: float = 100,
                                x_threshold: float = 50, min_area: float = 2000) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    if not points or len(points) == 0:
        return [], {'method': 'parking_proximity_clustering', 'rectangles': 0}

    rectangles = []
    total_points_processed = 0

    for point_set in points:
        if len(point_set) < 4:
            continue

        if not isinstance(point_set, np.ndarray):
            point_set = np.array(point_set)

        grouped_rows = group_points_by_proximity(point_set, y_threshold)

        if len(grouped_rows) >= 2:
            point_rectangles = create_rectangles_from_rows(grouped_rows, min_area)
            rectangles.extend(point_rectangles)
            total_points_processed += len(point_set)

    metadata = {
        'method': 'parking_proximity_clustering',
        'y_threshold': y_threshold,
        'x_threshold': x_threshold,
        'min_area': min_area,
        'points_processed': total_points_processed,
        'rectangles': len(rectangles),
        'description': f'Parking proximity clustering ({len(rectangles)} rectangles)'
    }

    return rectangles, metadata


def group_points_by_proximity(points: np.ndarray, y_threshold: float = 50) -> List[np.ndarray]:
    if len(points) < 4:
        return []

    y_coords = points[:, 1]

    sorted_y = np.sort(y_coords)
    y_diffs = np.diff(sorted_y)

    large_gaps = np.where(y_diffs > y_threshold)[0]
    n_rows = len(large_gaps) + 1

    if n_rows < 2:
        return []

    rows = []
    y_min = sorted_y[0]

    for gap_idx in large_gaps:
        y_max = sorted_y[gap_idx]
        mask = (y_coords >= y_min) & (y_coords <= y_max)
        row_points = points[mask]
        if len(row_points) > 0:
            row_points = row_points[np.argsort(row_points[:, 0])]
            rows.append(row_points)
        y_min = sorted_y[gap_idx + 1]

    mask = y_coords >= y_min
    row_points = points[mask]
    if len(row_points) > 0:
        row_points = row_points[np.argsort(row_points[:, 0])]
        rows.append(row_points)

    return rows


def create_rectangles_from_rows(rows: List[np.ndarray], min_area: float = 2000) -> List[np.ndarray]:
    rectangles = []

    for i in range(len(rows) - 1):
        top_row = rows[i]
        bottom_row = rows[i + 1]

        for j in range(min(len(top_row), len(bottom_row)) - 1):
            if j < len(top_row) - 1 and j < len(bottom_row) - 1:
                rect = np.array([
                    top_row[j],
                    top_row[j + 1],
                    bottom_row[j + 1],
                    bottom_row[j]
                ])

                area = cv2.contourArea(rect)
                if area >= min_area:
                    rectangles.append(rect)

    return rectangles


def create_rectangles_from_clusters(clustered_points: List[List[np.ndarray]],
                                   min_area: float = 2000) -> List[np.ndarray]:
    rectangles = []

    for cluster in clustered_points:
        if not cluster:
            continue

        for contour_points in cluster:
            if contour_points.ndim == 3:
                points = contour_points.reshape(-1, 2)
            else:
                points = contour_points

            rows = group_points_by_proximity(points)
            if len(rows) >= 2:
                cluster_rectangles = create_rectangles_from_rows(rows, min_area)
                rectangles.extend(cluster_rectangles)

    return rectangles


AVAILABLE_CLUSTERING = {
    'clustered_kmeans': kmeans_clustering,
    'tfm_levels': tfm_levels_method,
    'proximity_rows': row_clustering,
    'parking_proximity': parking_proximity_clustering,
    'clustered_dbscan': dbscan_clustering,
    'clustered_meanshift': meanshift_clustering,
    'clustered_hierarchical': hierarchical_clustering,
    'clustered_proximity': proximity_clustering,
}


def apply_clustering(contours: List[np.ndarray], method: str, **kwargs) -> Tuple[List[List[np.ndarray]], Dict[str, Any]]:
    if method not in AVAILABLE_CLUSTERING:
        raise ValueError(f"Method not available: {method}. Available: {list(AVAILABLE_CLUSTERING.keys())}")

    return AVAILABLE_CLUSTERING[method](contours, **kwargs)
