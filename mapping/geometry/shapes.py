from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon
from typing import Iterable

def rectangularity(poly_xy: np.ndarray) -> float:
    if len(poly_xy) < 3:
        return 0.0
    shp = Polygon(poly_xy)
    if shp.area <= 0:
        return 0.0
    rect = shp.minimum_rotated_rectangle
    return float(shp.area / max(rect.area, 1e-9))

def order_clockwise(poly_xy: np.ndarray) -> np.ndarray:
    c = poly_xy.mean(axis=0)
    ang = np.arctan2(poly_xy[:, 1] - c[1], poly_xy[:, 0] - c[0])
    idx = np.argsort(ang)
    return poly_xy[idx]

def nearest_points(points, point, distance=40):
    distances = np.sum((points - point)**2, axis=1)
    distances = np.absolute(np.sqrt(distances))
    return [i for i, dist in enumerate(distances) if dist < distance]

def medium_point(points):
    return np.around(np.mean(points, axis=0)).astype(int)

def simplify_contour(contour):
    processed_points = []
    simple_contour = np.array([])

    for i, point in enumerate(contour):
        if i not in processed_points:
            npoints = nearest_points(contour, point)
            mp = np.array([medium_point(np.take(contour, npoints, 0))])
            simple_contour = mp if simple_contour.size == 0 else np.append(simple_contour, mp, 0)
            processed_points += npoints

    return simple_contour

def sort_points_ttb(points):
    pts = np.array([list(point) for point in points])
    return pts[np.argsort(pts[:, 1]), :]

def get_top_point(points):
    return sort_points_ttb(points)[0]

def get_bottom_point(points):
    return sort_points_ttb(points)[-1]

def sort_points_ltr(points):
    pts = np.array([list(point) for point in points])
    return pts[np.argsort(pts[:, 0]), :]

def get_left_point(points):
    return sort_points_ltr(points)[0]

def get_right_point(points):
    return sort_points_ltr(points)[-1]

def intersection(spot_shape, vehicle_shape):
    vehicle = Polygon(vehicle_shape)
    spot = Polygon(spot_shape)
    if vehicle.intersects(spot):
        intersection = vehicle.intersection(spot)

        left_point = get_left_point(intersection.boundary.coords)[0]
        right_point = get_right_point(intersection.boundary.coords)[0]

        top_point = get_top_point(spot.boundary.coords)[1]
        bottom_point = get_bottom_point(spot.boundary.coords)[1]

        final_intersection = Polygon([
            (left_point, top_point),
            (left_point, bottom_point),
            (right_point, bottom_point),
            (right_point, top_point)
        ])

        return True, final_intersection.area

    return False, 0
