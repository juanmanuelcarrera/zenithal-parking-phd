"""
Area-based occupancy: O_i = area(spot_i intersect union(vehicles)) / area(spot_i)

Uses the Rust core (parking_logic) when it is installed, Shapely otherwise.
"""
from enum import Enum
from typing import List, Sequence

from shapely.geometry import Polygon
from shapely.ops import unary_union

try:
    import parking_logic
except ImportError:
    parking_logic = None


class OccupancyState(str, Enum):
    FREE = "free"          # < 10 %
    LOW = "low"            # 10 - 50 %
    HIGH = "high"          # 50 - 90 %
    OCCUPIED = "occupied"  # >= 90 %


THRESHOLDS = (0.10, 0.50, 0.90)


def state_from_ratio(ratio: float, thresholds=THRESHOLDS) -> OccupancyState:
    low, high, full = thresholds
    if ratio < low:
        return OccupancyState.FREE
    if ratio < high:
        return OccupancyState.LOW
    if ratio < full:
        return OccupancyState.HIGH
    return OccupancyState.OCCUPIED


def binary_state(ratio: float) -> OccupancyState:
    """Baseline used for comparison: same engine, single 90 % threshold."""
    return OccupancyState.OCCUPIED if ratio >= THRESHOLDS[2] else OccupancyState.FREE


def _to_list(polygons):
    return [[(float(x), float(y)) for x, y in p] for p in polygons]


def occupancy_ratios(spots: Sequence, vehicles: Sequence) -> List[float]:
    if parking_logic is not None:
        return parking_logic.occupancy(_to_list(spots), _to_list(vehicles))

    if not vehicles:
        return [0.0] * len(spots)

    union = unary_union([Polygon(v).buffer(0) for v in vehicles])
    ratios = []
    for spot in spots:
        p = Polygon(spot)
        ratios.append(p.intersection(union).area / p.area if p.area else 0.0)
    return ratios
