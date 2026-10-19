"""
Best-fit spot assignment.

Candidates: free spots where l(p) >= l(v) + dl and w(p) >= w(v) + dw.
Cost: J(p) = alpha * dist(p, E) / Dmax + beta * (A(p) - A(v)) / A(p), alpha + beta = 1.
"""
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass
class Spot:
    id: str
    length: float  # metres
    width: float
    center: Tuple[float, float]
    free: bool = True

    @property
    def area(self) -> float:
        return self.length * self.width


@dataclass
class Vehicle:
    length: float
    width: float

    @property
    def area(self) -> float:
        return self.length * self.width


def candidates(spots: Iterable[Spot], v: Vehicle, margin_l: float = 0.3, margin_w: float = 0.2) -> List[Spot]:
    return [p for p in spots
            if p.free and p.length >= v.length + margin_l and p.width >= v.width + margin_w]


def cost(p: Spot, v: Vehicle, entrance: Tuple[float, float], d_max: float,
         alpha: float = 0.5) -> float:
    beta = 1.0 - alpha
    dist = math.dist(p.center, entrance) / d_max
    waste = (p.area - v.area) / p.area
    return alpha * dist + beta * waste


def weights_for_occupancy(occupancy: float, threshold: float = 0.8,
                          normal: float = 0.7, dense: float = 0.3) -> float:
    """alpha for the current lot occupancy: favour distance normally, fit when the lot is nearly full."""
    return dense if occupancy >= threshold else normal


def assign(spots: List[Spot], v: Vehicle, entrance: Tuple[float, float],
           alpha: Optional[float] = None, **margins) -> Optional[Spot]:
    if alpha is None:
        occupancy = 1 - sum(p.free for p in spots) / len(spots)
        alpha = weights_for_occupancy(occupancy)

    cands = candidates(spots, v, **margins)
    if not cands:
        return None  # no spot fits: queue or reject

    d_max = max(math.dist(p.center, entrance) for p in spots)
    return min(cands, key=lambda p: cost(p, v, entrance, d_max, alpha))
