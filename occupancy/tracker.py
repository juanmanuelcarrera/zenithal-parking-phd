"""Temporal hysteresis: a spot changes state only after k consecutive frames agree."""
from typing import Dict, List

from .occupancy import OccupancyState, occupancy_ratios, state_from_ratio


class OccupancyTracker:
    def __init__(self, spots: Dict[str, list], k: int = 3):
        self.spots = spots
        self.k = k
        self.state = {sid: OccupancyState.FREE for sid in spots}
        self.ratio = {sid: 0.0 for sid in spots}
        self._candidate = {sid: (OccupancyState.FREE, 0) for sid in spots}

    def update(self, vehicles: List) -> Dict[str, OccupancyState]:
        """Feed the vehicle polygons of a new frame. Returns spots whose state changed."""
        ids = list(self.spots)
        ratios = occupancy_ratios([self.spots[i] for i in ids], vehicles)

        changed = {}
        for sid, ratio in zip(ids, ratios):
            self.ratio[sid] = ratio
            new = state_from_ratio(ratio)
            if new == self.state[sid]:
                self._candidate[sid] = (new, 0)
                continue

            cand, count = self._candidate[sid]
            count = count + 1 if cand == new else 1
            self._candidate[sid] = (new, count)
            if count >= self.k:
                self.state[sid] = new
                self._candidate[sid] = (new, 0)
                changed[sid] = new
        return changed
