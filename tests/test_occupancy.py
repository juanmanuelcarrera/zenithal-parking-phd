import pytest

from occupancy import OccupancyState, OccupancyTracker, occupancy_ratios, state_from_ratio
from occupancy.occupancy import binary_state

SPOT = [(0, 0), (10, 0), (10, 20), (0, 20)]


def car(x0, x1):
    return [(x0, 0), (x1, 0), (x1, 20), (x0, 20)]


@pytest.mark.parametrize("ratio,state", [
    (0.0, OccupancyState.FREE),
    (0.09, OccupancyState.FREE),
    (0.10, OccupancyState.LOW),
    (0.49, OccupancyState.LOW),
    (0.50, OccupancyState.HIGH),
    (0.89, OccupancyState.HIGH),
    (0.90, OccupancyState.OCCUPIED),
])
def test_states(ratio, state):
    assert state_from_ratio(ratio) == state


def test_ratio_half():
    assert occupancy_ratios([SPOT], [car(5, 15)])[0] == pytest.approx(0.5)


def test_overlapping_vehicles_counted_once():
    assert occupancy_ratios([SPOT], [car(0, 10), car(0, 10)])[0] == pytest.approx(1.0)


def test_car_between_two_spots():
    # the case the binary detector misses: half a car on each spot
    left, right = SPOT, [(10, 0), (20, 0), (20, 20), (10, 20)]
    ratios = occupancy_ratios([left, right], [car(5, 15)])
    assert [state_from_ratio(r) for r in ratios] == [OccupancyState.HIGH] * 2
    assert [binary_state(r) for r in ratios] == [OccupancyState.FREE] * 2


def test_hysteresis_needs_k_frames():
    tracker = OccupancyTracker({"P-01": SPOT}, k=3)
    assert tracker.update([car(0, 10)]) == {}
    assert tracker.update([car(0, 10)]) == {}
    assert tracker.update([car(0, 10)]) == {"P-01": OccupancyState.OCCUPIED}


def test_hysteresis_ignores_flicker():
    tracker = OccupancyTracker({"P-01": SPOT}, k=3)
    for vehicles in ([car(0, 10)], [], [car(0, 10)], []):
        tracker.update(vehicles)
    assert tracker.state["P-01"] == OccupancyState.FREE
