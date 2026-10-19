from assignment import Spot, Vehicle, assign, candidates

ENTRANCE = (0.0, 0.0)


def lot():
    return [
        Spot("near_small", 4.5, 2.2, (5, 0)),
        Spot("near_big", 5.5, 2.6, (6, 0)),
        Spot("far_fit", 5.0, 2.3, (40, 0)),
        Spot("taken", 5.0, 2.5, (2, 0), free=False),
    ]


def test_candidates_filter_size_and_state():
    ids = [p.id for p in candidates(lot(), Vehicle(4.5, 1.9))]
    assert ids == ["near_big", "far_fit"]


def test_distance_wins_with_high_alpha():
    assert assign(lot(), Vehicle(4.5, 1.9), ENTRANCE, alpha=0.9).id == "near_big"


def test_fit_wins_with_low_alpha():
    assert assign(lot(), Vehicle(4.5, 1.9), ENTRANCE, alpha=0.1).id == "far_fit"


def test_no_spot_fits():
    assert assign(lot(), Vehicle(6.0, 2.5), ENTRANCE) is None
