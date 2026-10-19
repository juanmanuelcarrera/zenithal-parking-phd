import pytest

from measurement import Calibration, polygon_area, vehicle_dimensions


def test_shoelace_rectangle():
    assert polygon_area([(0, 0), (4, 0), (4, 2), (0, 2)]) == 8


def test_shoelace_ignores_orientation():
    assert polygon_area([(0, 0), (0, 2), (4, 2), (4, 0)]) == 8


def test_gamma_from_reference():
    cal = Calibration.from_reference((0, 0), (100, 0), real_distance_m=5.0)
    assert cal.gamma == pytest.approx(0.05)
    assert cal.area(400) == pytest.approx(1.0)  # gamma^2 * A_img


def test_vehicle_dimensions_rotated():
    length, width = vehicle_dimensions([(0, 0), (10, 10), (8, 12), (-2, 2)])
    assert length > width
