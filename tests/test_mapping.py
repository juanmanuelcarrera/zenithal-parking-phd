import cv2
import numpy as np

from mapping import ParkingSpaceDetector


def synthetic_lot():
    img = np.full((400, 600, 3), 60, np.uint8)
    for x in range(100, 501, 100):
        cv2.line(img, (x, 100), (x, 300), (255, 255, 255), 6)
    cv2.line(img, (100, 100), (500, 100), (255, 255, 255), 6)
    cv2.line(img, (100, 300), (500, 300), (255, 255, 255), 6)
    return img


def test_detector_returns_spaces_and_timings():
    result = ParkingSpaceDetector().detect(synthetic_lot())
    assert "total" in result["timings_ms"]
    for space in result["parking_spaces"]:
        assert len(space["vertices"]) >= 4
        assert space["area"] > 0
