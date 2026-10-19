import argparse
import json

import cv2
import numpy as np

from .detector import ParkingSpaceDetector


def main():
    parser = argparse.ArgumentParser(description="Detect parking spaces in an empty-lot image")
    parser.add_argument("image")
    parser.add_argument("-o", "--output", default="parking_spaces.json")
    parser.add_argument("--preview", help="save an image with the detected polygons")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    result = ParkingSpaceDetector().detect(image)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    if args.preview:
        for space in result["parking_spaces"]:
            pts = np.array(space["vertices"], dtype=np.int32)
            cv2.polylines(image, [pts], True, (0, 0, 255), 2)
        cv2.imwrite(args.preview, image)

    print(f"{len(result['parking_spaces'])} spaces, {result['timings_ms']['total']} ms")


if __name__ == "__main__":
    main()
