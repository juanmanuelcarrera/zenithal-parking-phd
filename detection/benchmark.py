"""Latency of each segmenter on CPU (load, warm-up, inference)."""
import argparse
import json
import statistics
import time
from pathlib import Path

import cv2


def benchmark(segmenter, images, warmup: int = 2):
    for img in images[:warmup]:
        segmenter(img)

    times = []
    for img in images:
        t = time.perf_counter()
        segmenter(img)
        times.append(time.perf_counter() - t)
    return {"n": len(times), "mean_s": statistics.mean(times), "std_s": statistics.pstdev(times),
            "fps": 1 / statistics.mean(times)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="folder with test images")
    parser.add_argument("--maskrcnn-weights")
    parser.add_argument("--yolo-weights", default="yolo11n-seg.pt")
    parser.add_argument("-o", "--output", default="latency.json")
    args = parser.parse_args()

    images = [cv2.imread(str(p)) for p in sorted(Path(args.images).glob("*.jpg"))]

    from .maskrcnn import MaskRCNNSegmenter
    from .yolo import YoloSegmenter

    t = time.perf_counter()
    yolo = YoloSegmenter(args.yolo_weights)
    yolo_init = time.perf_counter() - t

    t = time.perf_counter()
    maskrcnn = MaskRCNNSegmenter(args.maskrcnn_weights)
    maskrcnn_init = time.perf_counter() - t

    results = {
        "yolo": {"init_s": yolo_init, **benchmark(yolo, images)},
        "maskrcnn": {"init_s": maskrcnn_init, **benchmark(maskrcnn, images)},
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
