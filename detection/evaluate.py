"""
Per-instance area precision on ZenithalParking.

precision = 100 - |A_det - A_gt| / A_gt * 100, matching each ground-truth
polygon with the detection of highest IoU.
"""
import argparse
import csv
import statistics

import cv2
import numpy as np
from shapely.geometry import Polygon

from .dataset import load_via


def iou(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = Polygon(a).buffer(0), Polygon(b).buffer(0)
    union = pa.union(pb).area
    return pa.intersection(pb).area / union if union else 0.0


def area_precision(a_det: float, a_gt: float) -> float:
    return 100.0 - abs(a_det - a_gt) / a_gt * 100.0


def evaluate(segmenter, split_dir: str, min_iou: float = 0.5):
    rows = []
    for sample in load_via(split_dir):
        detections = segmenter(cv2.imread(sample["file"]))
        for gt in sample["polygons"]:
            scored = [(iou(gt, d.polygon), d) for d in detections if len(d.polygon) >= 3]
            best_iou, best = max(scored, key=lambda x: x[0], default=(0.0, None))
            a_gt = Polygon(gt).area
            if best is None or best_iou < min_iou:
                rows.append((sample["file"], a_gt, 0.0, 0.0, best_iou))
                continue
            rows.append((sample["file"], a_gt, best.area, area_precision(best.area, a_gt), best_iou))
    return rows


def summary(rows):
    p = [r[3] for r in rows if r[2] > 0]
    return {
        "instances": len(rows),
        "detected": len(p),
        "mean": statistics.mean(p),
        "median": statistics.median(p),
        "std": statistics.stdev(p),
        "min": min(p),
        "max": max(p),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir")
    parser.add_argument("--model", choices=["maskrcnn", "yolo"], default="maskrcnn")
    parser.add_argument("--weights")
    parser.add_argument("--csv", default="area_precision.csv")
    args = parser.parse_args()

    if args.model == "maskrcnn":
        from .maskrcnn import MaskRCNNSegmenter
        segmenter = MaskRCNNSegmenter(args.weights)
    else:
        from .yolo import YoloSegmenter
        segmenter = YoloSegmenter(args.weights or "yolo11n-seg.pt")

    rows = evaluate(segmenter, args.split_dir)
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "area_gt", "area_det", "precision", "iou"])
        writer.writerows(rows)

    for k, v in summary(rows).items():
        print(f"{k:>10}: {v:.2f}" if isinstance(v, float) else f"{k:>10}: {v}")


if __name__ == "__main__":
    main()
