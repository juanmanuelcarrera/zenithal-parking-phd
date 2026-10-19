"""ZenithalParking loaders. Annotations are VIA polygons (one region per vehicle)."""
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def load_via(split_dir: str) -> List[Dict]:
    split_dir = Path(split_dir)
    regions = json.loads((split_dir / "via_region_data.json").read_text())

    samples = []
    for item in regions.values():
        polygons = []
        for region in item["regions"]:
            shape = region["shape_attributes"]
            polygons.append(np.stack([shape["all_points_x"], shape["all_points_y"]], axis=1))
        samples.append({"file": str(split_dir / item["filename"]), "polygons": polygons})
    return samples


def to_detectron2(split_dir: str) -> List[Dict]:
    from detectron2.structures import BoxMode

    records = []
    for idx, sample in enumerate(load_via(split_dir)):
        h, w = cv2.imread(sample["file"]).shape[:2]
        annotations = []
        for poly in sample["polygons"]:
            x0, y0 = poly.min(axis=0)
            x1, y1 = poly.max(axis=0)
            annotations.append({
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly.astype(float).ravel().tolist()],
                "category_id": 0,
            })
        records.append({"file_name": sample["file"], "image_id": idx,
                        "height": h, "width": w, "annotations": annotations})
    return records


def register_detectron2(root: str, splits=("train", "val")):
    from detectron2.data import DatasetCatalog, MetadataCatalog

    for split in splits:
        name = f"zenithal_{split}"
        if name not in DatasetCatalog.list():
            DatasetCatalog.register(name, lambda s=split: to_detectron2(f"{root}/{s}"))
            MetadataCatalog.get(name).set(thing_classes=["car"])


def export_yolo(root: str, out_dir: str, splits=("train", "val")) -> Path:
    """Write images/labels in YOLO-seg format plus a data.yaml."""
    out_dir = Path(out_dir)
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for sample in load_via(f"{root}/{split}"):
            image = cv2.imread(sample["file"])
            h, w = image.shape[:2]
            name = Path(sample["file"]).stem
            cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"), image)

            lines = []
            for poly in sample["polygons"]:
                norm = (poly / [w, h]).ravel()
                lines.append("0 " + " ".join(f"{v:.6f}" for v in norm))
            (out_dir / "labels" / split / f"{name}.txt").write_text("\n".join(lines))

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n")
    return data_yaml
