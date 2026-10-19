"""Fine-tune YOLOv11-seg from COCO on ZenithalParking."""
import argparse

from ultralytics import YOLO

from .dataset import export_yolo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ZenithalParking")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_yaml = export_yolo(args.data, "runs/yolo_dataset")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=1024,
        optimizer="SGD",
        lr0=0.0025,
        momentum=0.9,
        patience=10,
        freeze=10,
        degrees=15,
        scale=0.2,
        hsv_v=0.3,
        device=args.device,
        project="runs",
        name="yolo",
    )


if __name__ == "__main__":
    main()
