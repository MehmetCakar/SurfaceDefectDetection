"""Train a YOLOv8-nano defect detector on NEU-DET.

Usage:
    python model/train_yolo.py --epochs 40
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(ROOT / "data" / "yolo" / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "model" / "saved_models"),
        name="yolov8n_neu",
        exist_ok=True,
    )
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
