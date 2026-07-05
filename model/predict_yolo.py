"""YOLO-based defect detection: locate defects with bounding boxes.

Usage:
    python model/predict_yolo.py path/to/image.jpg
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

YOLO_CHECKPOINT = (
    Path(__file__).parent / "saved_models" / "yolov8n_neu" / "weights" / "best.pt"
)

_model = None


def load_yolo(checkpoint: Path = YOLO_CHECKPOINT):
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(checkpoint)
    return _model


def detect_defects(
    image: Image.Image, conf: float = 0.25, checkpoint: Path = YOLO_CHECKPOINT
) -> dict:
    """Run detection on a PIL image.

    Returns a dict with the annotated image (PIL) and a list of detections:
    [{"label": str, "confidence": float, "box": [x1, y1, x2, y2]}, ...]
    """
    model = load_yolo(checkpoint)
    result = model.predict(np.array(image.convert("RGB")), conf=conf, verbose=False)[0]

    detections = [
        {
            "label": result.names[int(cls)],
            "confidence": float(c),
            "box": [round(v) for v in xyxy.tolist()],
        }
        for cls, c, xyxy in zip(
            result.boxes.cls, result.boxes.conf, result.boxes.xyxy
        )
    ]
    annotated = Image.fromarray(result.plot()[..., ::-1])  # BGR -> RGB
    return {"annotated_image": annotated, "detections": detections}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, help="save annotated image here")
    args = parser.parse_args()

    result = detect_defects(Image.open(args.image), conf=args.conf)
    if not result["detections"]:
        print("No defects detected above confidence threshold.")
    for d in result["detections"]:
        print(f"{d['label']:>16} {d['confidence']:.1%} box={d['box']}")
    if args.out:
        result["annotated_image"].save(args.out)
        print(f"Annotated image saved to {args.out}")


if __name__ == "__main__":
    main()
