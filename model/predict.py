"""Single-image prediction for the defect classifier.

Usage:
    python model/predict.py path/to/image.jpg
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from model_loader import CLASSES, DEFAULT_CHECKPOINT, eval_transform, load_model


def predict_image(
    image: Image.Image, model=None, device: str | None = None, top_k: int = 3
) -> dict:
    """Return predicted class, confidence and top-k predictions for a PIL image."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(device=device)

    tensor = eval_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    class_names = sorted(CLASSES)
    top = torch.topk(probs, k=min(top_k, len(class_names)))
    top_predictions = [
        {"label": class_names[i], "confidence": p.item()}
        for p, i in zip(top.values, top.indices)
    ]
    return {
        "label": top_predictions[0]["label"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    result = predict_image(Image.open(args.image), model=model)
    print(f"Prediction: {result['label']} ({result['confidence']:.1%})")
    for p in result["top_predictions"]:
        print(f"  {p['label']:>16}: {p['confidence']:.1%}")


if __name__ == "__main__":
    main()
