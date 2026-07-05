"""Train a ResNet18 defect classifier on the processed NEU dataset.

Usage:
    python model/train.py --epochs 15 --batch-size 32
"""

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from model_loader import CLASSES, build_model, eval_transform, train_transform


def make_loader(data_dir: Path, split: str, batch_size: int, train: bool) -> DataLoader:
    tf = train_transform() if train else eval_transform()
    ds = datasets.ImageFolder(data_dir / split, transform=tf)
    if ds.classes != sorted(CLASSES):
        raise SystemExit(
            f"Class folders {ds.classes} don't match expected {sorted(CLASSES)}"
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, list, list]:
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds += preds.cpu().tolist()
        all_labels += labels.cpu().tolist()
    return correct / total, all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--out", type=Path, default=Path("model/saved_models/resnet18_neu.pt")
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    train_loader = make_loader(args.data_dir, "train", args.batch_size, train=True)
    val_loader = make_loader(args.data_dir, "val", args.batch_size, train=False)
    test_loader = make_loader(args.data_dir, "test", args.batch_size, train=False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, start = 0.0, time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        train_loss = epoch_loss / len(train_loader.dataset)
        val_acc, _, _ = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss {train_loss:.4f} | "
            f"val_acc {val_acc:.4f} | {time.time() - start:.0f}s"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": CLASSES}, args.out)
            print(f"  -> saved checkpoint (val_acc {val_acc:.4f})")

    # Final evaluation on the held-out test set with the best checkpoint
    state = torch.load(args.out, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    test_acc, preds, labels = evaluate(model, test_loader, device)
    class_names = sorted(CLASSES)
    print(f"\nTest accuracy: {test_acc:.4f}\n")
    print(classification_report(labels, preds, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "test_accuracy": test_acc,
                "best_val_accuracy": best_val_acc,
                "report": classification_report(
                    labels, preds, target_names=class_names, output_dict=True
                ),
            },
            indent=2,
        )
    )
    print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    main()
