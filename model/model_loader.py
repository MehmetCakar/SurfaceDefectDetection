"""Model construction and checkpoint loading for the defect classifier."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CHECKPOINT = Path(__file__).parent / "saved_models" / "resnet18_neu.pt"


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(checkpoint: Path = DEFAULT_CHECKPOINT, device: str | None = None) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device).eval()
    return model


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
