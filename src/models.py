from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torchvision import models


def build_model(architecture: str, num_classes: int, pretrained: bool) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if architecture == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def set_trainable(model: nn.Module, strategy: str, stage: int) -> None:
    """Apply freezing strategy required by transfer-learning experiments."""
    for param in model.parameters():
        param.requires_grad = True

    strategy = strategy.lower()
    if strategy == "none":
        return

    if strategy == "fc_only":
        _freeze_all(model)
        _unfreeze_classifier(model)
        return

    if strategy == "fc_then_layer4":
        if stage == 1:
            _freeze_all(model)
            _unfreeze_classifier(model)
        else:
            _freeze_all(model)
            _unfreeze_tail(model)
        return

    raise ValueError(f"Unsupported freeze strategy: {strategy}")


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (param for param in model.parameters() if param.requires_grad)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def save_architecture_summary(model: nn.Module, path: str | Path, image_size: int) -> None:
    total, trainable = count_parameters(model)
    text = [
        f"Model: {model.__class__.__name__}",
        f"Input tensor: [B, 3, {image_size}, {image_size}]",
        f"Total parameters: {total:,}",
        f"Trainable parameters at summary time: {trainable:,}",
        "",
        "Tensor shape flow for ResNet-style backbone:",
        f"[B, 3, {image_size}, {image_size}]",
        "-> conv1/bn/relu: [B, 64, 112, 112]",
        "-> maxpool: [B, 64, 56, 56]",
        "-> layer1: [B, 64, 56, 56]",
        "-> layer2: [B, 128, 28, 28]",
        "-> layer3: [B, 256, 14, 14]",
        "-> layer4: [B, 512, 7, 7] for ResNet-18/34, [B, 2048, 7, 7] for ResNet-50",
        "-> global average pooling: [B, C]",
        "-> classifier: [B, 10]",
        "",
        str(model),
    ]
    Path(path).write_text("\n".join(text), encoding="utf-8")


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_classifier(model: nn.Module) -> None:
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def _unfreeze_tail(model: nn.Module) -> None:
    if hasattr(model, "layer4"):
        for param in model.layer4.parameters():
            param.requires_grad = True
        _unfreeze_classifier(model)
    elif hasattr(model, "features"):
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        _unfreeze_classifier(model)
