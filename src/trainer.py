from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .models import set_trainable, trainable_parameters


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    freeze_strategy: str,
    stage1_epochs: int,
    stage2_epochs: int,
    stage1_lr: float,
    stage2_lr: float,
    weight_decay: float,
    best_model_path: str | Path,
) -> tuple[nn.Module, dict[str, list[float]]]:
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = -1.0
    best_weights = deepcopy(model.state_dict())

    if stage1_epochs > 0:
        set_trainable(model, freeze_strategy, stage=1)
        optimizer = optim.Adam(
            trainable_parameters(model), lr=stage1_lr, weight_decay=weight_decay
        )
        best_weights, best_acc = _fit_epochs(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            stage1_epochs,
            history,
            best_acc,
            best_weights,
            stage_name="stage1",
        )

    if stage2_epochs > 0:
        model.load_state_dict(best_weights)
        set_trainable(model, freeze_strategy, stage=2)
        optimizer = optim.Adam(
            trainable_parameters(model), lr=stage2_lr, weight_decay=weight_decay
        )
        best_weights, best_acc = _fit_epochs(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            stage2_epochs,
            history,
            best_acc,
            best_weights,
            stage_name="stage2",
        )

    model.load_state_dict(best_weights)
    best_model_path = Path(best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best_model_path)
    return model, history


def _fit_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    history: dict[str, list[float]],
    best_acc: float,
    best_weights: dict[str, torch.Tensor],
    stage_name: str,
) -> tuple[dict[str, torch.Tensor], float]:
    for epoch in range(epochs):
        train_loss, train_acc = _run_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_one_epoch(
            model, val_loader, criterion, optimizer=None, device=device, train=False
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"{stage_name} epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = deepcopy(model.state_dict())

    return best_weights, best_acc


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if train:
                assert optimizer is not None
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += inputs.size(0)

    return total_loss / total_count, total_correct / total_count
