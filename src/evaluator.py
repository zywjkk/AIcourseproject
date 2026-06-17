import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    classes: list[str],
    output_dir: str | Path,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds: list[int] = []
    all_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

    report_dict = classification_report(
        all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        all_labels, all_preds, target_names=classes, zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = float(report_dict["macro avg"]["f1-score"])
    weighted_f1 = float(report_dict["weighted avg"]["f1-score"])

    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    _save_classification_report_csv(report_dict, output_dir / "classification_report.csv")
    _plot_confusion_matrix(all_labels, all_preds, classes, output_dir / "confusion_matrix.png")

    metrics = {
        "top1_accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def plot_history(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    overfit_epoch = find_overfit_epoch(history["val_loss"])
    if overfit_epoch is not None:
        idx = overfit_epoch - 1
        plt.annotate(
            f"Overfit starts near epoch {overfit_epoch}",
            xy=(overfit_epoch, history["val_loss"][idx]),
            xytext=(overfit_epoch, history["val_loss"][idx] + 0.2),
            arrowprops={"facecolor": "red", "shrink": 0.05},
            color="red",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def find_overfit_epoch(val_losses: list[float], patience: int = 2) -> int | None:
    """Return first epoch where validation loss keeps rising after the best value."""
    if len(val_losses) < patience + 2:
        return None
    best_idx = int(np.argmin(val_losses))
    rising = 0
    for idx in range(best_idx + 1, len(val_losses)):
        if val_losses[idx] > val_losses[idx - 1]:
            rising += 1
            if rising >= patience:
                return idx - patience + 2
        else:
            rising = 0
    return None


def measure_inference_speed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 2,
    measure_batches: int = 10,
) -> float:
    model.eval()
    batch_times: list[float] = []
    with torch.no_grad():
        for index, (inputs, _) in enumerate(loader):
            inputs = inputs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= warmup_batches:
                batch_times.append(elapsed / inputs.size(0))
            if len(batch_times) >= measure_batches:
                break
    return float(np.mean(batch_times)) if batch_times else 0.0


def _save_classification_report_csv(report: dict, path: Path) -> None:
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "class": label,
                    "precision": values.get("precision", ""),
                    "recall": values.get("recall", ""),
                    "f1-score": values.get("f1-score", ""),
                    "support": values.get("support", ""),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "precision", "recall", "f1-score", "support"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion_matrix(
    labels: list[int], preds: list[int], classes: list[str], path: Path
) -> None:
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix on Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
