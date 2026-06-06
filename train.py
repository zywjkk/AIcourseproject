import argparse
import csv
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.config import EXPERIMENT_PRESETS, ExperimentConfig
from src.data import create_dataloaders
from src.evaluator import evaluate_model, measure_inference_speed, plot_history
from src.gradcam import save_gradcam_example
from src.models import build_model, count_parameters, save_architecture_summary
from src.trainer import run_training
from src.utils import configure_utf8_console, ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tomato disease classifiers.")
    parser.add_argument(
        "--preset",
        default="resnet18_transfer_fc_then_layer4",
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        help="Experiment preset.",
    )
    parser.add_argument("--data-dir", default=None, help="Dataset directory.")
    parser.add_argument("--output-dir", default=None, help="Experiment output root.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs.")
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    config = EXPERIMENT_PRESETS[args.preset]
    if args.data_dir:
        config = replace(config, data_dir=args.data_dir)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    if args.epochs is not None:
        config = replace(config, stage1_epochs=max(args.epochs // 2, 0), stage2_epochs=args.epochs)

    run_experiment(config)


def run_experiment(config: ExperimentConfig) -> dict[str, float]:
    set_seed(config.seed)
    device = get_device()
    run_dir = ensure_dir(config.run_dir)
    save_json(asdict(config), run_dir / "config.json")

    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, classes = create_dataloaders(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        val_split_from_val_dir=config.val_split_from_val_dir,
    )
    save_json(
        {
            "classes": classes,
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
        },
        run_dir / "dataset_summary.json",
    )

    model = build_model(config.architecture, len(classes), config.pretrained).to(device)
    total_params, trainable_params = count_parameters(model)
    save_json(
        {"total_parameters": total_params, "trainable_parameters_initial": trainable_params},
        run_dir / "model_parameters.json",
    )
    save_architecture_summary(model, run_dir / "model_architecture.txt", config.image_size)

    model, history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        freeze_strategy=config.freeze_strategy,
        stage1_epochs=config.stage1_epochs,
        stage2_epochs=config.stage2_epochs,
        stage1_lr=config.stage1_lr,
        stage2_lr=config.stage2_lr,
        weight_decay=config.weight_decay,
        best_model_path=run_dir / "best_model.pth",
    )
    save_json(history, run_dir / "history.json")
    plot_history(history, run_dir / "training_curves.png")

    metrics = evaluate_model(model, test_loader, device, classes, run_dir)
    speed = measure_inference_speed(model, test_loader, device)
    total_params, trainable_params = count_parameters(model)
    metrics.update(
        {
            "inference_seconds_per_image": speed,
            "total_parameters": total_params,
            "trainable_parameters_final": trainable_params,
        }
    )
    save_json(metrics, run_dir / "metrics.json")
    _append_summary(config, metrics, Path(config.output_dir) / "experiment_summary.csv")

    if config.gradcam:
        save_gradcam_example(model, test_loader, device, classes, run_dir / "grad_cam.png")

    print("Done. Key metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics


def _append_summary(config: ExperimentConfig, metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "experiment": config.experiment_name,
        "architecture": config.architecture,
        "pretrained": config.pretrained,
        "freeze_strategy": config.freeze_strategy,
        "stage1_lr": config.stage1_lr,
        "stage2_lr": config.stage2_lr,
        **metrics,
    }
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
