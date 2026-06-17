import argparse
from pathlib import Path

import torch

from src.data import create_dataloaders
from src.evaluator import evaluate_model, measure_inference_speed
from src.models import build_model, count_parameters
from src.utils import get_device, load_json, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained tomato model.")
    parser.add_argument("--run-dir", required=True, help="Experiment directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / "config.json")
    set_seed(config["seed"])
    device = get_device()

    _, _, test_loader, classes = create_dataloaders(
        data_dir=config["data_dir"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        seed=config["seed"],
        num_workers=config["num_workers"],
        val_split_from_val_dir=config["val_split_from_val_dir"],
    )
    model = build_model(config["architecture"], len(classes), config["pretrained"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
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
    print(metrics)


if __name__ == "__main__":
    main()
