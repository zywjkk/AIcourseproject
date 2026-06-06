import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument(
        "--summary-csv",
        default="outputs/experiments/experiment_summary.csv",
        help="Path to experiment_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments",
        help="Directory for comparison tables and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    df = df.drop_duplicates(subset=["experiment"], keep="last")
    df = df.sort_values("top1_accuracy", ascending=False)
    df.to_csv(output_dir / "experiment_summary_dedup.csv", index=False, encoding="utf-8")

    _plot_metric(
        df,
        metric="top1_accuracy",
        title="Top-1 Accuracy Comparison",
        ylabel="Top-1 Accuracy",
        path=output_dir / "accuracy_comparison.png",
    )
    _plot_metric(
        df,
        metric="macro_f1",
        title="Macro F1 Comparison",
        ylabel="Macro F1",
        path=output_dir / "macro_f1_comparison.png",
    )
    _plot_metric(
        df,
        metric="total_parameters",
        title="Model Parameter Comparison",
        ylabel="Parameters",
        path=output_dir / "parameter_comparison.png",
        scale=1_000_000,
        scale_label="M",
    )
    _plot_metric(
        df,
        metric="inference_seconds_per_image",
        title="Inference Time Comparison",
        ylabel="Seconds per Image",
        path=output_dir / "inference_time_comparison.png",
    )

    print(df.to_string(index=False))


def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    scale: float = 1.0,
    scale_label: str = "",
) -> None:
    values = df[metric] / scale
    labels = df["experiment"].str.replace("_", "\n")
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(f"{ylabel} ({scale_label})" if scale_label else ylabel)
    plt.xticks(rotation=0, ha="center", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        if metric in {"top1_accuracy", "macro_f1"}:
            label = f"{value:.3f}"
        elif scale_label:
            label = f"{value:.2f}{scale_label}"
        else:
            label = f"{value:.6f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
