from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_dir: str = "./tomato"
    output_dir: str = "./outputs/experiments"
    experiment_name: str = "resnet18_transfer_freeze_fc_then_layer4"
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    architecture: str = "resnet18"
    pretrained: bool = True
    freeze_strategy: str = "fc_then_layer4"
    stage1_epochs: int = 5
    stage2_epochs: int = 25
    stage1_lr: float = 1e-3
    stage2_lr: float = 1e-5
    weight_decay: float = 0.0
    val_split_from_val_dir: float = 0.5
    gradcam: bool = True

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name


EXPERIMENT_PRESETS: dict[str, ExperimentConfig] = {
    "resnet18_transfer_fc_then_layer4": ExperimentConfig(
        experiment_name="resnet18_transfer_fc_then_layer4",
        architecture="resnet18",
        pretrained=True,
        freeze_strategy="fc_then_layer4",
        stage1_lr=1e-3,
        stage2_lr=1e-5,
    ),
    "resnet18_scratch": ExperimentConfig(
        experiment_name="resnet18_scratch",
        architecture="resnet18",
        pretrained=False,
        freeze_strategy="none",
        stage1_epochs=0,
        stage2_epochs=10,
        stage2_lr=1e-3,
    ),
    "resnet18_transfer_fc_only": ExperimentConfig(
        experiment_name="resnet18_transfer_fc_only",
        architecture="resnet18",
        pretrained=True,
        freeze_strategy="fc_only",
        stage1_epochs=10,
        stage2_epochs=0,
        stage1_lr=1e-3,
    ),
    "resnet18_transfer_lr_low": ExperimentConfig(
        experiment_name="resnet18_transfer_lr_low",
        architecture="resnet18",
        pretrained=True,
        freeze_strategy="fc_then_layer4",
        stage1_lr=3e-4,
        stage2_lr=1e-5,
    ),
    "resnet50_transfer_fc_then_layer4": ExperimentConfig(
        experiment_name="resnet50_transfer_fc_then_layer4",
        architecture="resnet50",
        pretrained=True,
        freeze_strategy="fc_then_layer4",
    ),
    "mobilenet_v3_transfer_fc_then_features_tail": ExperimentConfig(
        experiment_name="mobilenet_v3_transfer_fc_then_features_tail",
        architecture="mobilenet_v3_small",
        pretrained=True,
        freeze_strategy="fc_then_layer4",
    ),
}
