from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    val_split_from_val_dir: float = 0.5,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Expected dataset folders: tomato/train and tomato/val")

    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_test_dataset = datasets.ImageFolder(root=val_dir, transform=eval_transform)

    if train_dataset.classes != val_test_dataset.classes:
        raise ValueError("Train and validation folders have different class order.")

    val_len = int(len(val_test_dataset) * val_split_from_val_dir)
    test_len = len(val_test_dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    val_dataset, test_dataset = random_split(
        val_test_dataset, [val_len, test_len], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, train_dataset.classes
