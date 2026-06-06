from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def close(self) -> None:
        self.handle.remove()

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient) -> None:
        self.gradients = gradient

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        input_tensor = input_tensor.detach().requires_grad_(True)
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()
        if self.gradients is None:
            raise RuntimeError("Grad-CAM target layer did not receive gradients.")

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def save_gradcam_example(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    classes: list[str],
    output_path: str | Path,
) -> None:
    target_layer = _get_default_target_layer(model)
    if target_layer is None:
        return

    images, labels = next(iter(loader))
    image_tensor = images[0:1].to(device)
    target_class = int(labels[0].item())
    extractor = GradCAM(model, target_layer)
    heatmap = extractor.generate(image_tensor, target_class)
    extractor.close()

    original = _denormalize(image_tensor[0].detach().cpu())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title(f"Original\nTrue: {classes[target_class]}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(original)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _get_default_target_layer(model: torch.nn.Module):
    if hasattr(model, "layer4"):
        return model.layer4[-1].conv2
    if hasattr(model, "features"):
        return model.features[-1]
    return None


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor * std + mean
    image = image.clamp(0, 1).numpy().transpose(1, 2, 0)
    return image
