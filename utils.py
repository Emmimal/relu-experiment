"""
utils.py
--------
Shared utilities: weight initialization, data loading,
training/evaluation loops, and gradient norm logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Tuple
import numpy as np


# ---------------------------------------------
# Weight Initialization
# ---------------------------------------------

def init_kaiming(model: nn.Module) -> nn.Module:
    """
    Kaiming (He) initialization — optimal for ReLU activations.
    Accounts for the fact that ReLU kills half the inputs on average.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
    return model


def init_xavier(model: nn.Module) -> nn.Module:
    """
    Xavier (Glorot) initialization — appropriate for linear or tanh activations.
    Keeps variance stable across layers when no activation is applied.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def init_for_activation(model: nn.Module, activation: str) -> nn.Module:
    """Auto-select initialization based on activation type."""
    if activation in ("relu", "leaky_relu"):
        return init_kaiming(model)
    return init_xavier(model)


# ---------------------------------------------
# Data Loading
# ---------------------------------------------

def get_mnist_loaders(
    batch_size: int = 64,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for MNIST.
    Normalizes using dataset mean (0.1307) and std (0.3081).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------
# Training & Evaluation
# ---------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_grad_norms: bool = False
) -> Tuple[float, float, Dict[str, float]]:
    """
    Runs one full training epoch.
    Returns: (avg_loss, accuracy_percent, grad_norms_dict)
    grad_norms_dict is empty if log_grad_norms=False.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    grad_norms: Dict[str, List[float]] = {}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if log_grad_norms:
            for name, param in model.named_parameters():
                if param.grad is not None and 'weight' in name:
                    grad_norms.setdefault(name, []).append(param.grad.norm().item())

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    # Average gradient norms across batches
    avg_grad_norms = {k: float(np.mean(v)) for k, v in grad_norms.items()}
    return avg_loss, accuracy, avg_grad_norms


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on a DataLoader. Returns (avg_loss, accuracy_percent)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------
# Gradient Norm Snapshot (single backward pass)
# ---------------------------------------------

def log_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Call immediately after loss.backward() to capture per-layer gradient norms.
    Only captures weight gradients (ignores biases for clarity).
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            norms[name] = param.grad.norm().item()
    return norms


def print_gradient_norms(norms: Dict[str, float], model_name: str = "") -> None:
    """Pretty-print gradient norm table."""
    header = f"\n{'-'*50}\nGradient Norms — {model_name}\n{'-'*50}"
    print(header)
    for layer, norm in norms.items():
        bar = "|" * min(int(norm * 5000), 40)
        print(f"  {layer:<25} {norm:.6f}  {bar}")
    print()
