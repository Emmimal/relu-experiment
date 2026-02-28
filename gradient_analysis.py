"""
gradient_analysis.py
--------------------
Detailed gradient norm analysis tracking how gradient magnitudes
evolve across layers and across training for both models.

Produces:
  - results/gradient_norms_evolution.png   (norms per layer over training)
  - results/gradient_ratio.png             (ReLU/Linear norm ratio per layer)
"""

import os
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import MLPWithReLU, MLPLinear
from utils  import init_kaiming, init_xavier, get_mnist_loaders, log_gradient_norms

EPOCHS      = 20
BATCH_SIZE  = 64
LR          = 1e-3
LOG_EVERY   = 1          # Log gradient norms every N epochs
RESULTS_DIR = "results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

LAYER_NAMES = ["fc1", "fc2", "fc3", "fc4", "fc5"]


def run_gradient_analysis():
    print(f"\n{'='*60}")
    print(f"  Gradient Norm Analysis: Layer-by-layer evolution")
    print(f"{'='*60}\n")

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    criterion       = nn.CrossEntropyLoss()

    model_relu   = init_kaiming(MLPWithReLU()).to(DEVICE)
    model_linear = init_xavier(MLPLinear()).to(DEVICE)

    opt_relu   = torch.optim.Adam(model_relu.parameters(),   lr=LR)
    opt_linear = torch.optim.Adam(model_linear.parameters(), lr=LR)

    # grad_log[model][epoch][layer] = norm value
    grad_log = {"relu": [], "linear": []}

    for epoch in range(1, EPOCHS + 1):
        relu_epoch_norms   = {l: [] for l in LAYER_NAMES}
        linear_epoch_norms = {l: [] for l in LAYER_NAMES}

        model_relu.train()
        model_linear.train()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # ReLU model
            opt_relu.zero_grad()
            loss = criterion(model_relu(images), labels)
            loss.backward()
            norms = log_gradient_norms(model_relu)
            for l in LAYER_NAMES:
                key = f"{l}.weight"
                if key in norms:
                    relu_epoch_norms[l].append(norms[key])
            opt_relu.step()

            # Linear model
            opt_linear.zero_grad()
            loss = criterion(model_linear(images), labels)
            loss.backward()
            norms = log_gradient_norms(model_linear)
            for l in LAYER_NAMES:
                key = f"{l}.weight"
                if key in norms:
                    linear_epoch_norms[l].append(norms[key])
            opt_linear.step()

        # Average over batches
        grad_log["relu"].append({l: np.mean(v) for l, v in relu_epoch_norms.items()})
        grad_log["linear"].append({l: np.mean(v) for l, v in linear_epoch_norms.items()})

        if epoch % 5 == 0 or epoch == 1:
            _print_norm_snapshot(epoch, grad_log)

    _plot_gradient_evolution(grad_log)
    _plot_gradient_ratio(grad_log)


def _print_norm_snapshot(epoch: int, grad_log: dict) -> None:
    relu_e   = grad_log["relu"][epoch - 1]
    linear_e = grad_log["linear"][epoch - 1]

    print(f"\n  Epoch {epoch} — Gradient Norms")
    print(f"  {'Layer':<8} {'ReLU':<14} {'Linear':<14} {'Ratio (ReLU/Lin)'}")
    print("  " + "-" * 50)
    for l in LAYER_NAMES:
        r = relu_e.get(l, 0)
        li = linear_e.get(l, 1e-10)
        ratio = r / li if li > 1e-10 else float('inf')
        print(f"  {l:<8} {r:<14.6f} {li:<14.6f} {ratio:.1f}x")


def _plot_gradient_evolution(grad_log: dict) -> None:
    epochs = range(1, EPOCHS + 1)
    n_layers = len(LAYER_NAMES)
    colors_relu   = plt.cm.Blues(np.linspace(0.4, 0.95, n_layers))
    colors_linear = plt.cm.Reds(np.linspace(0.4, 0.95, n_layers))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gradient Norm Evolution per Layer", fontsize=13, fontweight="bold")

    for ax, model_key, colors, title in [
        (axes[0], "relu",   colors_relu,   "ReLU Network"),
        (axes[1], "linear", colors_linear, "Linear Network"),
    ]:
        for i, layer in enumerate(LAYER_NAMES):
            vals = [grad_log[model_key][e][layer] for e in range(EPOCHS)]
            ax.plot(epochs, vals, color=colors[i], linewidth=2, label=layer)

        ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Gradient Norm")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gradient_norms_evolution.png"), dpi=150)
    plt.close()
    print(f"\n  [OK] Saved gradient_norms_evolution.png")


def _plot_gradient_ratio(grad_log: dict) -> None:
    """Plot ratio of ReLU gradient norm to Linear gradient norm per layer."""
    epochs = range(1, EPOCHS + 1)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(LAYER_NAMES)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, layer in enumerate(LAYER_NAMES):
        ratios = []
        for e in range(EPOCHS):
            r   = grad_log["relu"][e][layer]
            li  = grad_log["linear"][e].get(layer, 1e-10)
            ratios.append(r / li if li > 1e-10 else 0)
        ax.plot(epochs, ratios, color=colors[i], linewidth=2, label=layer)

    ax.axhline(y=1, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(1.5, 1.3, "Equal gradients", fontsize=9, color="#555")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Ratio (ReLU norm / Linear norm)")
    ax.set_title("How Much Larger Are ReLU Gradients vs Linear? (per layer)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gradient_ratio.png"), dpi=150)
    plt.close()
    print(f"  [OK] Saved gradient_ratio.png")


if __name__ == "__main__":
    run_gradient_analysis()
