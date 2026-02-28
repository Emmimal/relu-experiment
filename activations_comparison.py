"""
activations_comparison.py
--------------------------
Trains 6 variants of the same MLP on MNIST, each with a different
activation function. Matches the comparison table in the article:
  None (linear), Sigmoid, Tanh, ReLU, Leaky ReLU, GELU

Produces:
  - results/activations_comparison.png
  - results/activations_comparison_results.txt
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import MLPWithActivation
from utils  import init_for_activation, get_mnist_loaders, train_one_epoch, evaluate

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ACTIVATIONS = ["none", "sigmoid", "tanh", "relu", "leaky_relu", "gelu"]
EPOCHS      = 20
BATCH_SIZE  = 64
LR          = 1e-3
NUM_SEEDS   = 3
RESULTS_DIR = "results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISPLAY_NAMES = {
    "none":        "None (Linear)",
    "sigmoid":     "Sigmoid",
    "tanh":        "Tanh",
    "relu":        "ReLU",
    "leaky_relu":  "Leaky ReLU",
    "gelu":        "GELU",
}

COLORS = {
    "none":        "#9E9E9E",
    "sigmoid":     "#FF9800",
    "tanh":        "#FFEB3B",
    "relu":        "#2196F3",
    "leaky_relu":  "#4CAF50",
    "gelu":        "#9C27B0",
}

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_activation_comparison():
    print(f"\n{'='*60}")
    print(f"  Activation Function Comparison on MNIST")
    print(f"  Activations: {ACTIVATIONS}")
    print(f"{'='*60}\n")

    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()

    # Store full training history (loss + accuracy per epoch) per activation
    histories = {}
    final_results = {}

    for activation in ACTIVATIONS:
        print(f"  Training with activation: {DISPLAY_NAMES[activation]}")
        seed_test_accs = []
        seed_histories = []

        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            model = init_for_activation(
                MLPWithActivation(activation=activation), activation
            ).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            epoch_losses = []
            epoch_accs   = []
            for epoch in range(1, EPOCHS + 1):
                loss, acc, _ = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
                epoch_losses.append(loss)
                epoch_accs.append(acc)

            _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
            seed_test_accs.append(test_acc)
            seed_histories.append({"loss": epoch_losses, "acc": epoch_accs})
            print(f"    Seed {seed}: Test accuracy = {test_acc:.2f}%")

        mean_acc = np.mean(seed_test_accs)
        std_acc  = np.std(seed_test_accs)
        final_results[activation] = (mean_acc, std_acc)
        histories[activation] = seed_histories
        print(f"  → Mean: {mean_acc:.2f}% ± {std_acc:.2f}%\n")

    # ── Print summary table ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {'Activation':<15} {'Test Acc (mean ± std)'}")
    print("  " + "─" * 40)
    for act in ACTIVATIONS:
        mean, std = final_results[act]
        print(f"  {DISPLAY_NAMES[act]:<15} {mean:.2f}% ± {std:.2f}%")

    # ── Save text results ─────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "activations_comparison_results.txt"), "w") as f:
        f.write("Activation Function Comparison Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"{'Activation':<15} {'Mean Test Acc':<20} {'Std Dev'}\n")
        f.write("─"*45 + "\n")
        for act in ACTIVATIONS:
            mean, std = final_results[act]
            f.write(f"{DISPLAY_NAMES[act]:<15} {mean:.2f}%               ±{std:.2f}%\n")

    # ── Plot learning curves ──────────────────────────────────────
    _plot_learning_curves(histories)
    _plot_final_accuracy_bar(final_results)


def _plot_learning_curves(histories: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Activation Function Comparison — MNIST Training", fontsize=13, fontweight="bold")
    epochs = range(1, EPOCHS + 1)

    for activation in ACTIVATIONS:
        seed_histories = histories[activation]
        color = COLORS[activation]
        label = DISPLAY_NAMES[activation]

        # Average across seeds
        mean_losses = np.mean([[h["loss"][e] for h in seed_histories] for e in range(EPOCHS)], axis=1)
        mean_accs   = np.mean([[h["acc"][e]  for h in seed_histories] for e in range(EPOCHS)], axis=1)

        axes[0].plot(epochs, mean_losses, color=color, linewidth=2, label=label)
        axes[1].plot(epochs, mean_accs,   color=color, linewidth=2, label=label)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Loss by Activation"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Train Accuracy (%)")
    axes[1].set_title("Accuracy by Activation"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "activations_learning_curves.png"), dpi=150)
    plt.close()
    print(f"\n  ✓ Saved activations_learning_curves.png")


def _plot_final_accuracy_bar(final_results: dict) -> None:
    means  = [final_results[a][0] for a in ACTIVATIONS]
    stds   = [final_results[a][1] for a in ACTIVATIONS]
    labels = [DISPLAY_NAMES[a]    for a in ACTIVATIONS]
    colors = [COLORS[a]           for a in ACTIVATIONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5,
                  alpha=0.85, edgecolor="white", linewidth=1.2)

    # Add value labels on top of bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean + std + 0.1,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Final Test Accuracy by Activation Function (MNIST, 20 epochs)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(88, 100)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "activations_comparison.png"), dpi=150)
    plt.close()
    print(f"  ✓ Saved activations_comparison.png")


if __name__ == "__main__":
    run_activation_comparison()
