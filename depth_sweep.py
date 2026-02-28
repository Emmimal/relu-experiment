"""
depth_sweep.py
--------------
Tests whether adding depth to a *linear* network improves accuracy.
Spoiler: it doesn't — and actually makes things slightly worse.
Mirrors the depth sweep table in the article.

Produces:
  - results/depth_sweep.png
  - results/depth_sweep_results.txt
"""

import os
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import DeepLinearMLP, DeepReLUMLP
from utils  import init_xavier, init_kaiming, get_mnist_loaders, train_one_epoch, evaluate

# ---------------------------------------------
# Config
# ---------------------------------------------
DEPTHS      = [1, 2, 3, 5, 10]
HIDDEN_SIZE = 256
EPOCHS      = 15          # Enough to converge; linear models plateau fast
BATCH_SIZE  = 64
LR          = 1e-3
NUM_SEEDS   = 3           # Run each config multiple seeds for reliable numbers
RESULTS_DIR = "results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_depth_sweep():
    print(f"\n{'='*60}")
    print(f"  Depth Sweep: Linear vs ReLU Networks")
    print(f"  Depths tested: {DEPTHS}")
    print(f"  Seeds per config: {NUM_SEEDS}")
    print(f"{'='*60}\n")

    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()

    linear_results = {}   # depth -> list of test accuracies across seeds
    relu_results   = {}

    for depth in DEPTHS:
        linear_accs = []
        relu_accs   = []

        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)

            # -- Linear model -------------------------------------
            model_lin = init_xavier(DeepLinearMLP(depth, HIDDEN_SIZE)).to(DEVICE)
            opt_lin   = torch.optim.Adam(model_lin.parameters(), lr=LR)

            for _ in range(EPOCHS):
                train_one_epoch(model_lin, train_loader, opt_lin, criterion, DEVICE)
            _, lin_acc = evaluate(model_lin, test_loader, criterion, DEVICE)
            linear_accs.append(lin_acc)

            # -- ReLU model ---------------------------------------
            model_relu = init_kaiming(DeepReLUMLP(depth, HIDDEN_SIZE)).to(DEVICE)
            opt_relu   = torch.optim.Adam(model_relu.parameters(), lr=LR)

            for _ in range(EPOCHS):
                train_one_epoch(model_relu, train_loader, opt_relu, criterion, DEVICE)
            _, relu_acc = evaluate(model_relu, test_loader, criterion, DEVICE)
            relu_accs.append(relu_acc)

            print(f"  Depth {depth:>2} | Seed {seed} | Linear: {lin_acc:.2f}%  ReLU: {relu_acc:.2f}%")

        linear_results[depth] = linear_accs
        relu_results[depth]   = relu_accs

    # -- Print Summary Table ---------------------------------------
    print(f"\n{'='*60}")
    print(f"  {'Depth':<8} {'Linear Acc (mean+/-std)':<25} {'ReLU Acc (mean+/-std)':<25}")
    print("  " + "-" * 55)

    lines = []
    for depth in DEPTHS:
        l_mean = np.mean(linear_results[depth])
        l_std  = np.std(linear_results[depth])
        r_mean = np.mean(relu_results[depth])
        r_std  = np.std(relu_results[depth])
        line = f"  {depth:<8} {l_mean:.2f}% +/- {l_std:.2f}%          {r_mean:.2f}% +/- {r_std:.2f}%"
        print(line)
        lines.append(line)

    # -- Save text -------------------------------------------------
    with open(os.path.join(RESULTS_DIR, "depth_sweep_results.txt"), "w", encoding="utf-8") as f:
        f.write("Depth Sweep Results\n")
        f.write("="*60 + "\n")
        f.write(f"{'Depth':<8} {'Linear Acc (mean+/-std)':<28} {'ReLU Acc (mean+/-std)'}\n")
        f.write("-"*60 + "\n")
        for line in lines:
            f.write(line.strip() + "\n")

    # -- Plot ------------------------------------------------------
    _plot_depth_sweep(DEPTHS, linear_results, relu_results)


def _plot_depth_sweep(depths, linear_results, relu_results):
    lin_means = [np.mean(linear_results[d]) for d in depths]
    lin_stds  = [np.std(linear_results[d])  for d in depths]
    rel_means = [np.mean(relu_results[d])   for d in depths]
    rel_stds  = [np.std(relu_results[d])    for d in depths]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(depths, lin_means, yerr=lin_stds,
                color="#F44336", linewidth=2, marker="o", markersize=7,
                capsize=5, label="Linear (no activation)", zorder=3)
    ax.errorbar(depths, rel_means, yerr=rel_stds,
                color="#2196F3", linewidth=2, marker="s", markersize=7,
                capsize=5, label="ReLU", zorder=3)

    # Annotate the key insight
    ax.annotate("Linear model gets\nworse with more depth",
                xy=(depths[-1], lin_means[-1]),
                xytext=(depths[-2], lin_means[-1] - 0.6),
                arrowprops=dict(arrowstyle="->", color="#B71C1C"),
                color="#B71C1C", fontsize=9)

    ax.set_xlabel("Network Depth (number of hidden layers)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Depth vs Accuracy: Linear Network Ceiling", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Show logistic regression baseline
    ax.axhline(y=92.0, color="#F44336", linewidth=0.8, linestyle=":",
               alpha=0.5, label="Logistic Regression baseline")
    ax.text(depths[0] + 0.1, 92.3, "Logistic regression baseline ~ 92%",
            fontsize=8, color="#888")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "depth_sweep.png"), dpi=150)
    plt.close()
    print(f"\n  [OK] Saved depth_sweep.png")


if __name__ == "__main__":
    run_depth_sweep()
