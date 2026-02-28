"""
two_moons.py
------------
Trains ReLU and linear MLPs on the two-moons synthetic dataset,
then plots their decision boundaries side by side.

This is the visualization that makes the difference concrete:
- Linear model → straight line (fails on non-convex data)
- ReLU model   → curved boundary (wraps around each crescent)

Produces:
  - results/decision_boundary.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import TwoD_ReLU, TwoD_Linear
from utils  import init_kaiming, init_xavier

RESULTS_DIR = "results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_two_moons():
    print(f"\n{'='*60}")
    print(f"  Two-Moons Decision Boundary Experiment")
    print(f"{'='*60}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Data ──────────────────────────────────────────────────────
    X, y = make_moons(n_samples=1200, noise=0.2, random_state=SEED)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    X_test_t  = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t  = torch.LongTensor(y_test).to(DEVICE)

    # ── Train models ──────────────────────────────────────────────
    model_relu   = init_kaiming(TwoD_ReLU()).to(DEVICE)
    model_linear = init_xavier(TwoD_Linear()).to(DEVICE)
    criterion    = nn.CrossEntropyLoss()

    print("  Training ReLU model on two-moons...")
    relu_acc   = _train_2d(model_relu,   X_train_t, y_train_t, X_test_t, y_test_t, criterion, epochs=300)
    print("  Training Linear model on two-moons...")
    linear_acc = _train_2d(model_linear, X_train_t, y_train_t, X_test_t, y_test_t, criterion, epochs=300)

    print(f"\n  ReLU model test accuracy:   {relu_acc:.1f}%")
    print(f"  Linear model test accuracy: {linear_acc:.1f}%")

    # ── Plot ──────────────────────────────────────────────────────
    _plot_decision_boundaries(
        model_relu, model_linear,
        X, y, X_test, y_test,
        relu_acc, linear_acc
    )


def _train_2d(model, X_train, y_train, X_test, y_test, criterion, epochs=300, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out  = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc   = 100.0 * (preds == y_test).float().mean().item()
    return acc


def _plot_decision_boundaries(model_relu, model_linear, X, y, X_test, y_test,
                               relu_acc, linear_acc):
    # Create mesh grid
    h       = 0.02
    x_min   = X[:, 0].min() - 0.5
    x_max   = X[:, 0].max() + 0.5
    y_min   = X[:, 1].min() - 0.5
    y_max   = X[:, 1].max() + 0.5
    xx, yy  = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
    grid    = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(DEVICE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Decision Boundaries: Two-Moons Dataset", fontsize=14, fontweight="bold")

    for ax, model, title, acc, color in [
        (axes[0], model_linear, "Linear Network (No Activations)", linear_acc, "Reds"),
        (axes[1], model_relu,   "ReLU Network (3 Hidden Layers)",  relu_acc,   "Blues"),
    ]:
        model.eval()
        with torch.no_grad():
            Z = model(grid).softmax(dim=1)[:, 1].cpu().numpy()
        Z = Z.reshape(xx.shape)

        # Background probability contour
        ax.contourf(xx, yy, Z, levels=50, cmap=color, alpha=0.4)
        ax.contour(xx, yy, Z, levels=[0.5], colors=["black"], linewidths=[2])

        # Scatter: train points
        scatter_colors = ["#D32F2F" if label == 0 else "#1565C0" for label in y]
        ax.scatter(X[:, 0], X[:, 1], c=scatter_colors,
                   s=15, alpha=0.4, zorder=2)
        # Highlight test points
        test_colors = ["#FF5722" if label == 0 else "#0288D1" for label in y_test]
        ax.scatter(X_test[:, 0], X_test[:, 1], c=test_colors,
                   s=30, edgecolors="white", linewidths=0.5, zorder=3)

        ax.set_title(f"{title}\nTest Accuracy: {acc:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "decision_boundary.png"), dpi=150)
    plt.close()
    print(f"\n  ✓ Saved decision_boundary.png")
    print(f"    → Linear boundary is a straight line; misclassifies many points")
    print(f"    → ReLU boundary curves around each crescent shape")


if __name__ == "__main__":
    run_two_moons()
