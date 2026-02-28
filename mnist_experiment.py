"""
mnist_experiment.py
-------------------
Trains MLPWithReLU vs MLPLinear on MNIST under identical conditions.
Produces:
  - results/loss_curves.png
  - results/accuracy_curves.png
  - results/gradient_norms_epoch10.png
  - results/mnist_results.txt

Matches the exact experiment described in the article.
"""

import os
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from models import MLPWithReLU, MLPLinear
from utils  import (init_kaiming, init_xavier,
                    get_mnist_loaders, train_one_epoch,
                    evaluate, print_gradient_norms)

# ---------------------------------------------
# Config
# ---------------------------------------------
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3
GRAD_LOG_EPOCH = 10          # Capture gradient norms at this epoch
RESULTS_DIR    = "results"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment():
    print(f"\n{'='*60}")
    print(f"  ReLU vs Linear Network — MNIST Experiment")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # -- Data -----------------------------------------------------
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    # -- Models ---------------------------------------------------
    model_relu   = init_kaiming(MLPWithReLU()).to(DEVICE)
    model_linear = init_xavier(MLPLinear()).to(DEVICE)

    print(f"ReLU   model parameters: {sum(p.numel() for p in model_relu.parameters()):,}")
    print(f"Linear model parameters: {sum(p.numel() for p in model_linear.parameters()):,}\n")

    # -- Optimizers & Loss ----------------------------------------
    opt_relu   = torch.optim.Adam(model_relu.parameters(),   lr=LR)
    opt_linear = torch.optim.Adam(model_linear.parameters(), lr=LR)
    criterion  = nn.CrossEntropyLoss()

    # -- Logging --------------------------------------------------
    history = {
        "relu":   {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []},
        "linear": {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []},
    }
    grad_norms_at_target = {}

    # -- Training Loop --------------------------------------------
    print(f"{'Epoch':<6} {'ReLU Loss':<12} {'ReLU Acc%':<12} {'Lin Loss':<12} {'Lin Acc%':<12}")
    print("-" * 56)

    for epoch in range(1, EPOCHS + 1):
        log_grads = (epoch == GRAD_LOG_EPOCH)

        # Train both models on the same data order
        relu_loss, relu_acc, relu_grads   = train_one_epoch(
            model_relu,   train_loader, opt_relu,   criterion, DEVICE, log_grads)
        lin_loss,  lin_acc,  lin_grads    = train_one_epoch(
            model_linear, train_loader, opt_linear, criterion, DEVICE, log_grads)

        # Evaluate
        relu_test_loss,  relu_test_acc  = evaluate(model_relu,   test_loader, criterion, DEVICE)
        lin_test_loss,   lin_test_acc   = evaluate(model_linear, test_loader, criterion, DEVICE)

        # Record
        history["relu"]["train_loss"].append(relu_loss)
        history["relu"]["train_acc"].append(relu_acc)
        history["relu"]["test_loss"].append(relu_test_loss)
        history["relu"]["test_acc"].append(relu_test_acc)

        history["linear"]["train_loss"].append(lin_loss)
        history["linear"]["train_acc"].append(lin_acc)
        history["linear"]["test_loss"].append(lin_test_loss)
        history["linear"]["test_acc"].append(lin_test_acc)

        print(f"{epoch:<6} {relu_loss:<12.4f} {relu_acc:<12.2f} {lin_loss:<12.4f} {lin_acc:<12.2f}")

        if log_grads:
            grad_norms_at_target["relu"]   = relu_grads
            grad_norms_at_target["linear"] = lin_grads
            print(f"\n  -> Gradient norms captured at epoch {GRAD_LOG_EPOCH}")
            print_gradient_norms(relu_grads,  "ReLU Model")
            print_gradient_norms(lin_grads,   "Linear Model")

    # -- Final Results --------------------------------------------
    print(f"\n{'='*60}")
    print("  Final Results")
    print(f"{'='*60}")
    print(f"\n  ReLU Model:")
    print(f"    Train accuracy : {history['relu']['train_acc'][-1]:.1f}%")
    print(f"    Test  accuracy : {history['relu']['test_acc'][-1]:.1f}%")
    print(f"    Final train loss: {history['relu']['train_loss'][-1]:.4f}")
    print(f"\n  Linear Model:")
    print(f"    Train accuracy : {history['linear']['train_acc'][-1]:.1f}%")
    print(f"    Test  accuracy : {history['linear']['test_acc'][-1]:.1f}%")
    print(f"    Final train loss: {history['linear']['train_loss'][-1]:.4f}")

    # -- Save Text Results -----------------------------------------
    with open(os.path.join(RESULTS_DIR, "mnist_results.txt"), "w", encoding="utf-8") as f:
        f.write("MNIST Experiment Results\n")
        f.write("="*40 + "\n\n")
        f.write("Epoch | ReLU Loss | ReLU Acc% | Lin Loss | Lin Acc%\n")
        f.write("-"*55 + "\n")
        for i in range(EPOCHS):
            f.write(
                f"{i+1:<6} | {history['relu']['train_loss'][i]:.4f}    | "
                f"{history['relu']['train_acc'][i]:.2f}     | "
                f"{history['linear']['train_loss'][i]:.4f}    | "
                f"{history['linear']['train_acc'][i]:.2f}\n"
            )
        f.write("\n\nFinal Test Accuracy:\n")
        f.write(f"  ReLU Model:   {history['relu']['test_acc'][-1]:.2f}%\n")
        f.write(f"  Linear Model: {history['linear']['test_acc'][-1]:.2f}%\n")

    # -- Plots -----------------------------------------------------
    _plot_loss_curves(history)
    _plot_accuracy_curves(history)
    if grad_norms_at_target:
        _plot_gradient_norms(grad_norms_at_target, epoch=GRAD_LOG_EPOCH)

    print(f"\n  Plots saved to ./{RESULTS_DIR}/")


# ---------------------------------------------
# Plotting helpers
# ---------------------------------------------

def _plot_loss_curves(history: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, EPOCHS + 1)

    ax.plot(epochs, history["relu"]["train_loss"],   color="#2196F3", linewidth=2,   label="ReLU — Train")
    ax.plot(epochs, history["relu"]["test_loss"],    color="#2196F3", linewidth=2,   linestyle="--", label="ReLU — Test")
    ax.plot(epochs, history["linear"]["train_loss"], color="#F44336", linewidth=2,   label="Linear — Train")
    ax.plot(epochs, history["linear"]["test_loss"],  color="#F44336", linewidth=2,   linestyle="--", label="Linear — Test")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Loss Curves: ReLU vs Linear Network (MNIST)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, EPOCHS)

    # Annotate plateau
    final_lin_loss = history["linear"]["train_loss"][-1]
    ax.axhline(y=final_lin_loss, color="#F44336", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.annotate(f"Linear plateau ~ {final_lin_loss:.2f}",
                xy=(EPOCHS * 0.6, final_lin_loss + 0.05),
                color="#F44336", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curves.png"), dpi=150)
    plt.close()
    print(f"  [OK] Saved loss_curves.png")


def _plot_accuracy_curves(history: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, EPOCHS + 1)

    ax.plot(epochs, history["relu"]["train_acc"],   color="#2196F3", linewidth=2,   label="ReLU — Train")
    ax.plot(epochs, history["relu"]["test_acc"],    color="#2196F3", linewidth=2,   linestyle="--", label="ReLU — Test")
    ax.plot(epochs, history["linear"]["train_acc"], color="#F44336", linewidth=2,   label="Linear — Train")
    ax.plot(epochs, history["linear"]["test_acc"],  color="#F44336", linewidth=2,   linestyle="--", label="Linear — Test")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy: ReLU vs Linear Network (MNIST)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, EPOCHS)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curves.png"), dpi=150)
    plt.close()
    print(f"  [OK] Saved accuracy_curves.png")


def _plot_gradient_norms(grad_norms_at_target: dict, epoch: int) -> None:
    relu_norms   = grad_norms_at_target.get("relu", {})
    linear_norms = grad_norms_at_target.get("linear", {})

    # Align layers for both models
    layers = [k for k in relu_norms.keys() if k in linear_norms]
    if not layers:
        return

    relu_vals   = [relu_norms[l]   for l in layers]
    linear_vals = [linear_norms[l] for l in layers]
    x = range(len(layers))
    short_labels = [l.replace(".weight", "") for l in layers]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], relu_vals,   width, label="ReLU Model",   color="#2196F3", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], linear_vals, width, label="Linear Model", color="#F44336", alpha=0.85)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Gradient Norm (mean over epoch)", fontsize=12)
    ax.set_title(f"Gradient Norms at Epoch {epoch}: ReLU vs Linear", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h,
                f'{h:.5f}', ha='center', va='bottom', fontsize=7, color="#1565C0")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h,
                f'{h:.5f}', ha='center', va='bottom', fontsize=7, color="#B71C1C")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"gradient_norms_epoch{epoch}.png"), dpi=150)
    plt.close()
    print(f"  [OK] Saved gradient_norms_epoch{epoch}.png")


# ---------------------------------------------
if __name__ == "__main__":
    run_experiment()
