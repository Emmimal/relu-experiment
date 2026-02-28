"""
run_all.py
----------
Runs all four experiments in sequence and summarises results.

Usage:
    python run_all.py              # Full run (all experiments)
    python run_all.py --fast       # Reduced epochs for quick verification
    python run_all.py --mnist      # MNIST experiment only
    python run_all.py --depth      # Depth sweep only
    python run_all.py --moons      # Two-moons only
    python run_all.py --activations # Activation comparison only
    python run_all.py --gradients  # Gradient analysis only
"""

import argparse
import time
import os


def banner(title: str) -> None:
    print(f"\n{'#'*60}")
    print(f"#  {title}")
    print(f"{'#'*60}\n")


def run_all(args):
    os.makedirs("results", exist_ok=True)
    start = time.time()

    if args.mnist or args.all:
        banner("Experiment 1/4 — MNIST: ReLU vs Linear")
        if args.fast:
            import mnist_experiment
            mnist_experiment.EPOCHS = 5
        from mnist_experiment import run_experiment
        run_experiment()

    if args.depth or args.all:
        banner("Experiment 2/4 — Depth Sweep")
        if args.fast:
            import depth_sweep
            depth_sweep.EPOCHS = 5
            depth_sweep.NUM_SEEDS = 1
        from depth_sweep import run_depth_sweep
        run_depth_sweep()

    if args.moons or args.all:
        banner("Experiment 3/4 — Two-Moons Decision Boundary")
        from two_moons import run_two_moons
        run_two_moons()

    if args.activations or args.all:
        banner("Experiment 4a/4 — Activation Function Comparison")
        if args.fast:
            import activations_comparison
            activations_comparison.EPOCHS = 5
            activations_comparison.NUM_SEEDS = 1
        from activations_comparison import run_activation_comparison
        run_activation_comparison()

    if args.gradients or args.all:
        banner("Experiment 4b/4 — Gradient Norm Analysis")
        if args.fast:
            import gradient_analysis
            gradient_analysis.EPOCHS = 5
        from gradient_analysis import run_gradient_analysis
        run_gradient_analysis()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ✅ All experiments complete in {elapsed:.1f}s")
    print(f"  📁 Results saved to ./results/")
    print(f"\n  Files generated:")
    for f in sorted(os.listdir("results")):
        size = os.path.getsize(os.path.join("results", f))
        print(f"    {f:<45} ({size/1024:.1f} KB)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReLU experiment suite")
    parser.add_argument("--fast",        action="store_true", help="Quick run with fewer epochs")
    parser.add_argument("--mnist",       action="store_true", help="Run MNIST experiment only")
    parser.add_argument("--depth",       action="store_true", help="Run depth sweep only")
    parser.add_argument("--moons",       action="store_true", help="Run two-moons only")
    parser.add_argument("--activations", action="store_true", help="Run activation comparison only")
    parser.add_argument("--gradients",   action="store_true", help="Run gradient analysis only")
    args = parser.parse_args()

    # Default: run all if no specific flag given
    args.all = not any([args.mnist, args.depth, args.moons, args.activations, args.gradients])

    run_all(args)
