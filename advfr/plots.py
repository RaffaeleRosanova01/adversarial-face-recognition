"""Security Evaluation Curves and other plots."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from advfr import config


def _finish(ax, ylabel, title, path, legend_loc="upper right"):
    ax.set_xlabel(r"Perturbation budget $\epsilon$ ($L_\infty$, inputs in [0, 1])")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.arange(0.0, 0.11, 0.01))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xlim(-0.005, 0.105)
    ax.set_ylim(-5, 105)
    ax.axvline(config.EPSILON_MAX, color="black", linestyle=":", linewidth=2, label=r"max $\epsilon$ = 0.1")
    ax.grid(alpha=0.4)
    ax.legend(loc=legend_loc, fontsize=9)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def security_evaluation_curve(curves, path, title, baseline=None, metric="accuracy"):
    """
    curves: {label: (epsilons, values)}. metric: 'accuracy' (untargeted: accuracy of
    NN1 vs epsilon, starting from the clean baseline) or 'asr' (targeted: success rate).
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    start = baseline if metric == "accuracy" else 0.0
    for label, (eps, values) in curves.items():
        ax.plot([0.0] + list(eps), [start] + list(values), marker="o", markersize=4, linewidth=1.5, label=label)
    if metric == "accuracy" and baseline is not None:
        ax.plot(0.0, baseline, marker="o", color="green", linestyle="None", label=f"clean accuracy ({baseline:.1f}%)")
    _finish(ax, "Accuracy (%)" if metric == "accuracy" else "Attack success rate (%)", title, path,
            "upper right" if metric == "accuracy" else "upper left")
