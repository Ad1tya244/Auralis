"""
reliability_diagram.py
----------------------
Generate and save a reliability diagram comparing
confidence scores vs actual accuracy per bin.

A perfectly calibrated model lies on the diagonal (y = x).
Deviation above the diagonal = underconfidence.
Deviation below the diagonal = overconfidence.

Usage:
    cd sdnn_project
    python evaluation/reliability_diagram.py --checkpoint experiments/best_sdnn.pth
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sdnn_model import SDNN
from training.train import build_dataloaders


def plot_reliability_diagram(
    confidence:   np.ndarray,
    correct:      np.ndarray,
    n_bins:       int = 15,
    save_path:    str = None,
    model_label:  str = "SDNN",
):
    """
    Plot confidence vs actual accuracy.

    Args:
        confidence:  (N,) confidence scores in (0, 1)
        correct:     (N,) boolean array, True where model was correct
        n_bins:      number of bins for the histogram
        save_path:   if provided, saves the figure to this path
        model_label: label for the plot legend
    """
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accs  = []
    bin_confs = []
    bin_counts = []
    gaps = []

    for i in range(n_bins):
        lo, hi   = bin_edges[i], bin_edges[i + 1]
        in_bin   = (confidence >= lo) & (confidence < hi)
        n_in_bin = in_bin.sum()

        if n_in_bin == 0:
            bin_accs.append(0)
            bin_confs.append(bin_centres[i])
            bin_counts.append(0)
            gaps.append(0)
        else:
            acc  = correct[in_bin].mean()
            conf = confidence[in_bin].mean()
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(n_in_bin)
            gaps.append(conf - acc)   # positive = overconfident

    bin_accs   = np.array(bin_accs)
    bin_confs  = np.array(bin_confs)
    bin_counts = np.array(bin_counts)
    gaps       = np.array(gaps)

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Reliability Diagram — {model_label}\n(Autonomous Perception on CIFAR-10)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # --- Left: Reliability diagram ---
    ax = axes[0]
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.6)

    # Gap bars (over/underconfidence shading)
    bar_width = 1 / n_bins * 0.9
    for i in range(n_bins):
        if bin_counts[i] == 0:
            continue
        color = "#e74c3c" if gaps[i] > 0 else "#3498db"
        ax.bar(
            bin_centres[i], bin_accs[i],
            width=bar_width, align="center",
            color=color, alpha=0.7, edgecolor="white", linewidth=0.5,
        )

    # Confidence ticks
    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Actual Accuracy",  fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, alpha=0.3)

    over_patch  = mpatches.Patch(color="#e74c3c", alpha=0.7, label="Overconfident")
    under_patch = mpatches.Patch(color="#3498db", alpha=0.7, label="Underconfident")
    diag_patch  = plt.Line2D([0], [0], color="k", linestyle="--", label="Perfect calibration")
    ax.legend(handles=[diag_patch, over_patch, under_patch], fontsize=10)
    ax.set_title("Calibration Curve", fontsize=12)

    # --- Right: Sample distribution per bin ---
    ax2 = axes[1]
    ax2.set_facecolor("#f8f9fa")
    ax2.bar(
        bin_centres, bin_counts,
        width=bar_width, align="center",
        color="#2ecc71", alpha=0.8, edgecolor="white", linewidth=0.5,
    )
    ax2.set_xlabel("Confidence Score",        fontsize=12)
    ax2.set_ylabel("Number of Samples",       fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_xticks(np.linspace(0, 1, 6))
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_title("Sample Distribution per Confidence Bin", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Reliability diagram saved to: {save_path}")
    else:
        plt.show()

    plt.close()
    return fig


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_reliability_diagram(checkpoint_path: str, save_path: str = None, device_str: str = "auto"):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = SDNN(num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, test_loader = build_dataloaders(batch_size=128, num_workers=2)

    all_confidence = []
    all_correct    = []

    for images, labels in test_loader:
        images = images.to(device)
        out    = model(images)

        preds   = out["logits"].argmax(dim=1).cpu()
        correct = (preds == labels)

        all_confidence.append(out["confidence"].squeeze(1).cpu().numpy())
        all_correct.append(correct.numpy())

    confidence = np.concatenate(all_confidence)
    correct    = np.concatenate(all_correct)

    if save_path is None:
        exp_dir   = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"
        )
        save_path = os.path.join(exp_dir, "reliability_diagram.png")

    plot_reliability_diagram(confidence, correct, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reliability diagram for SDNN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-path",  type=str, default=None)
    parser.add_argument("--device",     type=str, default="auto")
    args = parser.parse_args()
    run_reliability_diagram(args.checkpoint, args.save_path, args.device)
