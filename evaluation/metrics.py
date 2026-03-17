"""
metrics.py
----------
Evaluation metrics for the SDNN — autonomy-focused reliability assessment.

Metrics implemented:
  - Classification Accuracy
  - Expected Calibration Error (ECE)
  - Negative Log-Likelihood (NLL)
  - Brier Score
  - AUROC for Error Detection

Usage:
    cd sdnn_project
    python evaluation/metrics.py --checkpoint experiments/best_sdnn.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sdnn_model import SDNN
from training.train import build_dataloaders


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_accuracy(correct: np.ndarray) -> float:
    """
    Args:
        correct: boolean array, True where model predicted correctly
    Returns:
        accuracy as a float in [0, 1]
    """
    return float(np.mean(correct))


def compute_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error.

    Bins predictions by confidence score and checks whether
    empirical accuracy matches confidence in each bin.

    Lower ECE → better calibration. Perfectly calibrated model has ECE = 0.

    Args:
        confidence: array of shape (N,), confidence scores in (0, 1)
        correct:    boolean array of shape (N,), True where model was correct
        n_bins:     number of equally-spaced bins
    Returns:
        ECE as a float
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    n         = len(confidence)

    for i in range(n_bins):
        lo, hi   = bin_edges[i], bin_edges[i + 1]
        in_bin   = (confidence >= lo) & (confidence < hi)
        n_in_bin = in_bin.sum()

        if n_in_bin == 0:
            continue

        bin_acc  = correct[in_bin].mean()
        bin_conf = confidence[in_bin].mean()
        ece     += (n_in_bin / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_nll(softmax_probs: np.ndarray, labels: np.ndarray, eps: float = 1e-9) -> float:
    """
    Negative Log-Likelihood.

    Measures quality of predicted class probabilities.
    Lower NLL → model assigns higher probability to the correct class.

    Args:
        softmax_probs: shape (N, num_classes), row-wise softmax probabilities
        labels:        shape (N,), ground truth class indices
    Returns:
        mean NLL per sample
    """
    n = len(labels)
    correct_probs = softmax_probs[np.arange(n), labels]
    nll = -np.log(np.clip(correct_probs, eps, 1.0))
    return float(np.mean(nll))


def compute_brier_score(softmax_probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier Score (multi-class generalisation).

    Measures accuracy of predicted probability distributions.
    Lower Brier Score → better calibration.

    Args:
        softmax_probs: shape (N, num_classes)
        labels:        shape (N,), ground truth class indices
    Returns:
        mean Brier Score per sample
    """
    n, c         = softmax_probs.shape
    one_hot      = np.zeros_like(softmax_probs)
    one_hot[np.arange(n), labels] = 1.0
    brier = np.mean(np.sum((softmax_probs - one_hot) ** 2, axis=1))
    return float(brier)


def compute_auroc_error_detection(error_prob: np.ndarray, correct: np.ndarray) -> float:
    """
    AUROC for Error Detection.

    Measures how well error_prob distinguishes correct predictions (0)
    from incorrect predictions (1).

    Higher AUROC → model can better flag its own mistakes.
    In autonomous systems: high AUROC = model knows when to trigger safe fallback.

    Args:
        error_prob: shape (N,), error probability from the SDNN error head
        correct:    boolean array (N,), True where model was correct
    Returns:
        AUROC score in [0, 1]
    """
    error_labels = (~correct).astype(int)   # 1 = model made an error
    if error_labels.sum() == 0 or error_labels.sum() == len(error_labels):
        return float("nan")
    return float(roc_auc_score(error_labels, error_prob))


def compute_all_metrics(
    softmax_probs: np.ndarray,
    confidence:    np.ndarray,
    error_prob:    np.ndarray,
    labels:        np.ndarray,
    n_bins:        int = 15,
) -> dict:
    """
    Compute all SDNN evaluation metrics in one call.

    Args:
        softmax_probs: (N, num_classes) — softmax probabilities from classification head
        confidence:    (N,)             — confidence head outputs
        error_prob:    (N,)             — error prediction head outputs
        labels:        (N,)             — ground truth class indices
    Returns:
        dict of metric names → values
    """
    predicted = softmax_probs.argmax(axis=1)
    correct   = (predicted == labels)

    return {
        "accuracy":    compute_accuracy(correct),
        "ece":         compute_ece(confidence, correct, n_bins),
        "nll":         compute_nll(softmax_probs, labels),
        "brier_score": compute_brier_score(softmax_probs, labels),
        "auroc_error": compute_auroc_error_detection(error_prob, correct),
    }


def print_metrics(metrics: dict, title: str = "SDNN Evaluation Metrics"):
    width = 52
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")
    print(f"  {'Accuracy':<30} {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  {'ECE (↓ better)':<30} {metrics['ece']:.4f}")
    print(f"  {'NLL (↓ better)':<30} {metrics['nll']:.4f}")
    print(f"  {'Brier Score (↓ better)':<30} {metrics['brier_score']:.4f}")
    print(f"  {'AUROC Error Detection (↑ better)':<30} {metrics['auroc_error']:.4f}")
    print(f"{'═' * width}\n")

    # Autonomous systems interpretation
    auroc = metrics["auroc_error"]
    ece   = metrics["ece"]
    print("  Autonomous Systems Assessment:")
    if ece < 0.05:
        print("  ✓ Confidence is well-calibrated — predictions are trustworthy")
    else:
        print("  ✗ Confidence is poorly calibrated — predictions may mislead downstream systems")

    if not np.isnan(auroc) and auroc >= 0.70:
        print("  ✓ Model can reliably flag its own errors — safe fallback is viable")
    else:
        print("  ✗ Error detection is weak — model cannot reliably self-diagnose")
    print()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(checkpoint_path: str, device_str: str = "auto"):
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

    all_softmax    = []
    all_confidence = []
    all_error_prob = []
    all_labels     = []

    for images, labels in test_loader:
        images = images.to(device)
        out    = model(images)

        softmax = torch.softmax(out["logits"], dim=1)
        all_softmax.append(softmax.cpu().numpy())
        all_confidence.append(out["confidence"].squeeze(1).cpu().numpy())
        all_error_prob.append(out["error_prob"].squeeze(1).cpu().numpy())
        all_labels.append(labels.numpy())

    softmax_probs = np.concatenate(all_softmax)
    confidence    = np.concatenate(all_confidence)
    error_prob    = np.concatenate(all_error_prob)
    labels        = np.concatenate(all_labels)

    metrics = compute_all_metrics(softmax_probs, confidence, error_prob, labels)
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SDNN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device",     type=str, default="auto")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.device)
