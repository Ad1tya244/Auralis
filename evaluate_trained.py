"""
evaluate_trained.py
-------------------
Run full evaluation on the provided trained + calibrated SDNN checkpoint:

    models/calibrated_sdnn_final.pth

This script:
  1. Loads the checkpoint (handles model_state_dict key + temperature)
  2. Applies temperature scaling to logits during evaluation
  3. Computes all 5 metrics on CIFAR-10 test set
  4. Generates and saves the reliability diagram
  5. Prints a self-diagnosis demo table (autonomous systems context)

Usage:
    cd sdnn_project
    python evaluate_trained.py

    # Custom checkpoint or device:
    python evaluate_trained.py --checkpoint models/calibrated_sdnn_final.pth --device cpu
"""

import argparse
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.checkpoint_utils import load_sdnn_checkpoint
from models.sdnn_model import CIFAR10_CLASSES
from training.train import build_dataloaders
from evaluation.metrics import compute_all_metrics, print_metrics
from evaluation.reliability_diagram import plot_reliability_diagram

os.makedirs("experiments", exist_ok=True)


def run(checkpoint_path: str, device_str: str = "auto"):

    # ── Device ────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device_str)

    print(f"\nDevice      : {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    model, temperature, meta = load_sdnn_checkpoint(checkpoint_path, device)
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Temperature : {temperature:.4f}" if temperature else "Temperature : None (not stored)")
    if meta:
        for k, v in meta.items():
            print(f"  {k}: {v}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nLoading CIFAR-10 test set...")
    _, test_loader = build_dataloaders(batch_size=128, num_workers=2)

    # ── Inference ─────────────────────────────────────────────────────────
    all_softmax, all_confidence, all_error_prob, all_labels = [], [], [], []

    print("Running inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            out    = model(images)

            # Apply temperature scaling to logits if available
            logits = out["logits"]
            if temperature is not None:
                logits = logits / temperature

            softmax = torch.softmax(logits, dim=1)

            all_softmax.append(softmax.cpu().numpy())
            all_confidence.append(out["confidence"].squeeze(1).cpu().numpy())
            all_error_prob.append(out["error_prob"].squeeze(1).cpu().numpy())
            all_labels.append(labels.numpy())

    softmax_probs = np.concatenate(all_softmax)
    confidence    = np.concatenate(all_confidence)
    error_prob    = np.concatenate(all_error_prob)
    labels_np     = np.concatenate(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = compute_all_metrics(softmax_probs, confidence, error_prob, labels_np)
    print_metrics(metrics, title="Calibrated SDNN — Autonomous Perception on CIFAR-10")

    # ── Save metrics to file ─────────────────────────────────────────────
    np.save("experiments/calibrated_sdnn_metrics.npy", metrics)
    print("Metrics saved → experiments/calibrated_sdnn_metrics.npy")

    # ── Reliability diagram ───────────────────────────────────────────────
    predicted = softmax_probs.argmax(axis=1)
    correct   = (predicted == labels_np)

    plot_reliability_diagram(
        confidence=confidence,
        correct=correct,
        n_bins=15,
        save_path="experiments/calibrated_sdnn_reliability.png",
        model_label="Calibrated SDNN (T=1.827)",
    )

    # ── Self-diagnosis demo ───────────────────────────────────────────────
    print("\n── Self-Diagnosis Demo (Autonomous Perception Context) ──────────")
    print("First 15 test images | Error threshold: 0.30")
    print()

    THRESHOLD  = 0.30
    N          = 15
    test_images, test_labels_batch = next(iter(test_loader))

    with torch.no_grad():
        out = model(test_images[:N].to(device))
        logits = out["logits"]
        if temperature is not None:
            logits = logits / temperature

    preds = logits.argmax(dim=1).cpu()
    confs = out["confidence"].squeeze(1).cpu()
    errs  = out["error_prob"].squeeze(1).cpu()
    acts  = test_labels_batch[:N]

    print(f"  {'#':>2}  {'Actual':<12} {'Predicted':<12} {'Conf':>6} {'ErrProb':>8} {'OK?':>4}  Action")
    print("  " + "─" * 72)
    for i in range(N):
        ok     = "✓" if preds[i] == acts[i] else "✗"
        action = "⚠  SAFE FALLBACK" if errs[i] > THRESHOLD else "Proceed"
        print(
            f"  {i+1:>2}  {CIFAR10_CLASSES[acts[i]]:<12} {CIFAR10_CLASSES[preds[i]]:<12} "
            f"{confs[i]:>6.3f} {errs[i]:>8.3f} {ok:>4}  {action}"
        )

    print()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/calibrated_sdnn_final.pth")
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()
    run(args.checkpoint, args.device)
