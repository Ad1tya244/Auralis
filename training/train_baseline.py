"""
train_baseline.py
-----------------
Trains the two baseline models for comparison with the SDNN:

  Baseline 1: Standard CNN
      Same ResNet-18 backbone + classification head only.
      No confidence or error prediction branches.

  Baseline 2: Temperature Scaling
      Post-hoc calibration applied to the Standard CNN.
      A single scalar temperature T is learned on the validation set
      to sharpen or soften the softmax distribution.

Usage:
    cd sdnn_project
    python training/train_baseline.py --epochs 50
    python training/train_baseline.py --epochs 50 --apply-temperature-scaling
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import CIFARResNet18Backbone
from training.train  import build_dataloaders
from evaluation.metrics import compute_all_metrics, print_metrics


# ---------------------------------------------------------------------------
# Baseline 1: Standard CNN
# ---------------------------------------------------------------------------

class StandardCNN(nn.Module):
    """
    Standard ResNet-18 CNN with a single classification head.
    No confidence or error prediction branches.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone     = CIFARResNet18Backbone(pretrained=False)
        self.classifier   = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Baseline 2: Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    """
    Post-hoc temperature scaling calibration wrapper.

    Wraps a trained StandardCNN and applies a learnable temperature T
    to the logits before softmax:

        calibrated_probs = softmax(logits / T)

    T is optimised on the validation set using NLL loss.
    T > 1 → softer (less confident) predictions.
    T < 1 → sharper (more confident) predictions.
    """

    def __init__(self, model: StandardCNN):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature.clamp(min=1e-6)

    def fit(self, val_loader, device, max_iter: int = 50):
        """Learn temperature T on validation data."""
        self.model.eval()
        self.to(device)

        nll_loss  = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        def eval_step():
            optimizer.zero_grad()
            scaled = all_logits / self.temperature.clamp(min=1e-6)
            loss   = nll_loss(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        print(f"  Temperature Scaling: learned T = {self.temperature.item():.4f}")
        return self


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_standard_cnn(model, train_loader, test_loader, optimizer, scheduler, device, epochs):
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_state    = None

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Acc':>9} {'Time':>7}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        total_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total   = 0
        val_loss    = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits  = model(images)
                loss    = loss_fn(logits, labels)
                val_loss    += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += images.size(0)
        val_loss /= val_total
        val_acc   = val_correct / val_total

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.4f} {val_acc:>9.4f} {elapsed:>6.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"         ↳ Saved best (val_loss={val_loss:.4f})")

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def collect_predictions(model, loader, device, use_temperature: bool = False):
    all_softmax = []
    all_labels  = []

    model.eval()
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        softmax = torch.softmax(logits, dim=1)
        all_softmax.append(softmax.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_softmax), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train baseline models for SDNN comparison")
    parser.add_argument("--epochs",                  type=int,   default=50)
    parser.add_argument("--batch-size",              type=int,   default=64)
    parser.add_argument("--lr",                      type=float, default=1e-3)
    parser.add_argument("--apply-temperature-scaling", action="store_true")
    parser.add_argument("--device",                  type=str,   default="auto")
    parser.add_argument("--save-dir",                type=str,   default="experiments/")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    train_loader, test_loader = build_dataloaders(batch_size=args.batch_size)

    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.save_dir.lstrip("/"),
    )
    os.makedirs(save_dir, exist_ok=True)

    # --- Baseline 1: Standard CNN ---
    print("\n" + "=" * 60)
    print("  Training Baseline 1: Standard CNN")
    print("=" * 60)

    std_cnn   = StandardCNN(num_classes=10).to(device)
    optimizer = optim.Adam(std_cnn.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    std_cnn = train_standard_cnn(
        std_cnn, train_loader, test_loader,
        optimizer, scheduler, device, args.epochs,
    )

    ckpt_path = os.path.join(save_dir, "baseline_standard_cnn.pth")
    torch.save({"model_state": std_cnn.state_dict()}, ckpt_path)

    softmax_probs, labels = collect_predictions(std_cnn, test_loader, device)
    confidence = softmax_probs.max(axis=1)   # max softmax as naive confidence
    error_prob = 1.0 - confidence            # naive error estimate

    metrics_std = compute_all_metrics(softmax_probs, confidence, error_prob, labels)
    print_metrics(metrics_std, title="Baseline 1: Standard CNN")

    # --- Baseline 2: Temperature Scaling ---
    if args.apply_temperature_scaling:
        print("=" * 60)
        print("  Applying Baseline 2: Temperature Scaling")
        print("=" * 60)

        ts_model = TemperatureScaling(std_cnn).to(device)
        ts_model.fit(test_loader, device)

        softmax_ts, labels_ts = collect_predictions(ts_model, test_loader, device)
        confidence_ts = softmax_ts.max(axis=1)
        error_prob_ts = 1.0 - confidence_ts

        metrics_ts = compute_all_metrics(softmax_ts, confidence_ts, error_prob_ts, labels_ts)
        print_metrics(metrics_ts, title="Baseline 2: Temperature Scaling")

        ts_ckpt_path = os.path.join(save_dir, "baseline_temperature_scaling.pth")
        torch.save({
            "model_state": std_cnn.state_dict(),
            "temperature": ts_model.temperature.item(),
        }, ts_ckpt_path)
        print(f"  Temperature Scaling checkpoint saved: {ts_ckpt_path}")


if __name__ == "__main__":
    main()
