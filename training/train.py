"""
train.py
--------
Main training script for the SDNN (Self-Diagnosing Neural Network).

Usage:
    python training/train.py [--epochs N] [--batch-size N] [--lr N]
                             [--lambda1 N] [--lambda2 N]
                             [--subset N] [--device cuda|cpu]
                             [--save-dir experiments/]

Run from inside sdnn_project/:
    cd sdnn_project
    python training/train.py --epochs 50
"""

import argparse
import os
import sys
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sdnn_model import SDNN, CIFAR10_CLASSES
from training.loss_functions import SDNNLoss


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(batch_size: int = 64, subset: int = None, num_workers: int = 2):
    """Build CIFAR-10 train and test DataLoaders with augmentation."""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    # Optional subset for quick smoke tests
    if subset is not None:
        train_dataset = Subset(train_dataset, range(min(subset, len(train_dataset))))
        test_dataset  = Subset(test_dataset,  range(min(subset // 5, len(test_dataset))))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(images)

        loss, _ = loss_fn(
            out["logits"],
            labels,
            out["confidence"],
            out["error_prob"],
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = out["logits"].argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    all_confidence = []
    all_error_prob = []
    all_correct    = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        out = model(images)
        loss, _ = loss_fn(
            out["logits"],
            labels,
            out["confidence"],
            out["error_prob"],
        )

        total_loss += loss.item() * images.size(0)
        preds       = out["logits"].argmax(dim=1)
        batch_correct = (preds == labels)
        correct      += batch_correct.sum().item()
        total        += images.size(0)

        all_confidence.append(out["confidence"].squeeze(1).cpu())
        all_error_prob.append(out["error_prob"].squeeze(1).cpu())
        all_correct.append(batch_correct.cpu())

    all_confidence = torch.cat(all_confidence).numpy()
    all_error_prob = torch.cat(all_error_prob).numpy()
    all_correct    = torch.cat(all_correct).numpy()

    return {
        "loss":       total_loss / total,
        "accuracy":   correct / total,
        "confidence": all_confidence,
        "error_prob": all_error_prob,
        "correct":    all_correct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SDNN for autonomous perception")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--lambda1",    type=float, default=0.5,
                        help="Weight for confidence calibration loss")
    parser.add_argument("--lambda2",    type=float, default=0.5,
                        help="Weight for error prediction loss")
    parser.add_argument("--subset",     type=int,   default=None,
                        help="Use a small subset (for smoke tests)")
    parser.add_argument("--device",     type=str,   default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--save-dir",   type=str,   default="experiments/")
    parser.add_argument("--num-workers", type=int,  default=2)
    args = parser.parse_args()

    # --- Device ---
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        subset=args.subset,
        num_workers=args.num_workers,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test  batches: {len(test_loader)}")

    # --- Model ---
    model = SDNN(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # --- Loss & Optimiser ---
    loss_fn   = SDNNLoss(lambda1=args.lambda1, lambda2=args.lambda2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)

    # --- Checkpoint directory ---
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.save_dir.lstrip("/"),
    )
    os.makedirs(save_dir, exist_ok=True)

    # --- Training loop ---
    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training for {args.epochs} epochs...\n")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Time':>7}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics           = evaluate(model, test_loader, loss_fn, device)
        val_loss, val_acc     = val_metrics["loss"], val_metrics["accuracy"]

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>9.4f} {elapsed:>6.1f}s")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(save_dir, "best_sdnn.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "args":        vars(args),
            }, ckpt_path)
            print(f"         ↳ Saved best checkpoint (val_loss={val_loss:.4f})")

    # Save training history
    history_path = os.path.join(save_dir, "training_history.npy")
    np.save(history_path, history)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(save_dir, 'best_sdnn.pth')}")
    print(f"History:    {history_path}")


if __name__ == "__main__":
    main()
