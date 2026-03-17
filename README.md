# SDNN — Self-Diagnosing Neural Network
### Reliable Image Classification for Autonomous Perception Systems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Domain](https://img.shields.io/badge/Domain-Autonomous%20Systems-green)](.)

---

## Overview

Standard CNNs output highly confident predictions even when they are wrong — a critical safety issue in autonomous systems. The **Self-Diagnosing Neural Network (SDNN)** addresses this by extending a ResNet-18 backbone with two extra diagnostic branches that allow the model to estimate its own reliability.

For every input image, the SDNN outputs:
```
Prediction:      Truck          ← what it sees
Confidence:      0.84           ← how sure it is (calibrated)
Error Probability: 0.18         ← probability it made a mistake
```

If `error_prob > 0.3`, a downstream autonomous system can trigger a safe fallback (e.g., slow down, defer to human operator).

---

## Architecture

```
Input Image (3 × 32 × 32)
        │
        ▼
CIFARResNet18Backbone (512-D feature vector)
        │
  ┌─────┼──────────┐
  ▼     ▼          ▼
Class  Confidence  Error
Head    Head       Prediction Head
  │      │              │
  ▼      ▼              ▼
logits  conf (0-1)   err_prob (0-1)
```

**Loss:**
```
Total Loss = CrossEntropyLoss(logits, labels)
           + 0.5 × BCELoss(confidence, correctness_target)
           + 0.5 × BCELoss(error_prob, error_target)
```

---

## Project Structure

```
sdnn_project/
├── data/                         # CIFAR-10 auto-download cache
├── models/
│   ├── backbone.py               # CIFAR-adapted ResNet-18
│   └── sdnn_model.py             # Full SDNN with 3 heads
├── training/
│   ├── loss_functions.py         # Combined weighted loss
│   ├── train.py                  # Main training script
│   └── train_baseline.py         # Standard CNN + Temperature Scaling
├── evaluation/
│   ├── metrics.py                # ECE, NLL, Brier Score, AUROC, Accuracy
│   └── reliability_diagram.py    # Confidence vs accuracy plot
├── experiments/                  # Checkpoints & results (auto-created)
├── notebooks/                    # Colab-ready notebooks
├── evaluate_trained.py           # Universal eval script for trained checkpoints
└── README.md
```

---

## Setup

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn tqdm
```

Recommended: **Google Colab** with T4 or A100 GPU.

---

## Usage

### 1. Train the SDNN
```bash
cd sdnn_project
python training/train.py --epochs 50 --batch-size 64 --lr 0.001
```

**Quick smoke test (CPU-friendly):**
```bash
python training/train.py --epochs 1 --subset 500
```

### 2. Evaluate Trained Checkpoints
We provide a universal evaluation script to handle temperature-scaled models (like `calibrated_sdnn_final.pth`) and standard checkpoints alike.

```bash
python evaluate_trained.py --checkpoint models/calibrated_sdnn_final.pth
```
*This generates a reliability diagram, saves metrics, and prints a self-diagnosis demo output.*

### 3. Evaluate Manually (Standard Checkpoints)
```bash
python evaluation/metrics.py --checkpoint experiments/best_sdnn.pth
python evaluation/reliability_diagram.py --checkpoint experiments/best_sdnn.pth
```

### 4. Train Baselines
```bash
# Standard CNN only
python training/train_baseline.py --epochs 50

# Standard CNN + Temperature Scaling
python training/train_baseline.py --epochs 50 --apply-temperature-scaling
```

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Correct predictions / total | ≥ 88% |
| **ECE** | Expected Calibration Error (↓ better) | ≤ 0.05 |
| **NLL** | Negative Log-Likelihood (↓ better) | — |
| **Brier Score** | Probability accuracy (↓ better) | — |
| **AUROC** | Error detection capability (↑ better) | ≥ 0.70 |

---

## Dataset: CIFAR-10

10 classes, each framed in the autonomous perception context:

| Class | Autonomous Relevance |
|-------|---------------------|
| Automobile, Truck | Vehicle detection |
| Airplane, Ship | Extended environment |
| Deer, Horse, Bird | Road hazard (animals) |
| Cat, Dog, Frog | Pedestrian-adjacent obstacles |

---

## Technology Stack

- **Language:** Python 3.8+
- **Framework:** PyTorch 2.0+
- **Libraries:** torchvision, numpy, scikit-learn, matplotlib, seaborn, tqdm
- **Environment:** Google Colab / Jupyter Notebook
