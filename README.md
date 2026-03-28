# Auralis — Self-Diagnosing Neural Network
### Reliable Image Classification for Autonomous Perception Systems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey)](https://flask.palletsprojects.com)
[![Domain](https://img.shields.io/badge/Domain-Autonomous%20Systems-green)](./)

---

## Overview

Standard CNNs output highly confident predictions even when they are wrong — a critical safety issue in autonomous systems. **Auralis SDNN** addresses this by extending a ResNet-18 backbone with two extra diagnostic branches that allow the model to estimate its own reliability.

For every input image, the SDNN outputs:
```
Prediction:        truck     ← what it sees
Confidence:        0.84      ← how sure it is (calibrated)
Entropy:           0.45      ← spread of the probability distribution
Error Probability: 0.18      ← probability it made a mistake
```

If `confidence < 0.7`, `entropy > 1.5`, or `error_prob > 0.4`, the model rejects the prediction as *Unknown / Out-of-Distribution*. A downstream autonomous system can then trigger a safe fallback (e.g. slow down, defer to a human operator).

The project also includes a **full web UI** — upload any image and get an interactive prediction report with an animated donut chart, entropy gauge, confidence bars, copy-to-clipboard, and PNG export.

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
logits  conf (0–1)   err_prob (0–1)
```

**Loss:**
```
L_total = CrossEntropyLoss(logits, labels)
        + λ₁ × BCELoss(confidence, correctness_target)
        + λ₂ × BCELoss(error_prob, error_target)
```

---

## Project Structure

```
Auralis - SDNN/
├── models/
│   ├── __init__.py
│   ├── backbone.py           # CIFAR-adapted ResNet-18
│   ├── checkpoint_utils.py   # Checkpoint load/save helpers
│   ├── sdnn_model.py         # Full SDNN with 3 heads
│   └── best_sdnn.pth         # Trained weights (not in git)
├── training/
│   ├── __init__.py
│   ├── loss_functions.py     # Combined weighted loss
│   ├── train.py              # Main training script
│   └── train_baseline.py     # Standard CNN + Temperature Scaling
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # ECE, NLL, Brier, AUROC, Accuracy
│   └── reliability_diagram.py
├── notebooks/
│   ├── sdnn_colab.ipynb              # Full Colab notebook
│   └── sdnn_fast_training_colab.ipynb
├── static/
│   ├── app.js                # Frontend logic
│   └── style.css             # UI styles
├── templates/
│   └── index.html            # Flask HTML template
├── data/                     # CIFAR-10 auto-downloaded here (gitignored)
├── app.py                    # Flask web server
├── evaluate_trained.py       # Evaluation script for trained checkpoints
├── package.json              # npm dev shortcut
└── README.md
```

---

## Requirements

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| **Python** | 3.8+ | Runtime |
| **pip** | bundled with Python | Package installer |
| **Node.js + npm** | any recent LTS | `npm run dev` shortcut (optional) |
| **Git** | any | Cloning the repo |

> **GPU:** Training is supported on CUDA (NVIDIA), Apple MPS (M-series Mac), and CPU. Apple MPS is auto-detected on M1/M2/M3 Macs.

---

## Step-by-Step Setup

### 1 — Clone the repository

```bash
git clone <your-repo-url>
cd "Auralis - SDNN"
```

---

### 2 — Create and activate a Python virtual environment

```bash
# Create the venv (only needed once)
python3 -m venv .venv

# Activate it — macOS / Linux
source .venv/bin/activate

# Activate it — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal prompt.

---

### 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install torch torchvision flask pillow numpy scikit-learn matplotlib seaborn tqdm
```

> **GPU-specific install (NVIDIA CUDA):** Visit https://pytorch.org/get-started/locally/ and use the custom install command for your CUDA version instead of the generic `pip install torch torchvision`.

---

### 4 — Obtain a trained model checkpoint

The model weights (`models/best_sdnn.pth`) are **not tracked in git** (they are gitignored).

**Option A — Train from scratch (recommended for best results):**
```bash
# Full training — 50 epochs
python training/train.py --epochs 50 --batch-size 64

# Quick smoke test (CPU-friendly, ~2 min)
python training/train.py --epochs 1 --subset 500
```
CIFAR-10 is downloaded automatically on first run. The checkpoint is saved to `models/best_sdnn.pth`.

**Option B — Use Google Colab then download locally:**
1. Open `notebooks/sdnn_fast_training_colab.ipynb` in Google Colab.
2. Run all cells. Training uses a free T4/A100 GPU.
3. Download the output `best_sdnn.pth` and place it at `models/best_sdnn.pth`.

---

## Running the Web UI

### Start the server

**With npm (recommended shortcut):**
```bash
npm run dev
```

**Without npm (direct Python):**
```bash
source .venv/bin/activate   # skip if already active
python app.py --port 8080
```

The server prints:
```
[SDNN] Loading checkpoint from: .../models/best_sdnn.pth
[SDNN] Using device: mps
[SDNN] Ready — listening for requests...
 * Running on http://127.0.0.1:8080
```

### Open the app
Navigate to **http://localhost:8080** in your browser.

### Terminate the server
Press **`Ctrl + C`** in the terminal where the server is running.

---

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | `50` | Number of training epochs |
| `--batch-size N` | `64` | Mini-batch size |
| `--lr N` | `0.001` | Learning rate |
| `--lambda1 N` | `0.5` | Weight for confidence calibration loss |
| `--lambda2 N` | `0.5` | Weight for error prediction loss |
| `--subset N` | `None` | Subset size (for quick tests) |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `--save-dir` | `models/` | Directory to save checkpoints |

**Example — fast local test:**
```bash
python training/train.py --epochs 3 --subset 2000 --device cpu
```

**Example — full Colab/GPU training:**
```bash
python training/train.py --epochs 50 --batch-size 128 --device cuda
```

---

## Evaluating a Checkpoint

```bash
python evaluate_trained.py --checkpoint models/best_sdnn.pth
```

This script:
1. Loads the checkpoint (handles temperature-scaled models automatically)
2. Runs inference on the full CIFAR-10 test set (10,000 images)
3. Prints all 5 reliability metrics
4. Saves `experiments/calibrated_sdnn_metrics.npy`
5. Generates `experiments/calibrated_sdnn_reliability.png`
6. Prints a self-diagnosis demo table (15 sample predictions)

### Train + evaluate the baseline CNN
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

Downloaded automatically on first training run into `data/`. 10 classes, each mapped to an autonomous-perception context:

| Class | Autonomous Context |
|-------|--------------------|
| Automobile, Truck | Vehicle detection |
| Airplane, Ship | Extended environment |
| Deer, Horse, Bird | Road hazard — animals |
| Cat, Dog, Frog | Pedestrian-adjacent obstacles |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch 2.0+, torchvision |
| Web Server | Flask 3.0+ |
| Frontend | Vanilla HTML / CSS / JS (no frameworks) |
| Dataset | CIFAR-10 (auto-downloaded via torchvision) |
| Training env | Local Mac (MPS) · Google Colab (CUDA) |

---

## Troubleshooting

**`FileNotFoundError: models/best_sdnn.pth`**
The checkpoint is missing. Follow [Step 4](#4--obtain-a-trained-model-checkpoint) above to train or download it.

**`ModuleNotFoundError`**
Make sure the virtual environment is activated (`source .venv/bin/activate`) and all dependencies are installed (`pip install ...`).

**`MPS backend not available`**
You're on an Intel Mac or Linux. The code automatically falls back to CPU.

**CIFAR-10 download fails**
Check your internet connection. Alternatively, manually download from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and extract into `data/cifar-10-batches-py/`.
