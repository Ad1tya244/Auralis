# Auralis — Self-Diagnosing Neural Network
### Reliable Image Classification for Autonomous Perception Systems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey)](https://flask.palletsprojects.com)
[![Domain](https://img.shields.io/badge/Domain-Autonomous%20Systems-green)](./)

---

## Overview

Standard CNNs are overconfident — they output high-probability predictions even when they are wrong, which is a critical safety issue in autonomous systems. **Auralis SDNN** addresses this by extending a ResNet-18 backbone with two extra diagnostic branches that let the model estimate its own reliability.

For every input image, the SDNN outputs:
```
Prediction:        truck     ← what it sees
Confidence:        0.91      ← how sure it is (calibrated)
Entropy:           0.31      ← spread of the probability distribution
Error Probability: 0.04      ← probability it made a mistake
```

If `confidence < 0.7`, `entropy > 1.5`, or `error_prob > 0.4`, the model rejects the prediction as *Unknown / Out-of-Distribution*. A downstream system can then trigger a safe fallback (e.g. slow down, defer to a human operator).

The project also ships a **full web UI** — upload any image and get an interactive prediction report with an animated donut chart, entropy gauge, confidence bars, session history, copy-to-clipboard, and PNG export.

---

## Architecture (v2)

```
Input Image (3 × 32 × 32)
        │
        ▼
CIFARResNet18Backbone (512-D feature vector)
        │
  ┌─────┼──────────────────────┐
  ▼     ▼                      ▼
Class  Confidence Head         Error Prediction Head
Head   (512→128→Dropout→1)     (512→128→Dropout→1)
  │         │                       │
  ▼         ▼                       ▼
logits  conf (0–1)             err_prob (0–1)
```

**v2 improvements over v1:**

| Technique | Benefit |
|-----------|---------|
| Deeper diagnostic heads (2-layer MLP + Dropout) | Better calibrated confidence & error signals |
| Label Smoothing (ε = 0.1) | Prevents overconfidence |
| CutMix + MixUp augmentation | Higher accuracy, softer learned distributions |
| 2-phase training (60 + 15 epochs) | Backbone first, then diagnostic heads fine-tuned |
| Progressive λ schedule (0.1 → 1.5) | Backbone gets a head start before calibration loss takes over |
| Post-hoc Temperature Scaling | Proven ECE reduction with no accuracy cost |
| Cosine Annealing with Warm Restarts | Better convergence |

**Loss:**
```
L_total = SoftCrossEntropy(logits, soft_labels)   ← supports CutMix/MixUp
        + λ × BCELoss(confidence, correctness)
        + λ × BCELoss(error_prob, error_target)

where λ ramps from 0.1 → 1.5 over training
```

---

## Project Structure

```
Auralis - SDNN/
├── models/
│   ├── __init__.py
│   ├── backbone.py             # CIFAR-adapted ResNet-18
│   ├── checkpoint_utils.py     # Universal checkpoint loader
│   ├── sdnn_model.py           # SDNNv2 — 3-head architecture
│   └── best_sdnn.pth           # Trained weights (not in git)
├── training/
│   ├── __init__.py
│   ├── loss_functions.py       # Combined weighted loss (v1)
│   ├── train.py                # Local training script
│   └── train_baseline.py       # Standard CNN + Temperature Scaling baseline
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # ECE, NLL, Brier, AUROC, Accuracy
│   └── reliability_diagram.py  # Confidence vs accuracy plot
├── notebooks/
│   ├── sdnn_training_v2_colab.ipynb   # ✅ Recommended — Train SDNNv2 on Colab
│   └── sdnn_colab.ipynb               # Legacy full-pipeline notebook
├── static/
│   ├── app.js                  # Frontend — particle animation, donut chart, history
│   └── style.css               # UI styles — dark glassmorphism theme
├── templates/
│   └── index.html              # Flask HTML template
├── data/                       # CIFAR-10 auto-downloaded here (gitignored)
├── app.py                      # Flask web server
├── evaluate_trained.py         # CLI evaluation script
├── package.json                # npm dev shortcut
└── README.md
```

---

## Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.8+ | Runtime |
| **pip** | bundled | Package installer |
| **Node.js + npm** | any LTS | `npm run dev` shortcut (optional) |

> **GPU:** Training supports CUDA (NVIDIA), Apple MPS (M-series Mac), and CPU. MPS is auto-detected.

---

## Step-by-Step Setup

### 1 — Clone the repository

```bash
git clone <your-repo-url>
cd "Auralis - SDNN"
```

### 2 — Create and activate a virtual environment

```bash
# Create (once only)
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install torch torchvision flask pillow numpy scikit-learn matplotlib seaborn tqdm pandas
```

> For NVIDIA GPU: use the custom install from https://pytorch.org/get-started/locally/

---

## Training a Model (Recommended: Google Colab)

### Option A — Colab (best results, free GPU)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/sdnn_training_v2_colab.ipynb`
3. **Runtime → Change runtime type → T4 GPU**
4. Run all cells (~40 min on T4, ~15 min on A100)
5. The last two cells download:
   - `best_sdnn.pth` → place at `models/best_sdnn.pth`
   - `sdnn_model.py` → replace `models/sdnn_model.py`

> The v2 notebook trains with CutMix, MixUp, label smoothing, 2-phase training, and temperature scaling. Expected: **Accuracy ≥ 90%, ECE ≤ 0.03**.

### Option B — Train locally

```bash
# Full training (saves to models/best_sdnn.pth)
python training/train.py --epochs 50 --batch-size 64

# Quick smoke test (CPU-friendly, ~2 min)
python training/train.py --epochs 1 --subset 500
```

---

## Running the Web UI

### Start

```bash
# With npm (recommended)
npm run dev

# Without npm
source .venv/bin/activate
python app.py --port 8080
```

Open **http://localhost:8080** in your browser.

### Terminate

Press **`Ctrl + C`** in the terminal.

### Web UI Features

| Feature | Description |
|---------|-------------|
| **Drag & drop upload** | JPEG, PNG, WebP, BMP |
| **Donut chart** | Interactive class distribution with hover |
| **4 stat cards** | Confidence, Error Probability, Top-1 Softmax, Shannon Entropy |
| **Session History** | Scrollable strip of all predictions; click to replay |
| **Copy** | Copies a formatted text summary to clipboard |
| **Export PNG** | Downloads a stats card as a PNG image |
| **Enter key shortcut** | Press Enter to trigger analysis |

---

## Evaluating a Checkpoint

```bash
python evaluate_trained.py --checkpoint models/best_sdnn.pth
```

Outputs: accuracy, ECE, NLL, Brier Score, AUROC, reliability diagram, self-diagnosis demo table.

### Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | `50` | Training epochs |
| `--batch-size N` | `64` | Batch size |
| `--lr N` | `0.001` | Learning rate |
| `--lambda1 N` | `0.5` | Confidence loss weight |
| `--lambda2 N` | `0.5` | Error prediction loss weight |
| `--subset N` | `None` | Use a small subset (smoke test) |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `--save-dir` | `models/` | Checkpoint output directory |

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Correct predictions / total | ≥ 90% |
| **ECE** | Expected Calibration Error (↓ better) | ≤ 0.03 |
| **NLL** | Negative Log-Likelihood (↓ better) | — |
| **Brier Score** | Probability forecast accuracy (↓ better) | — |
| **AUROC** | Error self-detection capability (↑ better) | ≥ 0.75 |

---

## Dataset: CIFAR-10

Downloaded automatically on first run into `data/`. 10 classes, each mapped to an autonomous-perception context:

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
| Training | Google Colab (T4/A100) · Local Mac (MPS) |

---

## Troubleshooting

**`FileNotFoundError: models/best_sdnn.pth`**
The checkpoint is missing. Train using `notebooks/sdnn_training_v2_colab.ipynb` on Colab and download the weights.

**`RuntimeError: Error(s) in loading state_dict`**
The checkpoint architecture doesn't match `sdnn_model.py`. If you trained with v2 notebook, make sure you also downloaded and replaced `models/sdnn_model.py` with the v2 version from Colab.

**`ModuleNotFoundError`**
Activate the virtual environment: `source .venv/bin/activate`, then reinstall dependencies.

**`MPS backend not available`**
Intel Mac or Linux — the code automatically falls back to CPU.

**CIFAR-10 download fails**
Check your internet connection. Alternatively, manually download from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and extract into `data/cifar-10-batches-py/`.
