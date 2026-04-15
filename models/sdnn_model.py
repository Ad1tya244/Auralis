"""
sdnn_model.py
-------------
Self-Diagnosing Neural Network — architecture matched to best_sdnn.pth.

Checkpoint metadata:
  epoch   : 72
  val_acc : 0.9382 (93.82%)
  temperature : 0.8784

Backbone : CIFARResNet18 (ResNet-18 adapted for 32×32 CIFAR images)
           Keys: backbone.features.{0..8}.*    feature_dim = 512

Heads (exact layer indices matching saved state_dict):
  classification_head : Linear(512, 10)

  confidence_head:
    [0] Linear(512, 128)   weight [128, 512]
    [1] ReLU
    [2] Dropout(0.3)
    [3] Linear(128, 1)     weight [1, 128]
    [4] Sigmoid

  error_head:
    [0] Linear(512, 128)   weight [128, 512]
    [1] ReLU
    [2] Dropout(0.3)
    [3] Linear(128, 1)     weight [1, 128]
    [4] Sigmoid
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class CIFARResNet18Backbone(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32×32 images):
      - First conv: 3×3, stride=1 (instead of 7×7, stride=2)
      - MaxPool replaced with Identity (avoids spatial downsampling on small inputs)
    Produces a 512-D global-average-pooled feature vector.

    Saved under: backbone.features.{0..8}.*
    """
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.features = nn.Sequential(
            base.conv1,    # index 0 — backbone.features.0.*
            base.bn1,      # index 1 — backbone.features.1.*
            base.relu,     # index 2
            base.maxpool,  # index 3 (Identity)
            base.layer1,   # index 4 — backbone.features.4.*
            base.layer2,   # index 5 — backbone.features.5.*
            base.layer3,   # index 6 — backbone.features.6.*
            base.layer4,   # index 7 — backbone.features.7.*
            base.avgpool,  # index 8
        )
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), 1)


# ---------------------------------------------------------------------------
# SDNN
# ---------------------------------------------------------------------------

class SDNN(nn.Module):
    """
    Self-Diagnosing Neural Network.

    Three heads on top of a CIFARResNet18 backbone:
      1. classification_head — raw logits (temperature-scaled at inference)
      2. confidence_head     — P(prediction is correct), shape (B, 1)
      3. error_head          — P(prediction is wrong),   shape (B, 1)

    Head architecture (indices 0–4) matches best_sdnn.pth:
      Linear(512→128) · ReLU · Dropout(0.3) · Linear(128→1) · Sigmoid
    """

    def __init__(self, num_classes: int = 10, head_dropout: float = 0.3):
        super().__init__()
        self.backbone = CIFARResNet18Backbone()
        d = self.backbone.feature_dim  # 512

        self.classification_head = nn.Linear(d, num_classes)

        self.confidence_head = nn.Sequential(
            nn.Linear(d, 128),       # [0] weight [128, 512]
            nn.ReLU(inplace=True),   # [1]
            nn.Dropout(head_dropout),# [2]
            nn.Linear(128, 1),       # [3] weight [1, 128]
            nn.Sigmoid(),            # [4]
        )

        self.error_head = nn.Sequential(
            nn.Linear(d, 128),       # [0] weight [128, 512]
            nn.ReLU(inplace=True),   # [1]
            nn.Dropout(head_dropout),# [2]
            nn.Linear(128, 1),       # [3] weight [1, 128]
            nn.Sigmoid(),            # [4]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        f = self.backbone(x)
        return {
            "logits":     self.classification_head(f),
            "confidence": self.confidence_head(f),
            "error_prob": self.error_head(f),
        }

    def predict(self, x: torch.Tensor) -> dict:
        """Convenience no-grad forward pass."""
        with torch.no_grad():
            out = self.forward(x)
        return {
            "predicted_class": out["logits"].argmax(dim=1),
            "confidence":      out["confidence"].squeeze(1),
            "error_prob":      out["error_prob"].squeeze(1),
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 dataset normalisation (must match training transform)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
