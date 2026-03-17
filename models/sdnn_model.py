"""
sdnn_model.py
-------------
Self-Diagnosing Neural Network (SDNN) for autonomous perception.

Architecture:
    Input Image (3 × 32 × 32)
        │
        ▼
    CIFARResNet18Backbone
        │
    512-D Feature Vector
        │
    ┌───┼───────────┐
    ▼   ▼           ▼
  Class  Confidence  Error
  Head    Head       Prediction Head
    │       │           │
    ▼       ▼           ▼
  logits  conf_score  error_prob
 (B, C)   (B, 1)      (B, 1)

- Classification Head  → CrossEntropyLoss
- Confidence Head      → BCELoss  (target: 1 if correct, 0 if wrong)
- Error Prediction Head→ BCELoss  (target: 1 if wrong,   0 if correct)
"""

import torch
import torch.nn as nn
try:
    from models.backbone import CIFARResNet18Backbone
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.backbone import CIFARResNet18Backbone


class SDNN(nn.Module):
    """
    Self-Diagnosing Neural Network.

    Args:
        num_classes (int): Number of output classes. Default: 10 (CIFAR-10).
        pretrained_backbone (bool): Whether to initialise backbone with
                                    ImageNet weights. Default: False.

    Outputs (dict):
        logits      – raw class scores, shape (B, num_classes)
        confidence  – calibrated confidence,  shape (B, 1),  range (0, 1)
        error_prob  – error probability,       shape (B, 1),  range (0, 1)
    """

    def __init__(self, num_classes: int = 10, pretrained_backbone: bool = False):
        super().__init__()

        self.backbone = CIFARResNet18Backbone(pretrained=pretrained_backbone)
        feat_dim = self.backbone.feature_dim   # 512

        # --- Head 1: Classification ---
        # No activation here; CrossEntropyLoss expects raw logits
        self.classification_head = nn.Linear(feat_dim, num_classes)

        # --- Head 2: Confidence Calibration ---
        # Sigmoid squashes output to (0, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),
        )

        # --- Head 3: Error Prediction ---
        # Sigmoid squashes output to (0, 1)
        # High value → model thinks it made an error
        self.error_head = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),
        )

        # Weight initialisation for the three heads
        self._init_heads()

    def _init_heads(self):
        for module in [
            self.classification_head,
            *self.confidence_head.children(),
            *self.error_head.children(),
        ]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: images, shape (B, 3, 32, 32)

        Returns:
            dict with keys:
                'logits'      – (B, num_classes)
                'confidence'  – (B, 1)
                'error_prob'  – (B, 1)
        """
        features = self.backbone(x)                     # (B, 512)

        logits     = self.classification_head(features) # (B, num_classes)
        confidence = self.confidence_head(features)     # (B, 1)
        error_prob = self.error_head(features)          # (B, 1)

        return {
            "logits":     logits,
            "confidence": confidence,
            "error_prob": error_prob,
        }

    def predict(self, x: torch.Tensor) -> dict:
        """
        Convenience method for inference. Returns human-readable predictions.

        Returns:
            dict with keys:
                'predicted_class' – (B,) int tensor
                'confidence'      – (B,) float tensor, range (0, 1)
                'error_prob'      – (B,) float tensor, range (0, 1)
        """
        with torch.no_grad():
            out = self.forward(x)
        predicted_class = out["logits"].argmax(dim=1)
        return {
            "predicted_class": predicted_class,
            "confidence":      out["confidence"].squeeze(1),
            "error_prob":      out["error_prob"].squeeze(1),
        }


# CIFAR-10 class names mapped to their autonomous systems relevance
CIFAR10_CLASSES = [
    "airplane",    # Airspace / drone detection
    "automobile",  # Ego / surrounding vehicles
    "bird",        # Road hazard (animal)
    "cat",         # Pedestrian-adjacent obstacle
    "deer",        # Road hazard (animal)
    "dog",         # Pedestrian-adjacent obstacle
    "frog",        # Pedestrian-adjacent obstacle
    "horse",       # Road hazard (animal)
    "ship",        # Maritime / port environments
    "truck",       # Large vehicle detection
]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = SDNN(num_classes=10)
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)

    assert out["logits"].shape     == (4, 10), f"logits: {out['logits'].shape}"
    assert out["confidence"].shape == (4, 1),  f"conf:   {out['confidence'].shape}"
    assert out["error_prob"].shape == (4, 1),  f"err:    {out['error_prob'].shape}"

    print("✓ SDNN forward pass OK")
    print(f"  logits shape:     {out['logits'].shape}")
    print(f"  confidence shape: {out['confidence'].shape}")
    print(f"  error_prob shape: {out['error_prob'].shape}")

    pred = model.predict(dummy)
    print(f"\n✓ predict() output:")
    print(f"  predicted_class: {pred['predicted_class']}")
    print(f"  confidence:      {pred['confidence']}")
    print(f"  error_prob:      {pred['error_prob']}")
