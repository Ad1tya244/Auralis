"""
loss_functions.py
-----------------
Combined weighted loss for the SDNN (Self-Diagnosing Neural Network).

Total Loss = CrossEntropyLoss(logits, labels)
           + λ1 × BCELoss(confidence, correctness_target)
           + λ2 × BCELoss(error_prob, error_target)

Where:
  correctness_target = 1.0  if prediction == ground truth label  (model was correct)
                     = 0.0  otherwise
  error_target       = 1.0  if prediction != ground truth label  (model made an error)
                     = 0.0  otherwise

Note: correctness_target + error_target = 1.0 always.

Recommended defaults: λ1 = λ2 = 0.5
"""

import torch
import torch.nn as nn
from typing import Tuple


class SDNNLoss(nn.Module):
    """
    Combined loss function for SDNN training.

    Args:
        lambda1 (float): Weight for the confidence calibration loss. Default: 0.5
        lambda2 (float): Weight for the error prediction loss.      Default: 0.5
    """

    def __init__(self, lambda1: float = 0.5, lambda2: float = 0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.confidence_loss_fn     = nn.BCELoss()
        self.error_loss_fn          = nn.BCELoss()

    def forward(
        self,
        logits:     torch.Tensor,   # (B, num_classes)  — raw class scores
        labels:     torch.Tensor,   # (B,)              — ground truth class indices
        confidence: torch.Tensor,   # (B, 1)            — confidence head output
        error_prob: torch.Tensor,   # (B, 1)            — error head output
    ) -> Tuple[torch.Tensor, dict]:
        """
        Computes the total SDNN loss and individual components.

        Returns:
            total_loss (Tensor): scalar loss for backpropagation
            components (dict):   individual loss values for logging
        """
        # --- Classification loss ---
        cls_loss = self.classification_loss_fn(logits, labels)

        # --- Build dynamic auxiliary targets ---
        with torch.no_grad():
            predicted_classes = logits.argmax(dim=1)            # (B,)
            # 1.0 where model predicted correctly, 0.0 otherwise
            correctness = (predicted_classes == labels).float().unsqueeze(1)  # (B, 1)
            # 1.0 where model made an error — complementary
            error_target = 1.0 - correctness                                  # (B, 1)

        # --- Confidence calibration loss ---
        conf_loss = self.confidence_loss_fn(confidence, correctness)

        # --- Error prediction loss ---
        err_loss = self.error_loss_fn(error_prob, error_target)

        # --- Combined loss ---
        total_loss = cls_loss + self.lambda1 * conf_loss + self.lambda2 * err_loss

        components = {
            "classification_loss": cls_loss.item(),
            "confidence_loss":     conf_loss.item(),
            "error_loss":          err_loss.item(),
            "total_loss":          total_loss.item(),
        }

        return total_loss, components


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loss_fn = SDNNLoss(lambda1=0.5, lambda2=0.5)

    logits     = torch.randn(8, 10)
    labels     = torch.randint(0, 10, (8,))
    confidence = torch.sigmoid(torch.randn(8, 1))
    error_prob = torch.sigmoid(torch.randn(8, 1))

    total, components = loss_fn(logits, labels, confidence, error_prob)

    assert total.item() > 0, "Loss should be positive"
    print("✓ SDNNLoss forward pass OK")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")
