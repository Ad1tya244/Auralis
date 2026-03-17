"""
backbone.py
-----------
CIFAR-10 adapted ResNet-18 backbone for the SDNN project.

Standard ResNet-18 is designed for 224×224 inputs. For CIFAR-10 (32×32), we:
  1. Replace the first 7×7 conv (stride=2) with a 3×3 conv (stride=1)
  2. Remove the initial MaxPool layer

This preserves the spatial dimensions through the early layers and produces
a valid 512-dimensional feature vector via Global Average Pooling.

Output shape: (batch_size, 512)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class CIFARResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone adapted for 32×32 CIFAR-10 inputs.

    Modifications vs standard ResNet-18:
      - First conv: 7×7 stride-2 → 3×3 stride-1, same padding
      - MaxPool after first conv: removed
      - FC classification head: removed (we only want the 512-D feature vector)

    The rest of the residual blocks and Global Average Pooling remain intact.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # Load standard ResNet-18
        base = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")

        # --- CIFAR adaptation ---
        # Replace first conv: 7×7, stride=2, pad=3 → 3×3, stride=1, pad=1
        base.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Remove MaxPool (would shrink 32×32 too aggressively)
        base.maxpool = nn.Identity()

        # Keep all residual layers and avgpool; drop the FC head
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,   # nn.Identity — no-op
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,   # Global Average Pooling → (batch, 512, 1, 1)
        )

        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images of shape (batch, 3, 32, 32)

        Returns:
            features: (batch, 512) feature vectors
        """
        out = self.features(x)          # (batch, 512, 1, 1)
        out = torch.flatten(out, 1)     # (batch, 512)
        return out


# ---------------------------------------------------------------------------
# Quick sanity check (run this file directly to verify shapes)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = CIFARResNet18Backbone(pretrained=False)
    dummy = torch.randn(4, 3, 32, 32)
    features = model(dummy)
    assert features.shape == (4, 512), f"Unexpected shape: {features.shape}"
    print(f"✓ Backbone output shape: {features.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
