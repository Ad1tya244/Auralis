"""
sdnn_model.py — SDNNv2 (updated architecture with deeper diagnostic heads)
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
    def __init__(self, num_classes=10, head_dropout=0.3):
        super().__init__()
        self.backbone = CIFARResNet18Backbone()
        d = self.backbone.feature_dim

        self.classification_head = nn.Linear(d, num_classes)

        self.confidence_head = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.error_head = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.backbone(x)
        return {
            "logits":     self.classification_head(f),
            "confidence": self.confidence_head(f),
            "error_prob": self.error_head(f),
        }

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return {
            "predicted_class": out["logits"].argmax(dim=1),
            "confidence":      out["confidence"].squeeze(1),
            "error_prob":      out["error_prob"].squeeze(1),
        }


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
