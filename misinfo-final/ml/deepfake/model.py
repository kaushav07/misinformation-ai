"""
EfficientNet-B4 deepfake detection model wrapper.

The model uses ImageNet-pretrained weights (no custom training needed).
For production, replace with FaceForensics++-finetuned weights.

Architecture:
  - Backbone: EfficientNet-B4 (timm)
  - Head: 2-class linear (real / fake)
  - Input: 224×224 RGB
  - Output: softmax probability
"""
import torch
import torch.nn as nn
from typing import Optional


class DeepfakeDetector(nn.Module):
    """EfficientNet-B4 binary classifier for deepfake detection."""

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities [real_prob, fake_prob]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


_model_instance: Optional[DeepfakeDetector] = None


def get_model(device: str = "cpu") -> DeepfakeDetector:
    """Singleton — loads once, reuses."""
    global _model_instance
    if _model_instance is None:
        _model_instance = DeepfakeDetector(pretrained=True)
        _model_instance.to(device)
        _model_instance.eval()
    return _model_instance
