"""
Encoder: CNN feature extractor.

Paper section 3.1.1 — extract annotation vectors a_i ∈ R^D from a lower
convolutional layer so the decoder can selectively attend to spatial locations.

Ablation extension: the `feature_grid_size` argument (14 or 7) controls the
spatial resolution of the annotation vectors fed to the decoder.  The default
of 14 matches the paper exactly.  Setting it to 7 applies an adaptive average
pool after the conv features, halving the grid and reducing L from 196 to 49.

Future work (Issue #4): Add a shape regression test using a `224x224` input tensor
that asserts `self.cnn(images)` is `(B, 512, 14, 14)` and `forward(images)` is
`(B, 196, 512)`.
Future work (Issue #4, deferred): Add an encoder fine-tuning schedule that keeps the
encoder frozen initially, then unfreezes only the top conv layers with a smaller
learning rate after a configurable warm-up epoch.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    Wraps a pretrained VGG-16 and exposes a (grid×grid)×512 convolutional
    feature map.

    Default (feature_grid_size=14): matches paper section 4.3 exactly.
    Ablation (feature_grid_size=7): halves spatial resolution via adaptive
    average pooling, giving L=49 annotation vectors instead of 196.

    Output shape: (batch_size, L, D) = (batch_size, grid*grid, 512)
    """

    def __init__(self, fine_tune: bool = False, feature_grid_size: int = 14):
        super().__init__()

        if feature_grid_size not in (7, 14):
            raise ValueError(
                f"feature_grid_size must be 7 or 14, got {feature_grid_size}."
            )
        self.feature_grid_size = feature_grid_size

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # features[:30] → layers 0-29 → output (batch, 512, 14, 14).
        # Stops after conv5_3+relu, before the 5th max-pool (index 30).
        self.cnn = nn.Sequential(*list(vgg.features.children())[:30])

        # Freeze all parameters by default
        for param in self.cnn.parameters():
            param.requires_grad = False

        # For 7x7 ablation: adaptive average pool from 14x14 → 7x7.
        # nn.Identity keeps the 14x14 map for the default setting.
        self.downsample: nn.Module = (
            nn.AdaptiveAvgPool2d((7, 7))
            if feature_grid_size == 7
            else nn.Identity()
        )

        if fine_tune:
            self._enable_fine_tune()

    def _enable_fine_tune(self):
        """Unfreeze the last two conv blocks for fine-tuning."""
        for param in self.cnn[20:].parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, 224, 224) — ImageNet-normalized

        Returns:
            features: (batch_size, L, D) where L = grid*grid, D = 512
        """
        out = self.cnn(images)          # (batch, 512, 14, 14)
        out = self.downsample(out)      # (batch, 512, 14, 14) or (batch, 512, 7, 7)

        batch_size, D, H, W = out.shape
        expected = self.feature_grid_size
        if D != 512 or H != expected or W != expected:
            raise ValueError(
                f"Encoder expected (batch, 512, {expected}, {expected}), "
                f"got {tuple(out.shape)}."
            )
        L = H * W  # 196 or 49

        out = out.permute(0, 2, 3, 1)    # (batch, H, W, 512)
        out = out.view(batch_size, L, D) # (batch, L, 512)

        return out
