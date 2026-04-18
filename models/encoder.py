"""
Encoder: CNN feature extractor.

Paper section 3.1.1 — extract annotation vectors a_i ∈ R^D from a lower
convolutional layer so the decoder can selectively attend to spatial locations.

TODO (Issue #4): Add a shape regression test using a `224x224` input tensor
that asserts `self.cnn(images)` is `(B, 512, 14, 14)` and `forward(images)` is
`(B, 196, 512)`.
TODO (Issue #4, deferred): Add an encoder fine-tuning schedule that keeps the
encoder frozen initially, then unfreezes only the top conv layers with a smaller
learning rate after a configurable warm-up epoch.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    Wraps a pretrained VGG-16 and exposes a 14x14x512 convolutional feature
    map.

    VGG-16 features layer indices (31 layers total, 0-30):
      0-3:   conv1_1, relu, conv1_2, relu
      4:     MaxPool → 112x112
      5-8:   conv2_1, relu, conv2_2, relu
      9:     MaxPool → 56x56
      10-15: conv3_1, relu, conv3_2, relu, conv3_3, relu
      16:    MaxPool → 28x28
      17-22: conv4_1, relu, conv4_2, relu, conv4_3, relu
      23:    MaxPool → 14x14   ← 4th max-pool; spatial size becomes 14x14
      24-29: conv5_1, relu, conv5_2, relu, conv5_3, relu
      30:    MaxPool → 7x7    ← 5th max-pool (NOT included)

    features[:30] keeps layers 0-29: through conv5_3+relu, before the 5th
    max-pool.  Output is (batch, 512, 14, 14) for a 224x224 input, matching
    the paper's "14×14×512 feature map … before max pooling" (section 4.3).

    Output shape: (batch_size, L, D) = (batch_size, 196, 512)
    where L = 14*14 = 196 annotation locations and D = 512 channels.

    TODO (Issue #4, deferred): If fine-tuning is assigned, expose exactly which
    VGG layers are unfrozen and preserve the current frozen-by-default behavior
    for the proposal reproduction.
    """

    def __init__(self, fine_tune: bool = False):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # features[:30] → layers 0-29 → output (batch, 512, 14, 14).
        # Stops after conv5_3+relu, before the 5th max-pool (index 30).
        self.cnn = nn.Sequential(*list(vgg.features.children())[:30])

        # Freeze all parameters by default
        for param in self.cnn.parameters():
            param.requires_grad = False

        # TODO (Issue #4, deferred): Keep `fine_tune=False` as the default; if
        # `True`, prove via a parameter-count or requires_grad check that only
        # the intended top layers become trainable.
        if fine_tune:
            self._enable_fine_tune()

    def _enable_fine_tune(self):
        """Unfreeze the last two conv blocks for fine-tuning."""
        # TODO (Issue #4, deferred): Restrict unfreezing to `self.cnn[20:]` or a
        # narrower top-block slice, then document the exact layer indices and
        # optimizer LR used for those params.
        for param in self.cnn[20:].parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, 224, 224) — ImageNet-normalized

        Returns:
            features: (batch_size, L, D) = (batch_size, 196, 512)

        TODO (Issue #4): Raise a clear assertion or ValueError if the CNN output
        is not `(batch, 512, 14, 14)` so preprocessing/model-slice mistakes fail
        fast instead of silently corrupting attention geometry.
        """
        # (batch, 512, 14, 14)
        out = self.cnn(images)

        batch_size, D, H, W = out.shape  # D=512, H=W=14
        L = H * W  # 196

        # Reshape to (batch, L, D) so each of the 196 locations is a vector
        # TODO (Issue #4): Add a tensor-ordering test confirming this permute +
        # flatten keeps each annotation vector length `D=512` and maps the
        # 14x14 spatial grid to `L=196` locations in a deterministic order.
        out = out.permute(0, 2, 3, 1)   # (batch, 14, 14, 512)
        out = out.view(batch_size, L, D) # (batch, 196, 512)

        return out
