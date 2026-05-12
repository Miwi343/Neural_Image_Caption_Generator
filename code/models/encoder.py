"""VGG-19 image encoder used by the attention decoder."""

import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """Expose VGG-19 conv features as 196 spatial annotation vectors."""

    def __init__(self, fine_tune: bool = False):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # Keep conv5_4 features before the final pool: (batch, 512, 14, 14).
        self.cnn = nn.Sequential(*list(vgg.features.children())[:36])

        for param in self.cnn.parameters():
            param.requires_grad = False

        if fine_tune:
            self._enable_fine_tune()

    def _enable_fine_tune(self):
        """Unfreeze the last two conv blocks for fine-tuning."""
        for param in self.cnn[20:].parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return features as (batch, 196, 512)."""
        out = self.cnn(images)

        batch_size, channels, height, width = out.shape
        if (channels, height, width) != (512, 14, 14):
            raise ValueError(
                "VGG encoder expected a (batch, 512, 14, 14) feature map from "
                f"224x224 ImageNet-normalized images, got {tuple(out.shape)}."
            )

        locations = height * width
        out = out.permute(0, 2, 3, 1)
        out = out.view(batch_size, locations, channels)

        return out
