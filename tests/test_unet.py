import numpy as np
import torch

from image_fidelity.restoration.nn.models import UNet
from image_fidelity.restoration.nn.torch_restorer import TorchRestorer


def test_unet_channels_and_shape():
    model = UNet()
    widths = [m[0].out_channels for m in model.encoders] + [model.bottleneck[0].out_channels]
    assert widths == [32, 64, 128, 256]
    x = torch.randn(2, 1, 50, 37)
    assert model(x).shape == x.shape


def test_residual_unet_starts_as_identity():
    model = UNet(residual=True).eval()
    x = torch.randn(1, 1, 32, 32)
    assert torch.allclose(model(x), x)


def test_torch_restorer_preserves_shape():
    restored = TorchRestorer(UNet(base_channels=4, depth=2))(np.random.rand(30, 20).astype(np.float32))
    assert restored.shape == (30, 20)
