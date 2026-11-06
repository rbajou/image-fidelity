import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_fidelity.degradation import GaussianNoise, PoissonNoise, ReadoutNoise, SaltPepperNoise


def test_poisson_noise_is_seeded_quantized_and_preserves_shape_dtype():
    """Poisson noise is repeatable, quantized, and leaves its input unchanged."""
    image = np.full((32, 24, 3), 0.5, dtype=np.float32)
    original = image.copy()

    first = PoissonNoise(photon_count=100, seed=7)(image)
    second = PoissonNoise(photon_count=100, seed=7)(image)

    assert first.shape == image.shape
    assert first.dtype == image.dtype
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first * 100, np.round(first * 100))
    np.testing.assert_array_equal(image, original)
    assert not np.array_equal(first, image)


def test_gaussian_noise_is_seeded_and_has_requested_mean_and_sigma():
    """Gaussian noise matches its configured statistics and is repeatable."""
    image = np.zeros((128, 128), dtype=np.float32)
    noise = GaussianNoise(sigma=0.1, mean=0.03, seed=11)

    result = noise(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert abs(float(result.mean()) - 0.03) < 0.01
    assert abs(float(result.std()) - 0.1) < 0.01
    np.testing.assert_array_equal(result, GaussianNoise(sigma=0.1, mean=0.03, seed=11)(image))


def test_salt_pepper_noise_applies_configured_extreme_values():
    """Salt-and-pepper noise uses the configured high and low values."""
    image = np.full((12, 10, 3), 0.5, dtype=np.float32)

    salt = SaltPepperNoise(amount=1.0, salt_vs_pepper=1.0, salt_value=0.9, seed=3)(image)
    pepper = SaltPepperNoise(amount=1.0, salt_vs_pepper=0.0, pepper_value=0.1, seed=3)(image)

    np.testing.assert_array_equal(salt, np.full_like(image, 0.9))
    np.testing.assert_array_equal(pepper, np.full_like(image, 0.1))
    np.testing.assert_array_equal(image, np.full_like(image, 0.5))


def test_readout_noise_is_seeded_additive_noise_with_bias():
    """Readout noise is repeatable, additive, and centered on its bias."""
    image = np.zeros((64, 64), dtype=np.float32)
    degradation = ReadoutNoise(sigma=0.02, bias=0.15, seed=19)

    result = degradation(image)
    repeated = ReadoutNoise(sigma=0.02, bias=0.15, seed=19)(image)
    shifted = degradation(image + 1.0)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result, repeated)
    np.testing.assert_allclose(shifted - result, 1.0, atol=1e-6)
    assert abs(float(result.mean()) - 0.15) < 0.005
