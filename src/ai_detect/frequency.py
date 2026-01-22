"""Frequency domain analysis for AI image detection.

GAN and diffusion models leave artifacts in the frequency domain that
can be detected via DCT/FFT analysis. This provides a complementary
signal to neural network classifiers.
"""

import numpy as np
from PIL import Image
from scipy import fftpack


def compute_dct_features(image: Image.Image) -> dict[str, float]:
    """Compute DCT-based features for AI detection.

    GANs often leave periodic artifacts visible in the DCT domain,
    particularly in high-frequency components.
    """
    img = np.array(image.convert("L"), dtype=np.float64)

    dct = fftpack.dct(fftpack.dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")

    h, w = dct.shape
    low_freq = dct[: h // 4, : w // 4]
    high_freq = dct[h // 2 :, w // 2 :]

    low_energy = np.sum(np.abs(low_freq) ** 2)
    high_energy = np.sum(np.abs(high_freq) ** 2)
    total_energy = np.sum(np.abs(dct) ** 2)

    if total_energy == 0:
        return {"dct_ratio": 0.5, "high_freq_ratio": 0.0}

    high_freq_ratio = high_energy / total_energy
    dct_ratio = high_energy / (low_energy + 1e-10)

    return {
        "dct_ratio": float(dct_ratio),
        "high_freq_ratio": float(high_freq_ratio),
    }


def compute_fft_features(image: Image.Image) -> dict[str, float]:
    """Compute FFT-based features for AI detection.

    AI-generated images often show characteristic patterns in
    the frequency spectrum, including periodic peaks from upsampling.
    """
    img = np.array(image.convert("L"), dtype=np.float64)

    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    inner_mask = r < min(h, w) // 8
    mid_mask = (r >= min(h, w) // 8) & (r < min(h, w) // 3)
    outer_mask = r >= min(h, w) // 3

    inner_mean = np.mean(magnitude[inner_mask]) if np.any(inner_mask) else 0
    mid_mean = np.mean(magnitude[mid_mask]) if np.any(mid_mask) else 0
    outer_mean = np.mean(magnitude[outer_mask]) if np.any(outer_mask) else 0

    if inner_mean == 0:
        spectral_ratio = 0.5
    else:
        spectral_ratio = outer_mean / inner_mean

    spectral_std = np.std(magnitude)

    return {
        "spectral_ratio": float(spectral_ratio),
        "spectral_std": float(spectral_std),
        "mid_freq_mean": float(mid_mean),
    }


def analyze_frequency(image: Image.Image) -> dict[str, float]:
    """Run full frequency analysis and return combined features."""
    dct_features = compute_dct_features(image)
    fft_features = compute_fft_features(image)

    combined = {**dct_features, **fft_features}

    dct_ratio = dct_features["dct_ratio"]
    spectral_ratio = fft_features["spectral_ratio"]

    dct_score = min(1.0, dct_ratio / 0.1)
    spectral_score = 1.0 - min(1.0, spectral_ratio / 0.5)

    freq_score = (dct_score + spectral_score) / 2
    combined["freq_score"] = float(freq_score)

    return combined


def frequency_suggests_ai(image: Image.Image, threshold: float = 0.6) -> bool:
    """Quick check if frequency analysis suggests AI generation."""
    features = analyze_frequency(image)
    return features["freq_score"] > threshold
