# Frequency/Spectral Analysis for AI-Generated Image Detection

**Date:** 2025-12-27
**Status:** Research complete
**Relevance:** Highly relevant - complements existing SigLIP-based detector

## Executive Summary

Frequency-domain analysis provides **complementary** detection capabilities to neural network classifiers like CLIP/ViT/SigLIP. The key finding: **hybrid approaches combining frequency features with semantic features achieve the best generalization and robustness**. Pure frequency methods are faster and more interpretable but less robust to compression/resizing. CLIP-based detectors generalize well but miss low-level artifacts. Fusion strategies combining both outperform either alone.

**Recommendation:** Add optional frequency-based features as a secondary signal, particularly for high-resolution uncompressed images where spectral artifacts are preserved.

---

## 1. GAN Fingerprints in Frequency Domain

### Core Insight

GANs leave distinctive artifacts in the frequency domain caused by **upsampling operations** (transposed convolutions, nearest-neighbor upsampling). These create:

- Periodic patterns visible in Fourier/DCT spectra
- Characteristic peaks at specific frequencies
- Differences in high-frequency energy distribution

### Key Papers & Implementations

| Paper                                                                   | Method                                    | Implementation                                                                                                    | Notes                      |
| ----------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------- |
| **Wavelet-Packets for Deepfake Detection** (ECML PKDD 2022)             | Wavelet packet decomposition + classifier | [gan-police/frequency-forensics](https://github.com/gan-police/frequency-forensics) (archived)                    | PyTorch, uses ptwt library |
| **Fighting Deepfakes by Detecting GAN DCT Anomalies** (J. Imaging 2021) | DCT coefficient beta statistics           | [olivergiudice/img-deepfake-detection-dct2](https://github.com/olivergiudice/img-deepfake-detection-dct2)         | Lightweight, explainable   |
| **Fourier-Based GAN Fingerprint Detection** (2025)                      | 2D DFT + ResNet50                         | [SaiTeja-Erukude/gan-fingerprint-detection-dft](https://github.com/SaiTeja-Erukude/gan-fingerprint-detection-dft) | StyleGAN2 focused          |
| **GANDCTAnalysis** (RUB-SysSec)                                         | DCT-based detection                       | Integrated in SIDBench                                                                                            | Classic baseline           |

### Technical Approach (DCT Method)

```python
# Core concept from Giudice et al.
import scipy.fftpack as fft

def extract_dct_features(image):
    # Convert to grayscale, apply 2D DCT
    gray = np.mean(image, axis=2)
    dct = fft.dct(fft.dct(gray.T, norm='ortho').T, norm='ortho')

    # Extract AC coefficients (exclude DC)
    ac_coeffs = dct[1:, 1:].flatten()

    # Fit generalized Gaussian distribution
    # Beta parameter differs between real and GAN images
    beta = fit_ggd(ac_coeffs)
    return beta
```

### Limitations for GANs

- **StyleGAN3** explicitly reduces aliasing, weakening frequency signatures
- Compression destroys high-frequency artifacts
- Resizing removes periodic patterns

---

## 2. Diffusion Model Spectral Artifacts

### Key Finding

Diffusion models exhibit **different** frequency artifacts than GANs:

- Progressive differences from real images across low-to-high frequency bands
- High frequencies show largest divergence
- Artifacts caused by denoising process, not upsampling

### Key Papers & Implementations

| Paper                                           | Method                                     | Venue              | Implementation                                        |
| ----------------------------------------------- | ------------------------------------------ | ------------------ | ----------------------------------------------------- |
| **FIRE: Frequency-Guided Reconstruction Error** | Reconstruction error in frequency bands    | CVPR 2025          | arXiv:2412.07140                                      |
| **SPAI: Spectral AI-Generated Image Detection** | Masked spectral learning + self-supervised | CVPR 2025          | [mever-team/spai](https://github.com/mever-team/spai) |
| **Synthbuster**                                 | Fourier spectrum analysis for diffusion    | HAL 2024           | Research paper                                        |
| **MaskSim**                                     | Masked spectrum similarity                 | CVPR 2024 Workshop | [Paper](https://hal.science/hal-04716636)             |

### SPAI Approach (State-of-the-Art)

- **Self-supervised**: Trains only on real images
- **Masked spectral learning**: Reconstructs missing frequency components
- **Spectral Reconstruction Similarity (SRS)**: Measures reconstruction divergence
- **Any-resolution**: Handles high-res images without resizing

**Performance (AUC across 13 generators):**
| Method | Average AUC |
|--------|-------------|
| SPAI | **91.0%** |
| RINE | 85.5% |
| DMID | 83.5% |
| PatchCraft | 80.4% |
| UnivFD | 67.3% |
| FreqDetect | 57.1% |

---

## 3. JPEG Compression Artifact Analysis

### Relevance

- JPEG uses 8x8 DCT blocks
- AI-generated images have inconsistent block artifacts
- Can detect manipulation/splicing

### Key Work

| Paper                                                | Focus                             | Notes                                      |
| ---------------------------------------------------- | --------------------------------- | ------------------------------------------ |
| **Learning JPEG Compression Artifacts** (IJCV 2022)  | Manipulation localization via DCT | Uses raw DCT coefficients as input         |
| **JPEG AI Impact on Forensics** (ICCV 2025 Workshop) | Neural compression effects        | JPEG AI confuses synthetic image detectors |

### Practical Consideration

Most online images are JPEG-compressed, which:

- Removes high-frequency GAN artifacts
- Adds compression artifacts that mask generation artifacts
- Requires robustness to varying quality factors (65-100)

---

## 4. Hybrid Approaches (Frequency + Neural)

### State-of-the-Art Hybrid Methods

| Method              | Architecture                                       | Key Innovation                                 |
| ------------------- | -------------------------------------------------- | ---------------------------------------------- |
| **Wavelet-CLIP**    | CLIP + DWT features                                | Best generalization to unseen diffusion models |
| **SpectraCLIP**     | CLIP + spectral branch                             | Merges vision and frequency features           |
| **WaViT-CDC**       | ViT + Wavelet + CDC                                | Central difference convolutions for edges      |
| **Dual-Branch CNN** | Spatial + Frequency branches                       | Siamese network for feature fusion             |
| **SCADET**          | Dynamic frequency attention + contrastive spectral | Art-focused detection                          |

### Wavelet-CLIP Results (Robustness to Unseen Diffusion)

| Model            | DDPM AUC  | DDIM AUC  | LDM AUC   | Avg AUC   |
| ---------------- | --------- | --------- | --------- | --------- |
| CLIP             | 0.781     | 0.879     | 0.876     | 0.845     |
| **Wavelet-CLIP** | **0.792** | **0.886** | **0.897** | **0.893** |
| Xception         | 0.712     | 0.729     | 0.658     | 0.699     |
| F3-Net           | 0.388     | 0.423     | 0.348     | 0.386     |

**Implementation:** [lalithbharadwajbaru/Wavelet-CLIP](https://github.com/lalithbharadwajbaru/Wavelet-CLIP)

---

## 5. CLIP-Based Detection (Current SOTA Baseline)

### Key Paper

**Raising the Bar of AI-generated Image Detection with CLIP** (CVPR 2024 Workshop)

### Findings

1. **Minimal training data needed**: 10-100 examples sufficient
2. **Generalization**: +6% AUC over previous SOTA on OOD data
3. **Robustness**: +13% on compressed/resized images
4. **Semantic features**: Largely independent of low-level frequency traces

### Critical Insight

> "CLIP features, even when adapted to forensic applications, are largely independent of low-level forensic traces."

This means **CLIP and frequency methods are complementary**.

### Fusion Strategy (from paper)

```
Decision: Image is REAL only if BOTH detectors agree
- CLIP-based detector score
- Frequency-based detector (Corvi et al.) score

Result: +3.6% AUC, +7.4% Accuracy improvement
```

---

## 6. Comparative Analysis

### When to Use Each Approach

| Scenario                  | Best Method               | Why                   |
| ------------------------- | ------------------------- | --------------------- |
| High-res uncompressed     | Frequency/Spectral        | Artifacts preserved   |
| Social media (compressed) | CLIP/ViT                  | Robust to JPEG        |
| Unknown generator         | SPAI or Hybrid            | Best generalization   |
| Real-time detection       | DCT statistics            | Fastest               |
| Explainability needed     | DCT/Wavelet               | Interpretable         |
| Maximum accuracy          | Fusion (CLIP + Frequency) | Complementary signals |

### Benchmark Comparison (SIDBench)

| Model      | Trained On               | Generalization     |
| ---------- | ------------------------ | ------------------ |
| CNNDetect  | ProGAN                   | Poor on diffusion  |
| FreqDetect | DCT features             | Moderate           |
| UnivFD     | ProGAN + CLIP            | Good               |
| DIRE       | Diffusion reconstruction | Diffusion-specific |
| RINE       | CLIP intermediate layers | Excellent          |

---

## 7. Practical Implementation Recommendations

### For ai-detect Project

#### Option A: Lightweight Frequency Feature (Recommended Start)

Add DCT beta statistics as a secondary signal:

```python
import numpy as np
from scipy.fftpack import dct

def compute_dct_score(image: np.ndarray) -> float:
    """Quick DCT-based anomaly score."""
    gray = np.mean(image, axis=2) if image.ndim == 3 else image

    # 2D DCT
    coeffs = dct(dct(gray.T, norm='ortho').T, norm='ortho')

    # Analyze high-frequency energy ratio
    h, w = coeffs.shape
    low_freq = np.abs(coeffs[:h//4, :w//4]).mean()
    high_freq = np.abs(coeffs[h//4:, w//4:]).mean()

    # AI images often have different high/low frequency ratio
    return high_freq / (low_freq + 1e-8)
```

#### Option B: Wavelet Features + Current Detector

Use PyWavelets for wavelet decomposition:

```python
import pywt

def extract_wavelet_features(image: np.ndarray) -> np.ndarray:
    """Extract wavelet packet features."""
    gray = np.mean(image, axis=2) if image.ndim == 3 else image

    # 3-level wavelet decomposition
    coeffs = pywt.wavedec2(gray, 'db4', level=3)

    # Extract statistics from each subband
    features = []
    for level in coeffs[1:]:  # Skip approximation
        for subband in level:
            features.extend([
                np.mean(np.abs(subband)),
                np.std(subband),
                np.max(np.abs(subband))
            ])
    return np.array(features)
```

#### Option C: Full Hybrid (SPAI-style)

- Requires training infrastructure
- Use SPAI codebase as reference
- Self-supervised on real images only

### Dependencies

```toml
# pyproject.toml additions
pywavelets = "^1.4.0"
scipy = "^1.11.0"  # for DCT
```

---

## 8. Key Repositories

| Repository                                                                                                      | Stars | Purpose                      | Status   |
| --------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------- | -------- |
| [mever-team/sidbench](https://github.com/mever-team/sidbench)                                                   | 47    | Benchmarking framework       | Active   |
| [mever-team/spai](https://github.com/mever-team/spai)                                                           | -     | CVPR 2025 spectral detection | Active   |
| [lalithbharadwajbaru/Wavelet-CLIP](https://github.com/lalithbharadwajbaru/Wavelet-CLIP)                         | 32    | Hybrid wavelet+CLIP          | Active   |
| [grip-unina/ClipBased-SyntheticImageDetection](https://grip-unina.github.io/ClipBased-SyntheticImageDetection/) | -     | CLIP baseline                | Active   |
| [gan-police/frequency-forensics](https://github.com/gan-police/frequency-forensics)                             | 50    | Wavelet packets              | Archived |

---

## 9. Conclusions

1. **Frequency analysis adds value** but is not a replacement for neural classifiers
2. **CLIP/ViT features are complementary** to frequency features - they capture different artifacts
3. **Hybrid approaches achieve best results** (+5-10% over single-method)
4. **Robustness is the key differentiator**: CLIP handles compression, frequency handles pristine images
5. **Self-supervised spectral methods (SPAI)** are the new SOTA for generalization

### Recommended Next Steps for ai-detect

1. **Quick win**: Add optional `--frequency` flag that computes DCT statistics
2. **Medium effort**: Integrate Wavelet-CLIP for improved diffusion detection
3. **Full integration**: Ensemble current SigLIP detector with spectral features

---

## References

1. Wolter et al. "Wavelet-Packets for Deepfake Image Analysis and Detection" ECML PKDD 2022
2. Giudice et al. "Fighting Deepfakes by Detecting GAN DCT Anomalies" J. Imaging 2021
3. Cozzolino et al. "Raising the Bar of AI-generated Image Detection with CLIP" CVPR 2024W
4. Karageorgiou et al. "Any-Resolution AI-Generated Image Detection by Spectral Learning" CVPR 2025
5. Chu et al. "FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error" CVPR 2025
6. Baru et al. "Wavelet-Driven Generalizable Framework for Deepfake Face Forgery Detection" arXiv 2024
7. Li et al. "MaskSim: Detection of synthetic images by masked spectrum similarity analysis" CVPR 2024W
