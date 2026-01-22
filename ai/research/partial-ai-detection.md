# Detecting Partial/Localized AI Edits in Images

Research on methods for detecting when only part of an image is AI-generated (inpainting, face swaps, object replacement).

## Key Scenarios

1. Real photo with AI-generated face swap
2. Real background with AI-inserted object
3. AI inpainting to remove/add elements
4. Text-guided local edits (DALL-E inpainting, Stable Diffusion inpainting)

---

## Recommended Approaches

### 1. TruFor (CVPR 2023) - Best Overall

**Architecture:** Transformer-based fusion combining RGB + learned noise fingerprint (Noiseprint++)

**How it works:**

- Extracts high-level semantic features from RGB
- Extracts low-level noise patterns via self-supervised Noiseprint++ (trained only on real images)
- Fuses both streams to detect anomalies (forgeries deviate from expected noise patterns)
- Outputs: pixel-level localization map, image-level integrity score, reliability map

**Strengths:**

- Generalizes well to unseen manipulation types
- Detects both classic edits and AI-generated content
- Training code now available (March 2025)

**Code:** https://github.com/grip-unina/TruFor
**Weights:** Available via their repo
**License:** Academic/non-commercial

```python
# Example usage pattern
from trufor import TruFor
model = TruFor.load_pretrained()
result = model.predict(image)
# result.localization_map: H x W probability map
# result.integrity_score: float 0-1
# result.reliability_map: confidence per pixel
```

---

### 2. IMDL-BenCo (NeurIPS 2024 Spotlight) - Best Framework

**What it is:** Comprehensive benchmark and codebase for Image Manipulation Detection & Localization

**Includes:**

- 8 SOTA models fully implemented
- Pretrained checkpoints available
- Standardized training/evaluation protocols
- 15 GPU-accelerated metrics

**Installation:**

```bash
pip install imdlbenco
```

**Implemented Models:**

- CAT-Net (compression artifacts)
- ManTraNet (manipulation tracing)
- SPAN (spatial pyramid attention)
- MVSSNet
- ObjectFormer
- PSCC-Net
- IML-ViT
- Mesorch (AAAI 2025)

**Code:** https://github.com/scu-zjz/IMDLBenCo
**Docs:** https://scu-zjz.github.io/IMDLBenCo-doc/

---

### 3. SAFIRE (AAAI 2025) - Novel Point-Prompt Approach

**Architecture:** SAM-style point prompting for forgery segmentation

**How it works:**

- Given a point on the image, segments the source region containing that point
- Can partition images into multiple source regions (first to achieve this)
- Focuses on uniform characteristics within each source region rather than memorizing forgery traces

**Strengths:**

- Interactive - can query specific regions
- Handles multi-source composites
- More interpretable than binary segmentation

**Code:** https://github.com/mjkwon2021/SAFIRE

---

### 4. FakeShield (ICLR 2025) - Explainable MLLM Approach

**Architecture:** Multi-modal LLM (based on LLaVA-like architecture)

**Outputs:**

- Authenticity evaluation
- Tampered region masks
- Natural language explanation of tampering clues

**Strengths:**

- Explainable - tells you WHY it thinks something is fake
- Generalizes across Photoshop, DeepFake, AIGC-Editing
- Leverages GPT-4o for training data enhancement

**Code:** https://github.com/zhipeixu/FakeShield

---

### 5. ManTraNet (CVPR 2019) - Classic Baseline

**Architecture:** Two-stage network

1. Image Manipulation Trace Feature Extractor (385-class manipulation classifier)
2. Local Anomaly Detection Network (compares local vs. reference features)

**Strengths:**

- End-to-end, no pre/post-processing needed
- Accepts arbitrary image sizes
- Fast inference

**Limitations:**

- Keras/TF 1.x (dated)
- No training code released

**Code:** https://github.com/ISICV/ManTraNet
**Colab:** Available in repo

---

### 6. CAT-Net (WACV 2021) - Compression Artifacts

**Architecture:** Dual-stream CNN (RGB + DCT coefficients)

**How it works:**

- Detects JPEG compression inconsistencies
- DCT stream pretrained on double-JPEG detection
- Multi-resolution to handle various object sizes

**Best for:** Traditional splicing where source/target have different compression histories

**Code:** https://github.com/mjkwon2021/CAT-Net

---

## Patch-Based Detection Strategies

### Sliding Window Approach

For adapting whole-image detectors to localized detection:

```python
def sliding_window_detection(image, detector, patch_size=224, stride=112):
    """
    Run detector on overlapping patches and aggregate.
    """
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            score = detector.predict(patch)  # AI probability
            heatmap[y:y+patch_size, x:x+patch_size] += score
            counts[y:y+patch_size, x:x+patch_size] += 1

    return heatmap / np.maximum(counts, 1)
```

### Aggregation Strategies

| Method               | Description                        | When to Use                                      |
| -------------------- | ---------------------------------- | ------------------------------------------------ |
| **Max pooling**      | Take highest patch score           | Conservative - any suspicious region flags image |
| **Mean pooling**     | Average all patch scores           | Balanced - overall assessment                    |
| **Voting**           | % patches exceeding threshold      | Robust to outliers                               |
| **Weighted by size** | Weight patches by manipulated area | When size matters                                |
| **Top-K mean**       | Average of K highest scores        | Balance between max and mean                     |

```python
def aggregate_patches(patch_scores, method='top_k_mean', k=5, threshold=0.5):
    if method == 'max':
        return np.max(patch_scores)
    elif method == 'mean':
        return np.mean(patch_scores)
    elif method == 'voting':
        return np.mean(patch_scores > threshold)
    elif method == 'top_k_mean':
        return np.mean(np.sort(patch_scores)[-k:])
```

### Patch Forensics (ECCV 2020)

**Key insight:** Small patches (e.g., 32x32 to 128x128) are sufficient to detect GAN artifacts

**Finding:** Artifacts concentrate in specific semantic regions (eyes, hair, teeth for faces)

**Code:** https://github.com/chail/patch-forensics

---

## Segmentation-Based Approaches

### Current Project Approach (Segment Subjects)

Your existing `segment.py` using Sa2VA is a valid approach:

1. Segment people/objects
2. Run AI detector on each segment
3. Flag if any segment is AI-generated

**Limitations:**

- Only works for known object classes (people)
- Misses background manipulations
- Requires CUDA

### Alternative: Use Localization Models Directly

Models like TruFor, SAFIRE output pixel-level masks directly - no need for separate segmentation.

---

## Practical Implementation Recommendations

### For Your Project (ai-detect)

**Option A: Add TruFor as secondary detector**

```python
# When --localize flag is set
if args.localize:
    trufor = TruForModel()
    result = trufor.predict(image)
    if result.integrity_score < threshold:
        # Show localization heatmap
        visualize_heatmap(image, result.localization_map)
```

**Option B: Patch-based with existing detector**

```python
# Extend current detector with sliding window
if args.localize:
    heatmap = sliding_window_detection(image, detector)
    ai_regions = heatmap > threshold
    # Aggregate for image-level verdict
    image_score = aggregate_patches(heatmap.flatten(), 'top_k_mean')
```

**Option C: Use IMDL-BenCo models**

```bash
pip install imdlbenco
# Use pretrained CAT-Net or ObjectFormer for localization
```

### Model Selection Guide

| Scenario                     | Recommended Model   |
| ---------------------------- | ------------------- |
| General-purpose localization | TruFor              |
| Face swaps specifically      | ManTraNet or TruFor |
| JPEG splicing                | CAT-Net             |
| Need explanations            | FakeShield          |
| Interactive exploration      | SAFIRE              |
| Benchmark/compare models     | IMDL-BenCo          |

---

## Available Pretrained Weights

| Model             | Weights Available   | Format   | Size       |
| ----------------- | ------------------- | -------- | ---------- |
| TruFor            | Yes                 | PyTorch  | ~500MB     |
| IMDL-BenCo models | Yes (Baidu NetDisk) | PyTorch  | Varies     |
| ManTraNet         | Yes                 | Keras/TF | ~50MB      |
| CAT-Net           | Yes                 | PyTorch  | ~200MB     |
| SAFIRE            | Yes                 | PyTorch  | TBD        |
| FakeShield        | Yes                 | PyTorch  | ~7B params |

---

## Key Papers

1. **TruFor** (CVPR 2023): "Leveraging all-round clues for trustworthy image forgery detection and localization"
2. **IMDL-BenCo** (NeurIPS 2024): "A Comprehensive Benchmark and Codebase for Image Manipulation Detection & Localization"
3. **SAFIRE** (AAAI 2025): "Segment Any Forged Image Region"
4. **FakeShield** (ICLR 2025): "Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models"
5. **ManTraNet** (CVPR 2019): "Manipulation Tracing Network For Detection And Localization of Image Forgeries"
6. **CAT-Net** (WACV 2021): "Compression Artifact Tracing Network for Detection and Localization of Image Splicing"
7. **Patch Forensics** (ECCV 2020): "What makes fake images detectable? Understanding properties that generalize"

---

## Related Work

- **DIRE** (ICCV 2023): Diffusion Reconstruction Error for detecting diffusion-generated images
- **UniversalFakeDetect**: CLIP-based universal fake detector (https://github.com/WisconsinAIVision/UniversalFakeDetect)
- **ForensicHub**: All-domain fake detection codebase (https://github.com/scu-zjz/ForensicHub)

---

_Research compiled: 2024-12-27_
