# AI-Generated Image Detection Models - Late 2025 Research

## Executive Summary

The field has consolidated around **CLIP-based and Vision Transformer architectures** as the dominant approach for generalizable AI image detection. Key challenges remain: robustness to compression/social media artifacts, and generalization to newest generators (Flux, GPT-4o, Gemini).

## Ranked Model Recommendations

### Tier 1: Production-Ready with Strong Generalization

#### 1. GRIP-UNINA CLIP-Based Detector (CVPRW 2024)

- **Paper**: "Raising the Bar of AI-generated Image Detection with CLIP"
- **Architecture**: CLIP ViT-L/14 with lightweight linear probe
- **GitHub**: https://github.com/grip-unina/ClipBased-SyntheticImageDetection
- **Weights**: Available in repo

| Pros                                                 | Cons                                      |
| ---------------------------------------------------- | ----------------------------------------- |
| State-of-the-art generalization to unseen generators | Requires only small training set          |
| Robust to JPEG compression and resizing              | May need retraining for newest generators |
| Lightweight - single linear layer on CLIP features   | Not specifically tested on Flux/GPT-4o    |
| Well-documented, reproducible                        |                                           |

**Tested on**: DALL-E 3, Midjourney v5, Firefly, SD variants
**Accuracy**: ~90%+ on in-distribution, strong OOD performance

---

#### 2. B-Free (CVPR 2025) - GRIP-UNINA

- **Paper**: "A Bias-Free Training Paradigm for More General AI-generated Image Detection"
- **GitHub**: https://github.com/grip-unina/b-free
- **Architecture**: Bias-free training paradigm, works with multiple backbones

| Pros                                                | Cons                                    |
| --------------------------------------------------- | --------------------------------------- |
| Addresses dataset bias problem directly             | Newer, less battle-tested               |
| Generates fakes from real images using conditioning | Requires more complex training pipeline |
| Improved generalization over prior CLIP methods     |                                         |
| CVPR 2025 - cutting edge                            |                                         |

**Key Innovation**: Ensures semantic alignment between real/fake during training

---

#### 3. SuSy (HPAI-BSC) - Spatial-Based Detection

- **HuggingFace**: https://huggingface.co/HPAI-BSC/SuSy
- **Architecture**: Custom spatial analysis model

| Pros                                                         | Cons                             |
| ------------------------------------------------------------ | -------------------------------- |
| **96.46% Recall on Flux.1-dev** - rare explicit Flux support | Lower recall on DALL-E 2 (20.7%) |
| 88.6% on DALL-E 3                                            | Less documentation available     |
| Strong on SD variants (85-95%)                               |                                  |
| Includes generator attribution                               |                                  |

**Explicit Flux Support**: One of few models with published Flux performance

---

#### 4. DRCT - Diffusion Reconstruction Contrastive Training (ICML 2024 Spotlight)

- **Paper**: "DRCT: Diffusion Reconstruction Contrastive Training"
- **GitHub**: https://github.com/beibuwandeluori/DRCT
- **Dataset**: DRCT-2M (16 diffusion model types)

| Pros                                               | Cons                                     |
| -------------------------------------------------- | ---------------------------------------- |
| Universal framework for diffusion-generated images | Focused on diffusion models specifically |
| Million-scale training dataset available           | Computationally intensive training       |
| Works with ConvNeXt and CLIP-ViT-L backbones       |                                          |
| >25% AUC improvement over baselines                |                                          |

---

### Tier 2: Specialized or Newer Models

#### 5. C2P-CLIP (AAAI 2025)

- **Paper**: "Injecting Category Common Prompt in CLIP"
- **GitHub**: https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection

| Pros                                 | Cons                            |
| ------------------------------------ | ------------------------------- |
| 12.41% improvement over vanilla CLIP | Primarily tested on deepfakes   |
| 93.79% mAcc on 20 generator models   | More complex prompt engineering |
| Uses LoRA for efficient fine-tuning  |                                 |

---

#### 6. SPAI - Spectral AI Detection (CVPR 2025)

- **Paper**: "Any-Resolution AI-Generated Image Detection by Spectral Learning"
- **GitHub**: https://github.com/mever-team/spai
- **Weights**: Available in repo

| Pros                                   | Cons                                          |
| -------------------------------------- | --------------------------------------------- |
| Resolution-agnostic (handles any size) | Newer approach, less validated                |
| Self-supervised spectral learning      | Spectral analysis may fail on some generators |
| 5.5% AUC improvement over prior SOTA   |                                               |
| Tested on 13 recent generators         |                                               |

---

#### 7. OmniAID (arXiv 2025)

- **Paper**: "Decoupling Semantic and Artifacts for Universal Detection"
- **GitHub**: https://github.com/yunncheng/OmniAID
- **Weights**: checkpoint_mirage.pth (recommended)

| Pros                                              | Cons                      |
| ------------------------------------------------- | ------------------------- |
| Mixture-of-Experts architecture                   | Very new (November 2025)  |
| Decouples semantic flaws from generator artifacts | Less community validation |
| Designed for "in the wild" detection              |                           |

---

#### 8. FakeVLM (NeurIPS 2025)

- **Paper**: "Spot the Fake: Large Multimodal Model-Based Detection"
- **HuggingFace**: https://huggingface.co/papers/2503.14905

| Pros                                            | Cons                          |
| ----------------------------------------------- | ----------------------------- |
| Provides natural language artifact explanations | Large model, slower inference |
| Multimodal approach (LMM-based)                 | Requires more compute         |
| Interpretable outputs                           |                               |

---

### Tier 3: HuggingFace Community Models

#### 9. CommunityForensics-DeepfakeDet-ViT

- **HuggingFace**: https://huggingface.co/aiwithoutborders-xyz/CommunityForensics-DeepfakeDet-ViT
- **Architecture**: ViT-Small (36M params)
- **Training**: 2.7M samples, 4,800+ generators

| Pros                        | Cons                                |
| --------------------------- | ----------------------------------- |
| 97.2% accuracy (unverified) | Accuracy not independently verified |
| Massive training diversity  | Community model, less rigorous eval |
| Small, fast model           |                                     |

---

#### 10. SigLIP2-Based Detectors

- **Deepfake-Detect-Siglip2**: https://huggingface.co/prithivMLmods/Deepfake-Detect-Siglip2
- **AIorNot-SigLIP2**: https://huggingface.co/prithivMLmods/AIorNot-SigLIP2
- **OpenSDI-Flux.1-SigLIP2**: https://huggingface.co/prithivMLmods/OpenSDI-Flux.1-SigLIP2

| Pros                                  | Cons                              |
| ------------------------------------- | --------------------------------- |
| Modern SigLIP2 backbone (2025)        | Community models, varying quality |
| 91.66% on Flux.1 (OpenSDI variant)    | Not peer-reviewed                 |
| 94.44% overall (deepfake-detector-v1) | Training details less clear       |
| Easy to use via transformers          |                                   |

---

## Robustness Considerations

### Models with Explicit Robustness Testing

| Model           | JPEG     | Resize           | Blur     | Social Media |
| --------------- | -------- | ---------------- | -------- | ------------ |
| GRIP-UNINA CLIP | Strong   | Strong           | Moderate | Good         |
| B-Free          | Strong   | Strong           | Strong   | Unknown      |
| DRCT            | Moderate | Unknown          | Unknown  | Unknown      |
| SPAI            | Strong   | Native (any-res) | Unknown  | Unknown      |

### Key Research on Robustness

1. **So-Fake Benchmark** (2025): Specifically tests social media artifacts
2. **RRDataset/RRBench** (ICCV 2025): Real-world robustness including re-digitization
3. **JPEG AI Impact** (ICCV 2025): Neural compression creates new forensic challenges

---

## Coverage of Latest Generators

| Generator            | Best Detection Options                           |
| -------------------- | ------------------------------------------------ |
| **Flux.1**           | SuSy (96.46%), OpenSDI-Flux.1-SigLIP2 (91.66%)   |
| **Midjourney v6/v7** | GRIP-UNINA CLIP, DRCT, GenImage-trained models   |
| **DALL-E 3**         | SuSy (88.6%), GRIP-UNINA CLIP, FakeInversion     |
| **SD 3.5**           | DRCT (DRCT-2M dataset), B-Free                   |
| **GPT-4o**           | Limited data - likely CLIP-based generalize best |
| **Gemini**           | No specific testing found                        |

---

## Recommended Implementation Strategy

### Quick Start (Highest ROI)

1. **GRIP-UNINA CLIP**: Best balance of performance/simplicity
   ```bash
   git clone https://github.com/grip-unina/ClipBased-SyntheticImageDetection
   ```

### For Flux-Specific Detection

2. **SuSy** or **OpenSDI-Flux.1-SigLIP2**: Explicit Flux training

### For Maximum Generalization

3. **B-Free + DRCT**: Combine bias-free training with contrastive learning

### For Production with Interpretability

4. **FakeVLM**: When you need to explain WHY an image is fake

---

## Key Datasets for Training/Evaluation

| Dataset             | Size    | Generators          | Notes                 |
| ------------------- | ------- | ------------------- | --------------------- |
| DRCT-2M             | 2M+     | 16 diffusion types  | Best for diffusion    |
| GenImage            | 1M+     | SD, MJ, BigGAN, etc | Cross-generator eval  |
| UniversalFakeDetect | Large   | 19 generators       | CVPR 2023 benchmark   |
| So-Fake             | Varies  | Multiple            | Social media focus    |
| RRDataset           | Varies  | Multiple            | Real-world robustness |
| MIRAGE              | Unknown | Modern generators   | Used by OmniAID       |

---

## Architecture Trends (2024-2025)

1. **CLIP/SigLIP dominance**: Pre-trained VLMs provide best generalization
2. **Parameter-efficient fine-tuning**: LoRA, LN-tuning preferred over full fine-tuning
3. **Contrastive learning**: DRCT-style approaches for artifact learning
4. **Mixture of Experts**: OmniAID uses MoE for semantic/artifact decoupling
5. **Spectral analysis**: SPAI introduces frequency-domain learning

---

## References

- GRIP-UNINA CLIP: https://arxiv.org/abs/2312.00195
- B-Free: https://github.com/grip-unina/b-free (CVPR 2025)
- DRCT: https://proceedings.mlr.press/v235/chen24ay.html (ICML 2024)
- C2P-CLIP: https://arxiv.org/abs/2408.09647 (AAAI 2025)
- SPAI: https://github.com/mever-team/spai (CVPR 2025)
- OmniAID: https://arxiv.org/abs/2511.08423
- FakeVLM: https://arxiv.org/abs/2503.14905 (NeurIPS 2025)
- UniversalFakeDetect: https://github.com/WisconsinAIVision/UniversalFakeDetect
- GenImage: https://genimage-dataset.github.io/

---

_Research compiled: 2025-12-27_
