# ai-detect

Detect AI-generated images using state-of-the-art CLIP-based detection.

## Quick Start

```bash
uv sync
uv run ai-detect photo.jpg           # Analyze single image
uv run ai-detect photos/ -r -s       # Sort directory into ai/ and real/
```

## How It Works

By default, ai-detect uses **smart detection** that automatically:

1. Analyzes the full image
2. Detects people and analyzes each separately (catches AI faces on real backgrounds)
3. Samples strategic patches on large images (catches partial edits/inpainting)

Returns the **maximum AI confidence** across all methods.

## Usage

```bash
ai-detect photo.jpg                  # Smart detection (recommended)
ai-detect photo.jpg --fast           # Full image only (~0.3s)
ai-detect photo.jpg --thorough       # All methods + frequency analysis
```

### Batch Processing

```bash
ai-detect photos/                    # Analyze directory
ai-detect photos/ -r                 # Recursive
ai-detect photos/ -f table           # Table output
ai-detect photos/ -o results.json    # Save to JSON
```

### Sort Mode

Move images into `ai/` and `real/` subdirectories:

```bash
ai-detect photos/ -s                 # Sort
ai-detect photos/ -rs                # Recursive sort
ai-detect photos/ -sn                # Dry run (preview)
ai-detect photos/ -s --force         # Re-analyze already sorted
```

## Detection Modes

| Mode         | Speed  | What it does                                   |
| ------------ | ------ | ---------------------------------------------- |
| (default)    | ~1-3s  | Full image + people + patches for large images |
| `--fast`     | ~0.3s  | Full image only                                |
| `--thorough` | ~5-10s | All methods + dense patches + frequency        |

## Example Output

Single image with smart detection:

```
AI (94%) [full_image:72%, person_1:94%, person_2:31%]
```

Batch processing:

```
Processing: 100%|████████████| 50/50 [01:23<00:00]
photo1.jpg: AI (94%)
photo2.jpg: REAL (89%)
...
Summary: 12/50 AI-generated
```

## Models

| Component | Model                    | Size |
| --------- | ------------------------ | ---- |
| Detection | GRIP-UNINA CLIP ViT-L/14 | ~2GB |
| People    | YOLO11n                  | ~6MB |

First run downloads models automatically.

## Advanced Options

```bash
ai-detect photo.jpg --backend siglip    # Use SigLIP instead of CLIP
ai-detect photo.jpg --segmenter sa2va   # Use Sa2VA segmenter (CUDA only)
ai-detect photo.jpg -t 0.7              # Custom threshold
```

## License

MIT
