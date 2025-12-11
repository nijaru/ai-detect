# ai-detect

Detect and sort AI-generated images. Optionally segments people to catch AI composites on real backgrounds.

## Quick Start

```bash
uv sync
uv run ai-detect photo.jpg           # Check single image
uv run ai-detect photos -r -s        # Sort directory into ai/ and real/
```

## Features

- Fast detection using SigLIP-based classifier
- Batch processing with progress bars
- Multiple output formats (text, json, table)
- **Subject segmentation** (`--subjects`): Analyzes people separately to catch AI-generated people on real backgrounds

## Usage

### Analyze

```bash
ai-detect photo.jpg                  # Single image
ai-detect photos                     # Directory
ai-detect photos -r                  # Recursive
ai-detect photos -f json             # JSON output
ai-detect photos -f table            # Table output
ai-detect photos -o results.json     # Save to file
ai-detect photos -t 0.7              # Custom threshold
```

### Sort

Move images into `ai/` and `real/` subdirectories:

```bash
ai-detect photos -s                  # Sort
ai-detect photos -rs                 # Recursive sort
ai-detect photos -sn                 # Dry run (preview)
ai-detect photos -s --force          # Re-analyze already sorted
ai-detect photos -s -t 0.7           # Custom threshold
```

### Subject Segmentation (--subjects)

> **Requires NVIDIA GPU (CUDA).** Will not work on CPU or Apple Silicon.

Segments people and analyzes each separately. Catches AI-generated people composited onto real photos. ~5x slower than default.

```bash
ai-detect photos --subjects          # Analyze with segmentation
ai-detect photos -rs --subjects      # Sort with segmentation
```

## Models

| Model                                                                                         | Purpose             | Size   |
| --------------------------------------------------------------------------------------------- | ------------------- | ------ |
| [Ateeqq/ai-vs-human-image-detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) | AI detection        | ~400MB |
| [ByteDance/Sa2VA-Qwen3-VL-1B](https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-1B)             | Person segmentation | ~5GB   |

## License

MIT
