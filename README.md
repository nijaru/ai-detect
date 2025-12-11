# ai-detect

Detect and sort AI-generated images. Optionally segments people to catch AI composites on real backgrounds.

## Quick Start

```bash
uv sync
uv run ai-detect detect photo.jpg   # Check single image
uv run ai-detect sort photos -r     # Sort directory into ai/ and real/
```

## Features

- Fast detection using SigLIP-based classifier
- Batch processing with progress bars
- Multiple output formats (text, json, table)
- **Subject segmentation** (`-S`): Analyzes people separately to catch AI-generated people on real backgrounds

## Usage

### detect

Analyze without moving files:

```bash
uv run ai-detect detect photo.jpg
uv run ai-detect detect photos -r             # Recursive
uv run ai-detect detect photos -f json        # JSON output
uv run ai-detect detect photos -f table       # Table output
uv run ai-detect detect photos -o out.json    # Save to file
uv run ai-detect detect photos -t 0.7         # Custom threshold
```

### sort

Analyze and move into `ai/` and `real/` subdirectories:

```bash
uv run ai-detect sort photos
uv run ai-detect sort photos -r       # Recursive
uv run ai-detect sort photos -f       # Re-analyze already sorted
uv run ai-detect sort photos -n       # Dry run (preview)
uv run ai-detect sort photos -t 0.7   # Custom threshold
```

### Subject Segmentation (--subjects)

> **Requires NVIDIA GPU (CUDA).** Will not work on CPU or Apple Silicon.

Segments people and analyzes each separately. Catches AI-generated people composited onto real photos. ~5x slower than default.

```bash
uv run ai-detect sort photos --subjects
uv run ai-detect sort photos -r -f --subjects   # Re-sort with segmentation
```

## Models

| Model                                                                                         | Purpose             | Size   |
| --------------------------------------------------------------------------------------------- | ------------------- | ------ |
| [Ateeqq/ai-vs-human-image-detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) | AI detection        | ~400MB |
| [ByteDance/Sa2VA-Qwen3-VL-1B](https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-1B)             | Person segmentation | ~5GB   |

## License

MIT
