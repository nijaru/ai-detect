"""CLI interface for AI image detection."""

import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse
from urllib.request import urlretrieve

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .models import Detector, DetectionResult

logger = logging.getLogger("ai_detect")


def is_url(path: str) -> bool:
    """Check if path is a URL."""
    try:
        result = urlparse(path)
        return result.scheme in ("http", "https")
    except Exception:
        return False


def download_image(url: str) -> Path | None:
    """Download image from URL to temp file."""
    try:
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            urlretrieve(url, f.name)
            return Path(f.name)
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


app = typer.Typer(
    name="ai-detect",
    help="Detect AI-generated images.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
DEFAULT_THRESHOLD = 0.5

logger = logging.getLogger("ai_detect")


def collect_images(
    path: Path,
    recursive: bool,
    ai_dir: Path | None = None,
    real_dir: Path | None = None,
) -> list[Path]:
    """Collect image files from path, optionally excluding ai/real dirs."""
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path]
        return []

    pattern = path.rglob("*") if recursive else path.glob("*")
    images = [
        f for f in pattern if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if ai_dir and real_dir:
        images = [
            img
            for img in images
            if not (ai_dir in img.parents or real_dir in img.parents)
        ]

    return images


def load_image(path: Path) -> Image.Image | None:
    """Load and prepare an image for detection."""
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        err_console.print(f"[yellow]Warning: Could not load {path}: {e}[/yellow]")
        return None


def format_result(
    path: Path, result: DetectionResult, elapsed: float, threshold: float
) -> dict:
    """Format result as JSON-serializable dict."""
    return {
        "file": str(path),
        "verdict": "ai" if result.confidence >= threshold and result.is_ai else "real",
        "confidence": result.confidence,
        "scores": result.scores,
        "time": elapsed,
    }


def unique_path(dest: Path) -> Path:
    """Generate unique path by adding numeric suffix if file exists."""
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    counter = 1
    while True:
        new_dest = parent / f"{stem}_{counter}{suffix}"
        if not new_dest.exists():
            return new_dest
        counter += 1


FORMATS = ["text", "json", "table"]


def validate_threshold(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise typer.BadParameter("Must be between 0.0 and 1.0")
    return value


@app.command()
def main(
    ctx: typer.Context,
    path: Annotated[
        str | None, typer.Argument(help="Image file, directory, or URL to analyze")
    ] = None,
    # Common options
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Search directories recursively"),
    ] = False,
    sort: Annotated[
        bool,
        typer.Option(
            "--sort", "-s", help="Sort images into ai/ and real/ subdirectories"
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save results to JSON file"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text, json, table"),
    ] = "text",
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Confidence threshold (0.0-1.0)",
            callback=validate_threshold,
        ),
    ] = DEFAULT_THRESHOLD,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Preview sort without moving files"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-analyze images already in ai/ or real/"),
    ] = False,
    # Detection modes
    fast: Annotated[
        bool,
        typer.Option("--fast", help="Fast mode: full image only (~0.3s)"),
    ] = False,
    thorough: Annotated[
        bool,
        typer.Option("--thorough", help="Thorough mode: all methods + frequency"),
    ] = False,
    # Verbosity
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed logging"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-essential output"),
    ] = False,
    # Advanced options (hidden from main help)
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            "-b",
            help="Detection backend: clip or siglip",
            hidden=True,
        ),
    ] = "clip",
    segmenter: Annotated[
        str,
        typer.Option(
            "--segmenter",
            help="Segmentation backend: yolo or sa2va",
            hidden=True,
        ),
    ] = "yolo",
) -> None:
    """Detect AI-generated images.

    By default, uses smart detection that combines:
    - Full image analysis
    - Person detection (catches AI people on real backgrounds)
    - Strategic patches for large images (catches partial edits)

    Use --fast for speed (~0.3s) or --thorough for maximum accuracy.

    Supports local files, directories, and URLs.
    """
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    elif quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARNING)

    if path is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)

    if format not in FORMATS:
        err_console.print(
            f"[red]Error: Invalid format '{format}'. Choose from: {', '.join(FORMATS)}[/red]"
        )
        raise typer.Exit(1)

    if fast and thorough:
        err_console.print("[red]Error: Cannot use --fast and --thorough together[/red]")
        raise typer.Exit(1)

    # Handle URL input
    temp_file = None
    if is_url(path):
        if sort:
            err_console.print("[red]Error: --sort not supported with URLs[/red]")
            raise typer.Exit(1)
        if not quiet:
            err_console.print(f"Downloading {path}...")
        temp_file = download_image(path)
        if temp_file is None:
            err_console.print(f"[red]Error: Failed to download {path}[/red]")
            raise typer.Exit(1)
        local_path = temp_file
    else:
        local_path = Path(path)

    if not local_path.exists():
        err_console.print(f"[red]Error: Path does not exist: {local_path}[/red]")
        raise typer.Exit(1)

    if sort and not local_path.is_dir():
        err_console.print(
            f"[red]Error: --sort requires a directory: {local_path}[/red]"
        )
        raise typer.Exit(1)

    ai_dir = local_path / "ai" if sort else None
    real_dir = local_path / "real" if sort else None

    exclude_dirs = (ai_dir, real_dir) if sort and not force else (None, None)
    images = collect_images(local_path, recursive, *exclude_dirs)

    if not images:
        err_console.print(f"[yellow]No images found at {local_path}[/yellow]")
        if temp_file:
            temp_file.unlink(missing_ok=True)
        raise typer.Exit(0)

    # Determine mode
    if fast:
        mode = "fast"
        mode_desc = "fast"
    elif thorough:
        mode = "thorough"
        mode_desc = "thorough"
    else:
        mode = "smart"
        mode_desc = "smart"

    if not quiet:
        err_console.print(f"Loading detector ({mode_desc} mode)...")
    detector = Detector(backend=backend, mode=mode)
    detector.load()

    if sort:
        ai_dir.mkdir(exist_ok=True)
        real_dir.mkdir(exist_ok=True)

    all_results = []
    show_progress = len(images) > 1
    ai_count = 0
    real_count = 0
    skipped_count = 0
    moves = []

    iterator = tqdm(
        images, desc="Processing", disable=not show_progress or quiet, file=sys.stderr
    )
    for image_path in iterator:
        image = load_image(image_path)
        if image is None:
            skipped_count += 1
            continue

        start = time.time()
        result = detector.detect(image)
        elapsed = time.time() - start

        data = format_result(image_path, result, elapsed, threshold)
        is_ai = data["verdict"] == "ai"

        if sort:
            dest_dir = ai_dir if is_ai else real_dir

            if image_path.parent == dest_dir:
                if is_ai:
                    ai_count += 1
                else:
                    real_count += 1
                continue

            dest = unique_path(dest_dir / image_path.name)

            if dry_run:
                moves.append((image_path, dest, is_ai, data["confidence"]))
            else:
                try:
                    shutil.move(str(image_path), str(dest))
                except OSError as e:
                    err_console.print(
                        f"[yellow]Warning: Could not move {image_path}: {e}[/yellow]"
                    )
                    skipped_count += 1
                    continue

            if is_ai:
                ai_count += 1
            else:
                real_count += 1
        else:
            all_results.append(data)

            if format == "text" and not show_progress:
                verdict = data["verdict"].upper()
                color = "red" if is_ai else "green"
                conf_str = f"{data['confidence']:.0%}"
                # Show methods breakdown for single image
                methods = data["scores"].get("methods", {})
                if methods:
                    method_strs = [f"{k}:{v:.0%}" for k, v in methods.items()]
                    console.print(
                        f"[{color}]{verdict}[/{color}] ({conf_str}) [{', '.join(method_strs)}]"
                    )
                else:
                    console.print(f"[{color}]{verdict}[/{color}] ({conf_str})")
            elif format == "json" and not output:
                console.print_json(json.dumps(data))

    if sort:
        if dry_run and moves:
            console.print("\n[bold]Would move:[/bold]")
            for src, dest, is_ai, conf in sorted(
                moves, key=lambda x: x[3], reverse=True
            ):
                verdict = "ai" if is_ai else "real"
                color = "red" if is_ai else "green"
                console.print(
                    f"  {src.name} → [{color}]{verdict}/[/{color}] ({conf:.0%})"
                )

        action = "Would sort" if dry_run else "Sorted"
        summary = f"[green]{action} {ai_count} to ai/, {real_count} to real/[/green]"
        if skipped_count:
            summary += f" [yellow]({skipped_count} skipped)[/yellow]"
        console.print(summary)
    else:
        all_results.sort(key=lambda r: r["confidence"], reverse=True)

        if format == "text" and show_progress:
            for r in all_results:
                verdict = r["verdict"].upper()
                color = "red" if r["verdict"] == "ai" else "green"
                console.print(
                    f"{Path(r['file']).name}: [{color}]{verdict}[/{color}] ({r['confidence']:.0%})"
                )

        if format == "table" and all_results:
            table = Table(title="Detection Results")
            table.add_column("File", style="cyan")
            table.add_column("Verdict", style="bold")
            table.add_column("Confidence")
            table.add_column("Time")

            for r in all_results:
                color = "red" if r["verdict"] == "ai" else "green"
                table.add_row(
                    Path(r["file"]).name,
                    f"[{color}]{r['verdict'].upper()}[/{color}]",
                    f"{r['confidence']:.0%}",
                    f"{r['time']:.1f}s",
                )
            console.print(table)

        if output:
            output.write_text(json.dumps(all_results, indent=2))
            err_console.print(f"[green]Results saved to {output}[/green]")
        elif format == "json" and show_progress:
            console.print_json(json.dumps(all_results))

        if show_progress and format != "json" and not quiet:
            ai_count = sum(1 for r in all_results if r["verdict"] == "ai")
            console.print(
                f"\n[bold]Summary:[/bold] {ai_count}/{len(all_results)} AI-generated"
            )

    # Cleanup temp file if we downloaded from URL
    if temp_file:
        temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    app()
