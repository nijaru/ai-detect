"""CLI interface for AI image detection."""

import json
import shutil
import time
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .models import Detector, DetectionResult

app = typer.Typer(
    name="ai-detect",
    help="Detect AI-generated images.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


def collect_images(path: Path, recursive: bool = False) -> list[Path]:
    """Collect image files from path."""
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path]
        return []

    pattern = path.rglob("*") if recursive else path.glob("*")
    return [f for f in pattern if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]


def load_image(path: Path) -> Image.Image | None:
    """Load and prepare an image for detection."""
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load {path}: {e}[/yellow]")
        return None


def format_result(path: Path, result: DetectionResult, elapsed: float) -> dict:
    """Format result as JSON-serializable dict."""
    return {
        "file": str(path),
        "verdict": "ai" if result.is_ai else "real",
        "confidence": result.confidence,
        "scores": result.scores,
        "time": elapsed,
    }


def detect_with_subjects(
    image: "Image.Image",
    detector: Detector,
    segmenter: "PersonSegmenter",
) -> DetectionResult:
    """Detect AI by analyzing segmented people in the image.

    If any person is detected as AI, the whole image is flagged.
    Falls back to full image detection if no people are found.
    """
    crops = segmenter.segment_people(image)

    if not crops:
        # No people found, fall back to full image
        return detector.detect(image)

    # Check each person crop
    max_ai_score = 0.0
    max_human_score = 0.0

    for crop in crops:
        result = detector.detect(crop.image)
        ai_score = result.scores.get("ai", 0.0)
        human_score = result.scores.get("hum", result.scores.get("human", 0.0))
        max_ai_score = max(max_ai_score, ai_score)
        max_human_score = max(max_human_score, human_score)

    # If any person is AI, flag the whole image
    is_ai = max_ai_score > max_human_score
    confidence = max_ai_score if is_ai else max_human_score

    return DetectionResult(
        is_ai=is_ai,
        confidence=confidence,
        scores={"ai": max_ai_score, "hum": max_human_score, "subjects": len(crops)},
    )


@app.command()
def detect(
    path: Annotated[Path, typer.Argument(help="Image file or directory to analyze")],
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Search directories recursively"),
    ] = False,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output format: text, json, table"),
    ] = "text",
    save: Annotated[
        Path | None,
        typer.Option("--save", "-s", help="Save results to JSON file"),
    ] = None,
    subjects: Annotated[
        bool,
        typer.Option(
            "--subjects", help="Segment and analyze people separately (CUDA only)"
        ),
    ] = False,
) -> None:
    """Detect AI-generated images."""
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    images = collect_images(path, recursive)
    if not images:
        console.print(f"[yellow]No images found at {path}[/yellow]")
        raise typer.Exit(0)

    err_console.print("Loading AI detector...")
    detector = Detector()
    detector.load()

    segmenter = None
    if subjects:
        err_console.print("Loading segmentation model (Sa2VA-Qwen3-VL-1B)...")
        from .segment import PersonSegmenter

        segmenter = PersonSegmenter()
        segmenter.load()

    all_results = []
    show_progress = len(images) > 1

    iterator = tqdm(images, desc="Processing", disable=not show_progress)
    for image_path in iterator:
        image = load_image(image_path)
        if image is None:
            continue

        start = time.time()
        if segmenter:
            result = detect_with_subjects(image, detector, segmenter)
        else:
            result = detector.detect(image)
        elapsed = time.time() - start

        data = format_result(image_path, result, elapsed)
        all_results.append(data)

        if output == "text" and not show_progress:
            verdict = "AI" if result.is_ai else "REAL"
            color = "red" if result.is_ai else "green"
            console.print(f"[{color}]{verdict}[/{color}] ({result.confidence:.0%})")
        elif output == "json":
            console.print_json(json.dumps(data))

    if output == "text" and show_progress:
        for r in all_results:
            verdict = r["verdict"].upper()
            color = "red" if r["verdict"] == "ai" else "green"
            console.print(
                f"{Path(r['file']).name}: [{color}]{verdict}[/{color}] ({r['confidence']:.0%})"
            )

    if output == "table" and all_results:
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

    if save:
        save.write_text(json.dumps(all_results, indent=2))
        console.print(f"[green]Results saved to {save}[/green]")

    if show_progress:
        ai_count = sum(1 for r in all_results if r["verdict"] == "ai")
        console.print(
            f"\n[bold]Summary:[/bold] {ai_count}/{len(all_results)} AI-generated"
        )


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


@app.command()
def sort(
    path: Annotated[Path, typer.Argument(help="Directory containing images to sort")],
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Search directories recursively"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Re-analyze images already in ai/ or real/"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Preview changes without moving files"),
    ] = False,
    subjects: Annotated[
        bool,
        typer.Option(
            "--subjects", help="Segment and analyze people separately (CUDA only)"
        ),
    ] = False,
) -> None:
    """Detect and sort images into ai/ and real/ subdirectories."""
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[red]Error: Path must be a directory: {path}[/red]")
        raise typer.Exit(1)

    ai_dir = path / "ai"
    real_dir = path / "real"

    images = collect_images(path, recursive)
    if not force:
        # Exclude images already in ai/ or real/ output directories
        images = [
            img
            for img in images
            if not (ai_dir in img.parents or real_dir in img.parents)
        ]

    if not images:
        console.print(f"[yellow]No images found at {path}[/yellow]")
        raise typer.Exit(0)

    err_console.print("Loading AI detector...")
    detector = Detector()
    detector.load()

    segmenter = None
    if subjects:
        err_console.print("Loading segmentation model (Sa2VA-Qwen3-VL-1B)...")
        from .segment import PersonSegmenter

        segmenter = PersonSegmenter()
        segmenter.load()

    ai_dir.mkdir(exist_ok=True)
    real_dir.mkdir(exist_ok=True)

    ai_count = 0
    real_count = 0
    skipped_count = 0

    moves = []  # Track moves for dry-run output

    for image_path in tqdm(images, desc="Processing"):
        image = load_image(image_path)
        if image is None:
            skipped_count += 1
            continue

        if segmenter:
            result = detect_with_subjects(image, detector, segmenter)
        else:
            result = detector.detect(image)
        dest_dir = ai_dir if result.is_ai else real_dir

        # Skip if already in correct directory
        if image_path.parent == dest_dir:
            if result.is_ai:
                ai_count += 1
            else:
                real_count += 1
            continue

        dest = unique_path(dest_dir / image_path.name)

        if dry_run:
            moves.append((image_path, dest, result.is_ai, result.confidence))
        else:
            try:
                shutil.move(str(image_path), str(dest))
            except OSError as e:
                console.print(
                    f"[yellow]Warning: Could not move {image_path}: {e}[/yellow]"
                )
                skipped_count += 1
                continue

        if result.is_ai:
            ai_count += 1
        else:
            real_count += 1

    if dry_run and moves:
        console.print("\n[bold]Would move:[/bold]")
        for src, dest, is_ai, conf in moves:
            verdict = "ai" if is_ai else "real"
            color = "red" if is_ai else "green"
            console.print(f"  {src.name} → [{color}]{verdict}/[/{color}] ({conf:.0%})")

    action = "Would sort" if dry_run else "Sorted"
    summary = f"[green]{action} {ai_count} to ai/, {real_count} to real/[/green]"
    if skipped_count:
        summary += f" [yellow]({skipped_count} skipped)[/yellow]"
    console.print(summary)


if __name__ == "__main__":
    app()
