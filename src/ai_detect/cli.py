"""CLI interface for AI image detection."""

import json
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
) -> None:
    """Detect AI-generated images."""
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    images = collect_images(path, recursive)
    if not images:
        console.print(f"[yellow]No images found at {path}[/yellow]")
        raise typer.Exit(0)

    console.print("Loading model...")
    detector = Detector()
    detector.load()

    all_results = []
    show_progress = len(images) > 1

    iterator = tqdm(images, desc="Processing", disable=not show_progress)
    for image_path in iterator:
        image = load_image(image_path)
        if image is None:
            continue

        start = time.time()
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


if __name__ == "__main__":
    app()
