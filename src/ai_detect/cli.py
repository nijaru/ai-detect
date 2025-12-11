"""CLI interface for AI image detection."""

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from tqdm import tqdm

from .consensus import ConsensusResult, compute_consensus
from .models import BaseDetector, DetectionResult, get_detectors

app = typer.Typer(
    name="ai-detect",
    help="Detect AI-generated images using ensemble of SOTA detectors.",
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

    if recursive:
        files = path.rglob("*")
    else:
        files = path.glob("*")

    return [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]


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


def run_detection(
    image: Image.Image,
    detectors: list[BaseDetector],
    verbose: bool = False,
) -> list[DetectionResult]:
    """Run all detectors on an image."""
    results = []
    for detector in detectors:
        try:
            result = detector.detect(image)
            results.append(result)
        except Exception as e:
            if verbose:
                console.print(f"[yellow]Warning: {detector.name} failed: {e}[/yellow]")
    return results


def format_result_tree(
    path: Path,
    consensus: ConsensusResult,
    processing_time: float,
) -> Tree:
    """Format result as a rich tree."""
    verdict = "AI-GENERATED" if consensus.is_ai else "REAL"
    verdict_color = "red" if consensus.is_ai else "green"

    tree = Tree(f"[bold]{path.name}[/bold]")
    tree.add(
        f"Verdict: [{verdict_color}]{verdict}[/{verdict_color}] "
        f"({consensus.confidence:.0%} confidence)"
    )
    tree.add(
        f"Agreement: {consensus.votes_ai}/{consensus.total_votes} models say AI "
        f"({consensus.agreement.value})"
    )

    breakdown = tree.add("Model breakdown:")
    for result in consensus.individual_results:
        vote = "AI" if result.is_ai else "Real"
        color = "red" if result.is_ai else "green"

        agrees = result.is_ai == consensus.is_ai
        marker = "" if agrees else " [yellow]<< disagrees[/yellow]"

        breakdown.add(
            f"{result.model_name}: [{color}]{vote}[/{color}] "
            f"({result.confidence:.2f}){marker}"
        )

    tree.add(f"Processing time: {processing_time:.1f}s")
    return tree


def format_result_json(
    path: Path,
    consensus: ConsensusResult,
    processing_time: float,
) -> dict:
    """Format result as JSON-serializable dict."""
    return {
        "file": str(path),
        "verdict": "ai" if consensus.is_ai else "real",
        "confidence": consensus.confidence,
        "agreement": consensus.agreement.value,
        "votes": {
            "ai": consensus.votes_ai,
            "real": consensus.votes_real,
        },
        "models": [
            {
                "name": r.model_name,
                "verdict": "ai" if r.is_ai else "real",
                "confidence": r.confidence,
                "raw_scores": r.raw_scores,
            }
            for r in consensus.individual_results
        ],
        "processing_time_seconds": processing_time,
    }


@app.command()
def detect(
    path: Annotated[Path, typer.Argument(help="Image file or directory to analyze")],
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Search directories recursively"),
    ] = False,
    models: Annotated[
        str,
        typer.Option("--models", "-m", help="Model set: 'all' or 'fast' (top 2 only)"),
    ] = "all",
    output: Annotated[
        str,
        typer.Option(
            "--output", "-o", help="Output format: 'tree', 'json', or 'table'"
        ),
    ] = "tree",
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Confidence threshold for AI verdict"),
    ] = 0.5,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed per-model breakdown"),
    ] = False,
    save: Annotated[
        Path | None,
        typer.Option("--save", "-s", help="Save detailed results to JSON file"),
    ] = None,
) -> None:
    """Detect AI-generated images using ensemble of detectors."""
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    images = collect_images(path, recursive)
    if not images:
        console.print(f"[yellow]No images found at {path}[/yellow]")
        raise typer.Exit(0)

    console.print(f"Loading {models} detectors...")
    detectors = get_detectors(mode=models)

    for detector in tqdm(detectors, desc="Loading models", disable=len(images) == 1):
        detector.load()

    all_results = []

    iterator = tqdm(images, desc="Processing", disable=len(images) == 1)
    for image_path in iterator:
        image = load_image(image_path)
        if image is None:
            continue

        start_time = time.time()
        results = run_detection(image, detectors, verbose)
        consensus = compute_consensus(results, threshold)
        processing_time = time.time() - start_time

        result_data = format_result_json(image_path, consensus, processing_time)
        all_results.append(result_data)

        if output == "tree":
            tree = format_result_tree(image_path, consensus, processing_time)
            console.print(tree)
            console.print()
        elif output == "json":
            console.print_json(json.dumps(result_data))

    if output == "table" and all_results:
        table = Table(title="Detection Results")
        table.add_column("File", style="cyan")
        table.add_column("Verdict", style="bold")
        table.add_column("Confidence")
        table.add_column("Agreement")
        table.add_column("Time")

        for r in all_results:
            verdict_style = "red" if r["verdict"] == "ai" else "green"
            table.add_row(
                Path(r["file"]).name,
                f"[{verdict_style}]{r['verdict'].upper()}[/{verdict_style}]",
                f"{r['confidence']:.0%}",
                f"{r['votes']['ai']}/{r['votes']['ai'] + r['votes']['real']} AI",
                f"{r['processing_time_seconds']:.1f}s",
            )

        console.print(table)

    if save:
        save.write_text(json.dumps(all_results, indent=2))
        console.print(f"[green]Results saved to {save}[/green]")

    ai_count = sum(1 for r in all_results if r["verdict"] == "ai")
    if len(all_results) > 1:
        console.print(
            f"\n[bold]Summary:[/bold] {ai_count}/{len(all_results)} images "
            f"detected as AI-generated"
        )


@app.command()
def list_models() -> None:
    """List available detection models."""
    from .models import DETECTORS, FAST_DETECTORS

    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Fast Mode", justify="center")

    for name, cls in DETECTORS.items():
        fast = "[green]yes[/green]" if name in FAST_DETECTORS else "no"
        table.add_row(name, cls.description, fast)

    console.print(table)


if __name__ == "__main__":
    app()
