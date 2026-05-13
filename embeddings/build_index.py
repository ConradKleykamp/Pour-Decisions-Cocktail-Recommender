import json
import sys
from pathlib import Path

# Adding the project root to sys.path so config.py is importable when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import typer
from rich.console import Console
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDINGS_INDEX_DIR,
    EMBEDDINGS_PATH,
    INSTRUCTIONS_MAX_CHARS,
    METADATA_PATH,
    RAW_COCKTAILS_PATH,
)

app = typer.Typer()
console = Console()


def _build_document(cocktail: dict) -> str:
    # Constructing a natural-language sentence from structured cocktail fields
    name = cocktail.get("name") or "Unknown"
    category = cocktail.get("category") or "cocktail"
    glass = cocktail.get("glass") or "a glass"
    ingredients = ", ".join(cocktail.get("ingredients") or [])
    instructions = cocktail.get("instructions") or ""

    doc = f"{name}. A {category} served in a {glass}."
    if ingredients:
        doc += f" Made with {ingredients}."
    if instructions:
        # Appending truncated instructions to capture flavor and style language
        truncated = instructions[:INSTRUCTIONS_MAX_CHARS].rsplit(" ", 1)[0]
        doc += f" {truncated}"
    return doc


def _extract_metadata(cocktail: dict) -> dict:
    # Extracting the fields needed for search result display
    return {
        "name": cocktail.get("name"),
        "category": cocktail.get("category"),
        "glass": cocktail.get("glass"),
        "ingredients": cocktail.get("ingredients", []),
        "instructions": cocktail.get("instructions"),
    }


@app.command()
def main(
    force: bool = typer.Option(False, "--force", help="Rebuild index even if it already exists."),
) -> None:
    # Skipping build if both index files already exist and --force is not set
    if EMBEDDINGS_PATH.exists() and METADATA_PATH.exists() and not force:
        console.print(
            f"[yellow]Index already exists at {EMBEDDINGS_INDEX_DIR}. "
            "Pass --force to rebuild.[/yellow]"
        )
        raise typer.Exit()

    # Verifying that the data cache exists before proceeding
    if not RAW_COCKTAILS_PATH.exists():
        console.print(
            "[red]Cache not found.[/red] "
            "Run [bold]python data/fetch.py[/bold] first."
        )
        raise typer.Exit(code=1)

    EMBEDDINGS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Loading cocktail records from the JSON cache
    with open(RAW_COCKTAILS_PATH, encoding="utf-8") as f:
        cocktails = json.load(f)

    console.print(f"[bold]Building index for [cyan]{len(cocktails)}[/cyan] cocktails...[/bold]")

    # Building composite document strings for each cocktail
    documents = [_build_document(c) for c in cocktails]

    # Loading the sentence-transformers model
    with console.status(f"[bold]Loading model [cyan]{EMBEDDING_MODEL}[/cyan]...[/bold]"):
        model = SentenceTransformer(EMBEDDING_MODEL)
    console.print(f"[green]Model loaded.[/green]")

    # Embedding all composite documents into dense vectors
    with console.status(f"[bold]Embedding {len(documents)} documents...[/bold]"):
        embeddings = model.encode(documents, show_progress_bar=False)
    console.print(f"[green]Embeddings computed.[/green]")

    # Saving embeddings array to disk as .npy
    np.save(EMBEDDINGS_PATH, embeddings)

    # Saving cocktail metadata as a list of dicts to disk as .pkl
    metadata = [_extract_metadata(c) for c in cocktails]
    joblib.dump(metadata, METADATA_PATH)

    console.print(
        f"[bold green]Done.[/bold green] Saved embeddings and metadata to [dim]{EMBEDDINGS_INDEX_DIR}[/dim]"
    )


if __name__ == "__main__":
    app()
