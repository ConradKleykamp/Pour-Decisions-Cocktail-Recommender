import json
import string

import pandas as pd
import requests
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from config import API_BASE_URL, DATA_CACHE_DIR, RAW_COCKTAILS_PATH

# Defining ingredient column names as they appear in TheCocktailDB responses
INGREDIENT_COLS = [f"strIngredient{i}" for i in range(1, 16)]

# Mapping raw API field names to clean output field names
FIELD_MAP = {
    "idDrink": "id",
    "strDrink": "name",
    "strCategory": "category",
    "strGlass": "glass",
    "strInstructions": "instructions",
}

app = typer.Typer()
console = Console()


def _normalize_ingredients(row: pd.Series) -> list[str]:
    # Collecting non-null ingredient fields into a flat list
    return [
        row[col].strip()
        for col in INGREDIENT_COLS
        if col in row and pd.notna(row[col]) and str(row[col]).strip()
    ]


def _fetch_all_cocktails() -> list[dict]:
    # Fetching cocktails from TheCocktailDB by iterating over every letter a-z
    records: list[dict] = []
    letters = list(string.ascii_lowercase)

    with Progress(
        TextColumn("[bold cyan]  {task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching cocktails", total=len(letters))

        for letter in letters:
            response = requests.get(
                f"{API_BASE_URL}/search.php",
                params={"f": letter},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("drinks"):
                records.extend(data["drinks"])

            progress.advance(task)

    return records


@app.command()
def main(
    force: bool = typer.Option(False, "--force", help="Re-fetch even if cache exists."),
) -> None:
    # Skipping fetch if cache already exists and --force is not set
    if RAW_COCKTAILS_PATH.exists() and not force:
        console.print(
            f"[yellow]Cache already exists at {RAW_COCKTAILS_PATH}. "
            "Pass --force to re-fetch.[/yellow]"
        )
        raise typer.Exit()

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Fetching cocktail data from TheCocktailDB...[/bold]")
    raw_records = _fetch_all_cocktails()

    # Loading records into a DataFrame for deduplication and normalization
    df = pd.DataFrame(raw_records)
    before = len(df)
    df = df.drop_duplicates(subset="idDrink").reset_index(drop=True)
    dupes = before - len(df)

    # Normalizing strIngredient1–15 columns into a single list per cocktail
    df["ingredients"] = df.apply(_normalize_ingredients, axis=1)

    # Selecting and renaming fields for clean output
    keep = list(FIELD_MAP.keys()) + ["ingredients"]
    df = df[[col for col in keep if col in df.columns]].rename(columns=FIELD_MAP)

    cocktails = df.to_dict(orient="records")

    # Saving normalized cocktail records to the JSON cache
    with open(RAW_COCKTAILS_PATH, "w", encoding="utf-8") as f:
        json.dump(cocktails, f, indent=2, ensure_ascii=False)

    console.print(
        f"[bold green]Done.[/bold green] Saved [cyan]{len(cocktails)}[/cyan] cocktails"
        + (f" (removed {dupes} duplicates)" if dupes else "")
        + f" to [dim]{RAW_COCKTAILS_PATH}[/dim]"
    )


if __name__ == "__main__":
    app()
