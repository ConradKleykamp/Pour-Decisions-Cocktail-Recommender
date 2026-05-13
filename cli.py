from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from config import DEFAULT_TOP_K
from recommender.search import search

app = typer.Typer(add_completion=False)
console = Console()


def _render_result(result: dict, rank: int) -> Panel:
    # Formatting ingredients as a bullet list
    ingredients = "\n".join(
        f"  • {ing}" for ing in result.get("ingredients") or []
    ) or "  —"

    instructions = result.get("instructions") or "No instructions available."

    content = (
        f"[dim]Match score:[/dim]  [bold]{result['score']:.3f}[/bold]\n"
        f"[dim]Category:[/dim]     {result.get('category') or '—'}\n"
        f"[dim]Glass:[/dim]        {result.get('glass') or '—'}\n\n"
        f"[bold]Ingredients[/bold]\n{ingredients}\n\n"
        f"[bold]Instructions[/bold]\n{instructions}"
    )

    return Panel(
        content,
        title=f"[bold cyan]#{rank}  {result['name']}[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


def _run_query(query: str, top_k: int) -> None:
    # Running a single query against the embedding index and displaying results
    try:
        with console.status("[bold]Searching...[/bold]", spinner="dots"):
            results = search(query, top_k)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    if not results:
        console.print("[yellow]No results found. Try a different query.[/yellow]")
        return

    for rank, result in enumerate(results, start=1):
        console.print(_render_result(result, rank))


def _run_repl(top_k: int) -> None:
    # Running the interactive REPL loop until the user types exit or quit
    console.print(
        Panel(
            "[bold]Pour Decisions[/bold] — Cocktail Recommender\n"
            "[dim]Describe a mood, vibe, ingredient, or occasion.[/dim]\n"
            "[dim]Type [bold]exit[/bold] or [bold]quit[/bold] to leave.[/dim]",
            border_style="magenta",
            padding=(1, 2),
        )
    )

    while True:
        try:
            query = console.input("\n[bold magenta]>[/bold magenta] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            break

        _run_query(query, top_k)

    console.print("[dim]Goodbye.[/dim]")


@app.command()
def main(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Single query string (non-interactive)."),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", "-k", help="Number of results to return."),
) -> None:
    # Routing to single-query mode or interactive REPL depending on flags
    if query:
        _run_query(query, top_k)
    else:
        _run_repl(top_k)


if __name__ == "__main__":
    app()
