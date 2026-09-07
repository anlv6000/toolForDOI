"""Entry point for Autonomous Academic Research Agent CLI.
Accepts user query and base paper DOI, validates environment,
executes the LangGraph workflow, and renders synthesized output.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure the research_agent package directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

load_dotenv(dotenv_path=CURRENT_DIR / ".env")
load_dotenv()  # Fallback to root .env if present

from graph import build_research_graph, clean_doi
from tools import find_local_pdf

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def print_banner():
    if HAS_RICH:
        console.print(
            Panel.fit(
                "[bold cyan]Autonomous Academic Research Agent[/bold cyan]\n"
                "[dim]Powered by LangGraph, ChromaDB, PyMuPDF, Unpaywall & Gemini 1.5 Flash[/dim]",
                border_style="cyan",
            )
        )
    else:
        print("=" * 60)
        print("     AUTONOMOUS ACADEMIC RESEARCH AGENT")
        print("=" * 60)


def validate_environment() -> bool:
    """Verifies that mandatory environment variables are set."""
    google_key = os.getenv("GOOGLE_API_KEY")
    unpaywall_email = os.getenv("UNPAYWALL_EMAIL")

    errors = []
    if not google_key or google_key == "your_google_api_key_here":
        errors.append("GOOGLE_API_KEY is not set. Get one from https://aistudio.google.com/")
    if not unpaywall_email or "example.com" in unpaywall_email:
        errors.append("UNPAYWALL_EMAIL must be a real email address (e.g., yourname@domain.com) for Unpaywall API access.")

    if errors:
        if HAS_RICH:
            console.print("[bold red]Configuration Error(s):[/bold red]")
            for err in errors:
                console.print(f"  [yellow]*[/yellow] {err}")
            console.print(
                "\n[dim]Please configure these in your [bold].env[/bold] file in the project folder.[/dim]\n"
            )
        else:
            print("Configuration Error(s):")
            for err in errors:
                print(f"  - {err}")
            print("\nPlease configure these in your .env file.\n")
        return False
    return True


def run_agent(doi: str, query: str):
    """Executes the research agent graph with the given DOI and query."""
    print_banner()

    if not validate_environment():
        sys.exit(1)

    doi = clean_doi(doi)
    if HAS_RICH:
        table = Table(title="Research Task Parameters", border_style="blue")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_row("Base DOI", doi)
        table.add_row("Research Query", query)
        console.print(table)
    else:
        print(f"\n[Task] Base DOI: {doi}")
        print(f"[Task] Query   : {query}\n")

    initial_state = {
        "user_query": query,
        "base_doi": doi,
        "current_dois": [],
        "visited_dois": [],
        "hop_count": 0,
        "accumulated_context": "",
        "status": "CONTINUE",
        "next_search_keywords": "",
        "eval_reasoning": "",
        "final_answer": "",
    }

    graph = build_research_graph()

    if HAS_RICH:
        console.print("\n[bold green]Starting Autonomous Research Loop...[/bold green]\n")

    try:
        final_state = graph.invoke(initial_state)

        # Output Results
        final_answer = final_state.get("final_answer", "No synthesis generated.")
        analyzed_dois = final_state.get("current_dois", [])

        if HAS_RICH:
            console.print("\n" + "=" * 70)
            console.print(Panel(
                Markdown(final_answer),
                title="[bold green]Final Research Synthesis[/bold green]",
                border_style="green",
                expand=False,
            ))

            doi_summary = ", ".join(analyzed_dois) if analyzed_dois else "None"
            console.print(f"\n[dim cyan]Total Literature Examined:[/dim cyan] {doi_summary}")
            console.print(f"[dim cyan]Total Hops Executed:[/dim cyan] {final_state.get('hop_count', 0)}\n")
        else:
            print("\n" + "=" * 70)
            print("FINAL RESEARCH SYNTHESIS")
            print("=" * 70)
            print(final_answer)
            print("-" * 70)
            print(f"Examined DOIs: {', '.join(analyzed_dois)}")
            print(f"Hops: {final_state.get('hop_count', 0)}\n")

    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[bold red]Execution failed:[/bold red] {e}")
        else:
            print(f"\nExecution failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Academic Research Agent (LangGraph + ChromaDB + PyMuPDF)"
    )
    parser.add_argument(
        "--doi",
        type=str,
        help="Base paper DOI (e.g., 10.1038/s41586-020-2649-2)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Research question to investigate",
    )

    args = parser.parse_args()

    doi = args.doi
    query = args.query

    # Interactive prompt if flags are not supplied
    if not doi:
        print_banner()
        local_pdfs = list((CURRENT_DIR / "PDF").glob("*.pdf"))
        if len(local_pdfs) == 1:
            doi = local_pdfs[0].stem
            print(f"Using the only local PDF found: {local_pdfs[0].name}")
        try:
            if not doi:
                doi = input("Enter base paper DOI (e.g. 10.1038/s41586-020-2649-2): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(0)

    if not query:
        try:
            query = input("Enter research question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(0)

    if not doi or not query:
        print("[Error] Both DOI and research question are required to run the agent.")
        sys.exit(1)

    run_agent(doi=doi, query=query)


if __name__ == "__main__":
    main()
