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


OUTPUT_DIR = CURRENT_DIR / "outputs"


def output_filename(doi: str, report_number: int = 1) -> str:
    """Create a filesystem-safe output filename while retaining the DOI identity."""
    cleaned = clean_doi(doi)
    safe_doi = "".join(character if character.isalnum() or character in ".-_" else "_" for character in cleaned)
    return f"{safe_doi}_{report_number}.txt"


def _safe_doi(doi: str) -> str:
    cleaned = clean_doi(doi)
    return "".join(character if character.isalnum() or character in ".-_" else "_" for character in cleaned)


def create_run_output_dir(doi: str) -> tuple[Path, int]:
    """Reserve a numbered folder for one DOI/question run."""
    safe_doi = _safe_doi(doi)
    doi_dir = OUTPUT_DIR / safe_doi
    doi_dir.mkdir(parents=True, exist_ok=True)
    existing_numbers = []
    for path in doi_dir.glob(f"{safe_doi}_*.txt"):
        try:
            existing_numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    report_number = max(existing_numbers, default=0) + 1
    run_dir = doi_dir / f"{safe_doi}_{report_number}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, report_number


def save_synthesis(doi: str, query: str, final_answer: str, analyzed_dois=None, hop_count=0, output_dir=None, report_number=1) -> Path:
    """Save a synthesis and its run metadata as a readable text report."""
    if output_dir is None:
        output_dir, report_number = create_run_output_dir(doi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(doi, report_number)
    examined = ", ".join(analyzed_dois or []) or clean_doi(doi)
    report = (
        f"Base DOI: {clean_doi(doi)}\n"
        f"Research Query: {query}\n"
        f"Examined DOIs: {examined}\n"
        f"Hops: {hop_count}\n"
        f"Report Number: {report_number}\n"
        f"\n{'=' * 80}\n"
        f"FINAL RESEARCH SYNTHESIS\n"
        f"{'=' * 80}\n\n"
        f"{final_answer.rstrip()}\n"
    )
    output_path.write_text(report, encoding="utf-8")
    return output_path

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
        "citation_network": [],
        "evidence_pool": [],
        "final_draft": "",
        "generated_files": {},
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
    run_output_dir, report_number = create_run_output_dir(doi)
    initial_state["run_output_dir"] = str(run_output_dir)
    initial_state["report_number"] = report_number

    if HAS_RICH:
        console.print("\n[bold green]Starting Autonomous Research Loop...[/bold green]\n")

    try:
        final_state = graph.invoke(initial_state)

        # Output Results
        final_answer = final_state.get("final_answer", "No synthesis generated.")
        analyzed_dois = final_state.get("current_dois", [])
        generated_files = final_state.get("generated_files", {})
        output_path = save_synthesis(
            doi,
            query,
            final_answer,
            analyzed_dois,
            final_state.get("hop_count", 0),
            output_dir=run_output_dir,
            report_number=report_number,
        )

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
            console.print(f"[dim cyan]Saved Report:[/dim cyan] {output_path}")
            for artifact_type, artifact_path in generated_files.items():
                console.print(f"[dim cyan]Artifact {artifact_type}:[/dim cyan] {artifact_path}")
        else:
            print("\n" + "=" * 70)
            print("FINAL RESEARCH SYNTHESIS")
            print("=" * 70)
            print(final_answer)
            print("-" * 70)
            print(f"Examined DOIs: {', '.join(analyzed_dois)}")
            print(f"Hops: {final_state.get('hop_count', 0)}\n")
            print(f"Saved report: {output_path}")
            for artifact_type, artifact_path in generated_files.items():
                print(f"Artifact {artifact_type}: {artifact_path}")

        return final_answer, output_path

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
