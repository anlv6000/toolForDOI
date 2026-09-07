"""Tools for academic literature retrieval, PDF parsing, and reference graph discovery.
Interacts with Unpaywall API, Semantic Scholar API, and PyMuPDF.
"""

import os
import io
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Common headers to mimic standard browser requests and avoid 403 blocks from academic hosts
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/json,*/*",
}


def clean_doi(doi: str) -> str:
    """Cleans and standardizes a DOI string."""
    doi = doi.strip()
    if doi.lower().startswith("https://doi.org/"):
        doi = doi[16:]
    elif doi.lower().startswith("http://doi.org/"):
        doi = doi[15:]
    elif doi.lower().startswith("doi:"):
        doi = doi[4:].strip()
    return doi


def find_local_pdf(doi: str, pdf_dir: Optional[Path] = None) -> Optional[Path]:
    """Find a locally downloaded PDF whose filename is derived from a DOI."""
    cleaned_doi = clean_doi(doi)
    pdf_dir = pdf_dir or Path(__file__).resolve().parent / "PDF"
    if not pdf_dir.is_dir():
        return None

    candidates = {
        cleaned_doi.lower(),
        cleaned_doi.replace("/", "").lower(),
        cleaned_doi.replace("/", "_").lower(),
    }
    for path in pdf_dir.glob("*.pdf"):
        if path.stem.lower() in candidates:
            return path
    return None


def get_unpaywall_pdf_link(doi: str) -> Optional[str]:
    """Retrieves an open-access PDF link for a given DOI from Unpaywall.

    Args:
        doi: The paper's DOI.

    Returns:
        Direct URL string to the PDF if available, otherwise None.
    """
    cleaned_doi = clean_doi(doi)
    email = os.getenv("UNPAYWALL_EMAIL", "").strip()

    if not email or "example.com" in email:
        print("[tools] Warning: UNPAYWALL_EMAIL is missing or using example.com. Unpaywall requires a valid email format.")
        # Fallback to an agent identifier email if not set
        email = email or "academic.agent@domain.org"

    url = f"https://api.unpaywall.org/v2/{cleaned_doi}"
    params = {"email": email}

    try:
        response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=15)
        if response.status_code == 404:
            print(f"[tools] Paper DOI '{cleaned_doi}' not found in Unpaywall.")
            return None
        elif response.status_code == 422:
            print(f"[tools] Unpaywall rejected email '{email}' (422 Unprocessable Entity).")
            return None
        response.raise_for_status()

        data = response.json()
        if not data.get("is_oa", False):
            print(f"[tools] Paper '{cleaned_doi}' is not marked as Open Access on Unpaywall.")
            return None

        # 1. Try best_oa_location
        best_oa = data.get("best_oa_location")
        if best_oa and best_oa.get("url_for_pdf"):
            return best_oa["url_for_pdf"]
        if best_oa and best_oa.get("url", "").lower().endswith(".pdf"):
            return best_oa["url"]

        # 2. Check other oa_locations
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
            if loc.get("url", "").lower().endswith(".pdf"):
                return loc["url"]

        # 3. Fallback to best OA landing page if available
        if best_oa and best_oa.get("url"):
            return best_oa["url"]

        print(f"[tools] No direct PDF link found in Unpaywall for '{cleaned_doi}'.")
        return None

    except requests.RequestException as e:
        print(f"[tools] Error fetching Unpaywall data for DOI '{cleaned_doi}': {e}")
        return None


def _parse_and_index_pdf(content: bytes, doi: str, source: str) -> bool:
    """Parse PDF bytes and index the extracted text for a DOI."""
    cleaned_doi = clean_doi(doi)
    try:
        import fitz  # PyMuPDF
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from .vector_store import add_paper_to_db
    except ImportError:
        import fitz
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from vector_store import add_paper_to_db

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = [page.get_text() for page in doc if page.get_text()]
        full_text = "\n".join(text_parts).strip()
        page_count = len(doc)
        doc.close()

        if not full_text:
            print(f"[tools] Warning: No readable text extracted from PDF for '{cleaned_doi}'.")
            return False

        print(
            f"[tools] Extracted {len(full_text)} characters across "
            f"{page_count} pages for '{cleaned_doi}' from {source}."
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(full_text)
        print(f"[tools] Split into {len(chunks)} chunks.")
        success = add_paper_to_db(cleaned_doi, chunks)
        if success:
            print(f"[tools] Successfully indexed paper '{cleaned_doi}' in ChromaDB.")
            return True
        print(f"[tools] Failed to index paper '{cleaned_doi}' in ChromaDB.")
        return False
    except Exception as e:
        print(f"[tools] Error parsing PDF for DOI '{cleaned_doi}': {e}")
        return False


def parse_local_pdf(pdf_path: Path, doi: str) -> bool:
    """Parse and index a locally downloaded PDF."""
    cleaned_doi = clean_doi(doi)
    print(f"[tools] Processing local PDF for DOI '{cleaned_doi}': {pdf_path}")
    try:
        return _parse_and_index_pdf(pdf_path.read_bytes(), cleaned_doi, "local file")
    except OSError as e:
        print(f"[tools] Error reading local PDF '{pdf_path}': {e}")
        return False


def download_and_parse_pdf(pdf_url: str, doi: str) -> bool:
    """Downloads a PDF from a URL, extracts text using PyMuPDF (fitz),
    chunks it, and stores the chunks into ChromaDB.

    Args:
        pdf_url: URL to download the PDF.
        doi: DOI associated with this paper.

    Returns:
        bool: True if downloaded, parsed, and indexed successfully; False otherwise.
    """
    cleaned_doi = clean_doi(doi)

    print(f"[tools] Downloading PDF for DOI '{cleaned_doi}' from: {pdf_url}")
    try:
        response = requests.get(
            pdf_url,
            headers=HTTP_HEADERS,
            timeout=30,
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()

        # Check content type or length
        content = response.content
        if len(content) < 1000:
            # Likely an error page or captcha
            print(f"[tools] Downloaded file is too small ({len(content)} bytes), likely not a PDF.")
            return False

        return _parse_and_index_pdf(content, cleaned_doi, "download")

    except Exception as e:
        print(f"[tools] Error downloading/parsing PDF for DOI '{cleaned_doi}': {e}")
        return False


def get_semantic_scholar_references(doi: str, max_refs: int = 10) -> List[Dict[str, Any]]:
    """Fetches reference papers for a given DOI using the Semantic Scholar Graph API.

    Args:
        doi: DOI of the paper to query.
        max_refs: Maximum number of references to return (default 10).

    Returns:
        List of dicts: [{"doi": str, "title": str, "abstract": str}, ...]
    """
    cleaned_doi = clean_doi(doi)
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()

    url = f"https://api.semanticscholar.org/graph/v1/paper/{cleaned_doi}"
    params = {
        "fields": "references.title,references.abstract,references.externalIds,references.openAccessPdf"
    }
    headers = dict(HTTP_HEADERS)
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        if response.status_code == 429:
            print("[tools] Semantic Scholar API rate limit hit (429). Waiting 3 seconds...")
            time.sleep(3)
            response = requests.get(url, params=params, headers=headers, timeout=20)

        if response.status_code == 404:
            print(f"[tools] Paper '{cleaned_doi}' not found in Semantic Scholar.")
            return []

        response.raise_for_status()
        data = response.json()

        raw_refs = data.get("references", [])
        if not raw_refs:
            print(f"[tools] No references found for paper '{cleaned_doi}'.")
            return []

        parsed_refs: List[Dict[str, Any]] = []
        for ref in raw_refs:
            if not ref or not isinstance(ref, dict):
                continue
            ext_ids = ref.get("externalIds") or {}
            ref_doi = ext_ids.get("DOI")
            ref_title = ref.get("title") or "Untitled Paper"
            ref_abstract = ref.get("abstract") or ""
            ref_oa = ref.get("openAccessPdf") or {}
            ref_oa_url = ref_oa.get("url") if isinstance(ref_oa, dict) else None

            # Only include papers with a valid DOI
            if ref_doi:
                parsed_refs.append({
                    "doi": ref_doi.strip(),
                    "title": ref_title.strip(),
                    "abstract": ref_abstract.strip(),
                    "pdf_url": ref_oa_url,
                })
            if len(parsed_refs) >= max_refs:
                break

        print(f"[tools] Retrieved {len(parsed_refs)} valid reference(s) with DOIs for '{cleaned_doi}'.")
        return parsed_refs

    except requests.RequestException as e:
        print(f"[tools] Semantic Scholar API error for DOI '{cleaned_doi}': {e}")
        return []
