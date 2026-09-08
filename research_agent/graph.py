"""LangGraph workflow definition for Autonomous Academic Research Agent.
Implements the multi-hop reasoning loop between base paper processing,
evaluation, reference exploration via Semantic Scholar, and final synthesis.
"""

import json
import os
from typing import Any, Dict, List, Optional, TypedDict
from urllib import response
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

try:
    from .tools import (
        clean_doi,
        find_local_pdf,
        parse_local_pdf,
        get_unpaywall_pdf_link,
        download_and_parse_pdf,
        get_semantic_scholar_references,
        export_citation_network,
        export_evidence_matrix,
        export_bibtex,
        extract_arxiv_id,
    )
    from .vector_store import query_db, add_paper_to_db
except ImportError:
    from tools import (
        clean_doi,
        find_local_pdf,
        parse_local_pdf,
        get_unpaywall_pdf_link,
        download_and_parse_pdf,
        get_semantic_scholar_references,
        export_citation_network,
        export_evidence_matrix,
        export_bibtex,
        extract_arxiv_id,
    )
    from vector_store import query_db, add_paper_to_db

load_dotenv()


class AgentState(TypedDict):
    user_query: str
    base_doi: str
    citation_network: List[Dict[str, Any]]
    evidence_pool: List[Dict[str, Any]]
    final_draft: str
    generated_files: Dict[str, str]
    current_dois: List[str]
    visited_dois: List[str]
    hop_count: int
    accumulated_context: str
    status: str
    next_search_keywords: str
    eval_reasoning: str
    final_answer: str
    run_output_dir: str
    report_number: int


def extract_text(resp_content) -> str:
    """Safely extracts text string from LLM response content (supports both str and list)."""
    if isinstance(resp_content, str):
        return resp_content
    if isinstance(resp_content, list):
        parts = []
        for item in resp_content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(resp_content)


def get_llm():
    """Initializes ChatGoogleGenerativeAI with active Gemini flash model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set in environment or .env file. "
            "Please obtain an API key from https://aistudio.google.com/"
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=api_key,
    )


def generate_gap_question(base_doi: str) -> str:
    """Generate one research-gap question grounded in the paper's related work."""
    cleaned_doi = clean_doi(base_doi)
    search_queries = [
        "related work prior methods citations limitations future work",
        "references existing methods unresolved challenge gap",
        "multi-task learning lesion segmentation image super-resolution DR grading limitations",
    ]
    snippets = []
    seen = set()
    for search_query in search_queries:
        for result in query_db(search_query, k=6, doi_filter=cleaned_doi):
            text = result.get("text", "").strip()
            if text and text not in seen:
                snippets.append(text)
                seen.add(text)

    if not snippets:
        raise ValueError(
            f"No indexed citation context found for {cleaned_doi}. "
            "Run the base-paper processing first."
        )

    llm = get_llm()
    citation_context = "\n\n".join(f"Snippet {index}: {text}" for index, text in enumerate(snippets, 1))
    prompt = f"""You are a rigorous research-gap analyst.
Read only the related-work, cited-method, limitation, and future-work evidence below from one base paper.
Generate exactly ONE strong research question that targets an explicit unresolved gap in those citations.

Rules:
- Do not assume that the base paper already proposes the solution.
- Do not force the topic of deferral unless the cited evidence supports a missing decision, uncertainty, or clinical workflow mechanism.
- Prefer a question that compares, measures, or designs a concrete missing capability.
- The question must identify the task, setting, and evaluable outcome.
- Return plain text only: one question ending with '?'.

Base DOI: {cleaned_doi}
Related citation evidence:
{citation_context}
"""
    response = llm.invoke(prompt)
    question = extract_text(response.content).strip()
    question = question.replace("```", "").strip()
    if "?" in question:
        question = question[:question.find("?") + 1]
    if not question:
        raise ValueError("The model returned an empty gap question.")
    return question


# ----------------------------------------------------------------------
# Node 1: node_process_base
# ----------------------------------------------------------------------
def node_process_base(state: AgentState) -> Dict[str, Any]:
    """Processes the base paper: fetches PDF via Unpaywall, parses into VectorDB,
    retrieves initial context for the user query.
    """
    base_doi = clean_doi(state["base_doi"])
    user_query = state["user_query"]
    print(f"\n[node_process_base] === Processing Base Paper: {base_doi} ===")

    current_dois = list(state.get("current_dois") or [])
    visited_dois = list(state.get("visited_dois") or [])
    visited_dois.append(base_doi)

    parsed_success = False

    # Prefer a DOI-named PDF already downloaded by the user, including paywalled files.
    local_pdf = find_local_pdf(base_doi)
    if local_pdf:
        parsed_success = parse_local_pdf(local_pdf, base_doi)

    # 1. Fetch PDF link from Unpaywall only when no usable local PDF exists.
    pdf_url = get_unpaywall_pdf_link(base_doi) if not parsed_success else None

    if pdf_url:
        print(f"[node_process_base] Found OA PDF link: {pdf_url}")
        parsed_success = download_and_parse_pdf(pdf_url, base_doi)

    # Fallback to Semantic Scholar if Unpaywall failed or no OA link
    if not parsed_success:
        print(f"[node_process_base] Unpaywall PDF unavailable. Attempting Semantic Scholar fallback for {base_doi}...")
        try:
            import requests
            s2_headers = {}
            s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
            if s2_key:
                s2_headers["x-api-key"] = s2_key
            arxiv_id = extract_arxiv_id(base_doi)
            s2_identifier = f"ARXIV:{arxiv_id}" if arxiv_id else base_doi
            s2_resp = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/{s2_identifier}?fields=title,abstract,openAccessPdf",
                headers=s2_headers,
                timeout=15
            )
            if s2_resp.status_code == 200:
                s2_data = s2_resp.json()
                oa_pdf = s2_data.get("openAccessPdf", {}).get("url") if s2_data.get("openAccessPdf") else None
                if oa_pdf:
                    print(f"[node_process_base] Trying S2 openAccessPdf: {oa_pdf}")
                    parsed_success = download_and_parse_pdf(oa_pdf, base_doi)

                if not parsed_success and s2_data.get("abstract"):
                    print(f"[node_process_base] Indexing abstract from Semantic Scholar as fallback.")
                    title = s2_data.get("title", "Unknown Title")
                    abstract = s2_data.get("abstract", "")
                    abstract_chunk = f"Title: {title}\nAbstract: {abstract}"
                    add_paper_to_db(base_doi, [abstract_chunk])
                    parsed_success = True
        except Exception as e:
            print(f"[node_process_base] Fallback retrieval error: {e}")

    # 2. Query VectorDB for relevant context
    retrieved_chunks = query_db(user_query, k=5, doi_filter=base_doi)
    if not retrieved_chunks:
        # If filter yielded nothing or parsing failed, query without filter
        retrieved_chunks = query_db(user_query, k=3)

    if retrieved_chunks:
        context_snippets = "\n\n".join([f"- {c['text']}" for c in retrieved_chunks])
        context_update = (
            f"### [DOI: {base_doi}] Base Paper Findings\n"
            f"{context_snippets}\n"
        )
    else:
        context_update = (
            f"### [DOI: {base_doi}] Base Paper Findings\n"
            f"Note: Full text or abstract could not be indexed for this DOI. "
            f"Agent will seek references for further evidence.\n"
        )

    current_dois.append(base_doi)
    print(f"[node_process_base] Retrieved {len(retrieved_chunks)} initial chunks.")

    return {
        "current_dois": current_dois,
        "visited_dois": visited_dois,
        "accumulated_context": context_update,
        "status": "CONTINUE",
        "hop_count": 0,
    }


# ----------------------------------------------------------------------
# Node 2: node_evaluate
# ----------------------------------------------------------------------
def node_evaluate(state: AgentState) -> Dict[str, Any]:
    """Evaluates accumulated context against user query using Gemini 1.5 Flash.
    Decides whether information is sufficient or references must be explored.
    """
    user_query = state["user_query"]
    accumulated_context = state.get("accumulated_context", "")
    hop_count = state.get("hop_count", 0)

    print(f"\n[node_evaluate] === Evaluating Context (Hop {hop_count}/2) ===")

    # Hard stop at maximum hop limit
    if hop_count >= 2:
        print("[node_evaluate] Maximum hop count reached (2). Routing to synthesis.")
        return {
            "status": "FINISH",
            "eval_reasoning": "Maximum hop depth (2) reached.",
            "next_search_keywords": "",
        }

    llm = get_llm()

    prompt = f"""You are an autonomous academic research evaluator.
Analyze the following research question and the accumulated context retrieved so far.
Determine if the accumulated context is sufficient to answer the research question with high academic rigor.

Research Question:
{user_query}

Accumulated Context:
{accumulated_context}

Return a valid JSON object with EXACTLY the following structure (no other markdown or extra text):
{{
    "is_sufficient": <true or false>,
    "reasoning": "<concise explanation of whether the context answers the question or what information is missing>",
    "next_search_keywords": "<if is_sufficient is false, 2 to 5 specific academic search terms to locate relevant reference papers; empty if true>"
}}
"""
    try:
        response = llm.invoke(prompt)
        content = extract_text(response.content).strip()
        # Clean potential markdown wrapping
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        eval_data = json.loads(content)
        is_sufficient = bool(eval_data.get("is_sufficient", False))
        reasoning = eval_data.get("reasoning", "")
        next_keywords = eval_data.get("next_search_keywords", "")

        print(f"[node_evaluate] Is Sufficient: {is_sufficient}")
        print(f"[node_evaluate] Reasoning: {reasoning}")
        if not is_sufficient:
            print(f"[node_evaluate] Next Keywords: {next_keywords}")

        status = "FINISH" if is_sufficient else "EXPLORE"
        return {
            "status": status,
            "eval_reasoning": reasoning,
            "next_search_keywords": next_keywords,
        }

    except Exception as e:
        print(f"[node_evaluate] Error in evaluation LLM call: {e}. Defaulting to EXPLORE if hop < 2.")
        return {
            "status": "EXPLORE" if hop_count < 2 else "FINISH",
            "eval_reasoning": f"Evaluation parse error: {e}",
            "next_search_keywords": user_query,
        }


# ----------------------------------------------------------------------
# Node 3: node_explore_refs
# ----------------------------------------------------------------------
def node_explore_refs(state: AgentState) -> Dict[str, Any]:
    """Finds references for the latest DOI via Semantic Scholar, picks the most
    relevant paper via LLM, downloads its PDF into ChromaDB, and enriches context.
    """
    current_dois = list(state.get("current_dois") or [])
    visited_dois = list(state.get("visited_dois") or [])
    accumulated_context = state.get("accumulated_context", "")
    next_keywords = state.get("next_search_keywords", state["user_query"])
    user_query = state["user_query"]
    hop_count = state.get("hop_count", 0)

    target_doi = current_dois[-1] if current_dois else state["base_doi"]
    print(f"\n[node_explore_refs] === Exploring References for DOI: {target_doi} ===")

    # 1. Fetch references from Semantic Scholar
    refs = get_semantic_scholar_references(target_doi, max_refs=10)

    # Filter out already visited DOIs
    candidate_refs = [
        r for r in refs
        if r.get("doi") and clean_doi(r["doi"]) not in visited_dois
    ]

    if not candidate_refs:
        print(f"[node_explore_refs] No new unvisited reference papers found for {target_doi}.")
        context_update = (
            f"{accumulated_context}\n"
            f"[Note: No additional references found for {target_doi} to explore further.]\n"
        )
        return {
            "accumulated_context": context_update,
            "status": "FINISH",
            "hop_count": hop_count + 1,
        }

    print(f"[node_explore_refs] Found {len(candidate_refs)} candidate reference paper(s).")

    # 2. LLM selects the single most relevant reference
    llm = get_llm()
    refs_text = ""
    for idx, ref in enumerate(candidate_refs):
        refs_text += (
            f"Paper #{idx + 1}:\n"
            f"  DOI: {ref['doi']}\n"
            f"  Title: {ref['title']}\n"
            f"  Abstract: {ref['abstract'][:400]}...\n\n"
        )

    selection_prompt = f"""You are an expert academic research selector.
Given the research question and required target keywords, select the ONE single most promising reference paper from the list below.

Research Question:
{user_query}

Target Keywords / Needed Information:
{next_keywords}

Candidate Reference Papers:
{refs_text}

Return a valid JSON object with EXACTLY this structure:
{{
    "selected_doi": "<the exact DOI of the chosen paper>",
    "reasoning": "<why this reference is the most relevant>"
}}
"""
    selected_doi = None
    selection_reasoning = ""
    try:
        resp = llm.invoke(selection_prompt)
        raw_text = resp.content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        data = json.loads(raw_text.strip())
        selected_doi = clean_doi(data.get("selected_doi", ""))
        selection_reasoning = data.get("reasoning", "")
    except Exception as e:
        print(f"[node_explore_refs] Error in selection LLM call: {e}. Falling back to first candidate.")
        selected_doi = clean_doi(candidate_refs[0]["doi"])
        selection_reasoning = "Fallback to first candidate."

    # Validate selected DOI
    selected_ref = next((r for r in candidate_refs if clean_doi(r["doi"]) == selected_doi), candidate_refs[0])
    selected_doi = clean_doi(selected_ref["doi"])
    visited_dois.append(selected_doi)

    print(f"[node_explore_refs] Selected Reference: {selected_doi} ({selected_ref.get('title')})")
    print(f"[node_explore_refs] Reason: {selection_reasoning}")

    # 3. Download and parse selected reference
    pdf_url = get_unpaywall_pdf_link(selected_doi) or selected_ref.get("pdf_url")
    indexed = False
    if pdf_url:
        indexed = download_and_parse_pdf(pdf_url, selected_doi)

    # If PDF download fails, index abstract as fallback
    if not indexed and selected_ref.get("abstract"):
        print(f"[node_explore_refs] Indexing abstract for {selected_doi} into ChromaDB.")
        content_to_index = f"Title: {selected_ref['title']}\nAbstract: {selected_ref['abstract']}"
        add_paper_to_db(selected_doi, [content_to_index])
        indexed = True

    # 4. Query ChromaDB for information from this reference
    retrieved = query_db(f"{next_keywords} {user_query}", k=4, doi_filter=selected_doi)
    if retrieved:
        ref_snippets = "\n\n".join([f"- {r['text']}" for r in retrieved])
    elif selected_ref.get("abstract"):
        ref_snippets = f"- Abstract: {selected_ref['abstract']}"
    else:
        ref_snippets = "- Full text could not be retrieved; paper was identified as relevant."

    new_context_entry = (
        f"\n### [DOI: {selected_doi}] Reference: {selected_ref.get('title')}\n"
        f"**Selection Context**: {selection_reasoning}\n"
        f"**Evidence Snippets**:\n"
        f"{ref_snippets}\n"
    )

    current_dois.append(selected_doi)
    new_accumulated = accumulated_context + new_context_entry

    return {
        "current_dois": current_dois,
        "visited_dois": visited_dois,
        "accumulated_context": new_accumulated,
        "hop_count": hop_count + 1,
        "status": "CONTINUE",
    }


# ----------------------------------------------------------------------
# Node 4: node_synthesize
# ----------------------------------------------------------------------
def node_synthesize(state: AgentState) -> Dict[str, Any]:
    """Synthesizes the final research answer from all accumulated context.
    Ensures clear academic citations linking evidence back to specific DOIs.
    """
    user_query = state["user_query"]
    accumulated_context = state.get("accumulated_context", "")
    current_dois = state.get("current_dois", [])

    print("\n[node_synthesize] === Synthesizing Final Comprehensive Research Answer ===")
    llm = get_llm()

    synthesis_prompt = f"""You are a distinguished academic research synthesis agent.
Your task is to provide a rigorous, comprehensive, and well-structured response to the research question below.
Base your answer exclusively on the retrieved academic context, making explicit citations to the relevant DOIs.

Research Question:
{user_query}

Retrieved Literature & Context:
{accumulated_context}

Analyzed DOIs:
{', '.join(current_dois)}

Formatting Guidelines:
1. Format your response cleanly in GitHub-flavored Markdown.
2. Structure your answer with an Executive Summary, Key Findings / Methodological Details, Cross-Paper Comparison/Synthesis, and References / Examined DOIs.
3. Every major claim or data point must cite its source using `[DOI: <doi>]`.
4. If certain aspects of the user question remain unanswered or uncertain based on the available literature, explicitly note them in a 'Limitations' section.
"""

    response = llm.invoke(synthesis_prompt)
    final_answer = extract_text(response.content).strip()

    print("[node_synthesize] Synthesis complete.")
    return {
        "final_answer": final_answer,
        "status": "FINISH",
    }


# ----------------------------------------------------------------------
# v2 workflow: CiteNet -> SynthDesk -> IntroWri
# ----------------------------------------------------------------------
def _artifact_output_dir(state=None):
    from pathlib import Path
    output_dir = Path((state or {}).get("run_output_dir") or (Path(__file__).resolve().parent / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _parse_json_response(content: Any) -> Any:
    raw = extract_text(content).strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())


def node_citenet(state: AgentState) -> Dict[str, Any]:
    """Collect references and export the citation graph and BibTeX library."""
    base_doi = clean_doi(state["base_doi"])
    references = get_semantic_scholar_references(base_doi, max_refs=10)
    output_dir = _artifact_output_dir(state)
    generated_files = dict(state.get("generated_files") or {})
    generated_files["html"] = export_citation_network(base_doi, references, output_dir)
    generated_files["bib"] = export_bibtex(base_doi, references, output_dir)
    dois = [base_doi] + [clean_doi(ref["doi"]) for ref in references if ref.get("doi")]
    print(f"[node_citenet] Collected {len(references)} references.")
    return {
        "citation_network": references,
        "generated_files": generated_files,
        "current_dois": dois,
        "visited_dois": dois,
    }


def node_synthdesk(state: AgentState) -> Dict[str, Any]:
    """Index the base/reference evidence and export a structured evidence matrix."""
    base_doi = clean_doi(state["base_doi"])
    references = list(state.get("citation_network") or [])
    local_pdf = find_local_pdf(base_doi)
    indexed = parse_local_pdf(local_pdf, base_doi) if local_pdf else False
    if not indexed:
        pdf_url = get_unpaywall_pdf_link(base_doi)
        if pdf_url:
            indexed = download_and_parse_pdf(pdf_url, base_doi)

    for reference in references[:5]:
        reference_doi = clean_doi(reference.get("doi", ""))
        if not reference_doi:
            continue
        reference_pdf = find_local_pdf(reference_doi)
        indexed_reference = parse_local_pdf(reference_pdf, reference_doi) if reference_pdf else False
        if not indexed_reference and reference.get("pdf_url"):
            indexed_reference = download_and_parse_pdf(reference["pdf_url"], reference_doi)
        if not indexed_reference and reference.get("abstract"):
            add_paper_to_db(
                reference_doi,
                [f"Title: {reference.get('title', '')}\nAbstract: {reference['abstract']}"],
            )

    retrieved = query_db(state["user_query"], k=10, doi_filter=base_doi)
    evidence_text = "\n\n".join(item["text"] for item in retrieved)
    if not evidence_text:
        evidence_text = "No indexed evidence was retrieved."
    extraction_prompt = f"""Extract structured evidence from the academic context below.
Return ONLY a JSON array. Each item must contain exactly: doi, title, methodology, dataset, key_findings.
Do not invent details. Use empty strings when evidence is absent.

Base DOI: {base_doi}
Research question: {state['user_query']}
Context:
{evidence_text}
"""
    evidence_pool = []
    try:
        evidence_pool = _parse_json_response(get_llm().invoke(extraction_prompt).content)
        if not isinstance(evidence_pool, list):
            evidence_pool = []
    except Exception as error:
        print(f"[node_synthdesk] Evidence extraction fallback: {error}")
    if not evidence_pool:
        evidence_pool = [{
            "doi": base_doi,
            "title": "Base paper",
            "methodology": "",
            "dataset": "",
            "key_findings": evidence_text[:4000],
        }]

    generated_files = dict(state.get("generated_files") or {})
    generated_files["csv"] = export_evidence_matrix(base_doi, evidence_pool, _artifact_output_dir(state))
    return {"evidence_pool": evidence_pool, "generated_files": generated_files}


def node_introwri(state: AgentState) -> Dict[str, Any]:
    """Write the final academic draft with author-year citations."""
    evidence_json = json.dumps(state.get("evidence_pool") or [], ensure_ascii=False, indent=2)
    prompt = f"""Write a rigorous academic research draft answering this question:
{state['user_query']}

Use only the evidence pool below. Every substantive claim must end with an author-year citation
in the form [Author, Year]. If author or year is unavailable, use [DOI: <doi>] instead.
Clearly separate evidence from proposed research directions and state limitations.
Use Markdown with: Executive Summary, Evidence Synthesis, Research Gap, Proposed Evaluation, and References.

Evidence pool:
{evidence_json}
"""
    draft = extract_text(get_llm().invoke(prompt).content).strip()
    return {"final_draft": draft, "final_answer": draft, "status": "FINISH"}


# ----------------------------------------------------------------------
# Routing Logic
# ----------------------------------------------------------------------
def route_after_evaluate(state: AgentState) -> str:
    """Routes to node_explore_refs if status is EXPLORE, else to node_synthesize."""
    status = state.get("status", "FINISH")
    if status == "EXPLORE":
        return "node_explore_refs"
    return "node_synthesize"


# ----------------------------------------------------------------------
# Graph Construction
# ----------------------------------------------------------------------
def build_research_graph():
    """Builds the static v2 CiteNet -> SynthDesk -> IntroWri workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("node_citenet", node_citenet)
    workflow.add_node("node_synthdesk", node_synthdesk)
    workflow.add_node("node_introwri", node_introwri)
    workflow.add_edge(START, "node_citenet")
    workflow.add_edge("node_citenet", "node_synthdesk")
    workflow.add_edge("node_synthdesk", "node_introwri")
    workflow.add_edge("node_introwri", END)

    return workflow.compile()
