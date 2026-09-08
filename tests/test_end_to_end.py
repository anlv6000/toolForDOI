"""End-to-end workflow test mocking external LLM and network calls.
Validates state progression across all LangGraph nodes.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_DIR = Path(__file__).resolve().parent.parent / "research_agent"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from graph import build_research_graph


class TestEndToEndWorkflow(unittest.TestCase):

    @patch("graph.get_llm")
    @patch("graph.download_and_parse_pdf")
    @patch("graph.get_unpaywall_pdf_link")
    @patch("graph.get_semantic_scholar_references")
    @patch("graph.query_db")
    def test_complete_research_flow(
        self,
        mock_query_db,
        mock_s2_refs,
        mock_unpaywall,
        mock_download,
        mock_get_llm,
    ):
        # 1. Mock external network/storage tools
        mock_unpaywall.return_value = "https://example.org/base.pdf"
        mock_download.return_value = True
        mock_query_db.return_value = [
            {"text": "Base paper findings on GW150914 reproducibility.", "doi": "10.1038/s41586-020-2649-2"}
        ]
        mock_s2_refs.return_value = [
            {
                "doi": "10.1109/MCSE.2021.3059232",
                "title": "Reproducing GW150914",
                "abstract": "First observation workflow replication study.",
                "pdf_url": "https://arxiv.org/pdf/2010.07244",
            }
        ]

        extraction_resp = MagicMock()
        extraction_resp.content = '[{"doi": "10.1038/s41586-020-2649-2", "title": "Base paper", "methodology": "Workflow study", "dataset": "GW150914", "key_findings": "Reproducible analysis."}]'

        intro_resp = MagicMock()
        intro_resp.content = "# Research Synthesis\n\nThe workflow is reproducible [Author, 2021]."

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [extraction_resp, intro_resp]
        mock_get_llm.return_value = mock_llm

        # Build and invoke graph
        graph = build_research_graph()
        initial_state = {
            "user_query": "How was GW150914 analyzed and reproduced?",
            "base_doi": "10.1038/s41586-020-2649-2",
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

        final_state = graph.invoke(initial_state)

        # Assertions
        self.assertEqual(final_state["status"], "FINISH")
        self.assertEqual(final_state["hop_count"], 0)
        self.assertIn("10.1038/s41586-020-2649-2", final_state["current_dois"])
        self.assertIn("10.1109/MCSE.2021.3059232", final_state["current_dois"])
        self.assertIn("# Research Synthesis", final_state["final_answer"])
        self.assertEqual(set(final_state["generated_files"]), {"html", "csv", "bib"})


if __name__ == "__main__":
    unittest.main()
