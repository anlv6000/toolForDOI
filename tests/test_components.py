"""Unit and integration tests for Autonomous Academic Research Agent components.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add research_agent to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent / "research_agent"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools import (
    clean_doi,
    export_bibtex,
    export_citation_network,
    export_evidence_matrix,
    find_local_pdf,
    get_unpaywall_pdf_link,
    get_semantic_scholar_references,
)
from graph import build_research_graph, route_after_evaluate
from main import create_run_output_dir, output_filename, save_synthesis


class TestTools(unittest.TestCase):

    def test_clean_doi(self):
        self.assertEqual(clean_doi("https://doi.org/10.1038/s41586-020-2649-2"), "10.1038/s41586-020-2649-2")
        self.assertEqual(clean_doi("http://doi.org/10.1038/s41586-020-2649-2"), "10.1038/s41586-020-2649-2")
        self.assertEqual(clean_doi("doi:10.1038/s41586-020-2649-2"), "10.1038/s41586-020-2649-2")
        self.assertEqual(clean_doi(" 10.1038/s41586-020-2649-2 \n"), "10.1038/s41586-020-2649-2")

    def test_find_local_pdf_from_doi_derived_filename(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            pdf_dir = Path(directory)
            expected = pdf_dir / "10.1109JBHI.2021.3119519.pdf"
            expected.write_bytes(b"%PDF-test")
            pdf_path = find_local_pdf("10.1109/JBHI.2021.3119519", pdf_dir)
            self.assertEqual(pdf_path, expected)

    def test_synthesis_output_is_named_by_doi(self):
        self.assertEqual(output_filename("10.1109/JBHI.2021.3119519", 2), "10.1109_JBHI.2021.3119519_2.txt")

    def test_create_run_output_dir_increments_existing_run_folders(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            doi = "10.1000/example"
            doi_dir = Path(directory) / "10.1000_example"
            (doi_dir / "10.1000_example_1").mkdir(parents=True)
            (doi_dir / "10.1000_example_2").mkdir()

            with patch("main.OUTPUT_DIR", Path(directory)):
                run_dir, report_number = create_run_output_dir(doi)

            self.assertEqual(report_number, 3)
            self.assertEqual(run_dir, doi_dir / "10.1000_example_3")
            self.assertTrue(run_dir.is_dir())

    def test_export_artifacts(self):
        from tempfile import TemporaryDirectory

        references = [{
            "doi": "10.1000/reference",
            "title": "Reference paper",
            "authors": [{"name": "A. Author"}],
            "abstract": "A useful abstract.",
        }]
        claims = [{
            "doi": "10.1000/reference",
            "title": "Reference paper",
            "methodology": "CNN",
            "dataset": "DDR",
            "key_findings": "Improved grading.",
        }]
        with TemporaryDirectory() as directory:
            html_path = export_citation_network("10.1000/base", references, Path(directory))
            csv_path = export_evidence_matrix("10.1000/base", claims, Path(directory))
            bib_path = export_bibtex("10.1000/base", references, Path(directory))
            self.assertTrue(Path(html_path).is_file())
            self.assertIn("Reference paper", Path(html_path).read_text(encoding="utf-8"))
            self.assertIn("methodology", Path(csv_path).read_text(encoding="utf-8-sig"))
            self.assertIn("@article", Path(bib_path).read_text(encoding="utf-8"))

    @patch("requests.get")
    def test_get_unpaywall_pdf_link_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "is_oa": True,
            "best_oa_location": {
                "url_for_pdf": "https://example.org/paper.pdf",
                "url": "https://example.org/paper"
            }
        }
        mock_get.return_value = mock_response

        pdf_url = get_unpaywall_pdf_link("10.1038/s41586-020-2649-2")
        self.assertEqual(pdf_url, "https://example.org/paper.pdf")

    @patch("requests.get")
    def test_get_unpaywall_pdf_link_not_oa(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "is_oa": False,
            "best_oa_location": None
        }
        mock_get.return_value = mock_response

        pdf_url = get_unpaywall_pdf_link("10.1000/closed-access")
        self.assertIsNone(pdf_url)

    @patch("requests.get")
    def test_get_unpaywall_pdf_link_invalid_json(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("empty response")
        mock_get.return_value = mock_response

        pdf_url = get_unpaywall_pdf_link("10.1000/invalid-json")
        self.assertIsNone(pdf_url)

    @patch("requests.get")
    def test_get_semantic_scholar_references(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "references": [
                {
                    "title": "Reference Paper 1",
                    "abstract": "Abstract of paper 1",
                    "externalIds": {"DOI": "10.1109/MCSE.2021.1"},
                    "openAccessPdf": {"url": "https://arxiv.org/pdf/2010.07244"}
                },
                {
                    "title": "Reference Paper Without DOI",
                    "abstract": "Abstract of paper 2",
                    "externalIds": {}
                }
            ]
        }
        mock_get.return_value = mock_response

        refs = get_semantic_scholar_references("10.1038/s41586-020-2649-2", max_refs=5)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["doi"], "10.1109/MCSE.2021.1")
        self.assertEqual(refs[0]["title"], "Reference Paper 1")
        self.assertEqual(refs[0]["pdf_url"], "https://arxiv.org/pdf/2010.07244")


class TestGraphRouting(unittest.TestCase):

    def test_route_after_evaluate(self):
        state_explore = {"status": "EXPLORE"}
        state_finish = {"status": "FINISH"}
        state_other = {"status": "CONTINUE"}

        self.assertEqual(route_after_evaluate(state_explore), "node_explore_refs")
        self.assertEqual(route_after_evaluate(state_finish), "node_synthesize")
        self.assertEqual(route_after_evaluate(state_other), "node_synthesize")

    def test_graph_compiles(self):
        graph = build_research_graph()
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
