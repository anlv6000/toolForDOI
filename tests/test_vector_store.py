"""Unit tests for ChromaDB storage and PDF processing.
"""

import os
import sys
import unittest
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent / "research_agent"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from vector_store import add_paper_to_db, query_db, reset_vector_store, DB_DIR


class TestVectorStore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reset_vector_store()

    @classmethod
    def tearDownClass(cls):
        reset_vector_store()
        # Clean up test DB directory if exists
        if DB_DIR.exists():
            try:
                shutil.rmtree(DB_DIR)
            except Exception:
                pass

    def test_add_and_query_paper(self):
        doi_1 = "10.1000/182"
        chunks_1 = [
            "Gravitational wave astronomy began with GW150914 observed by LIGO.",
            "The merger consisted of two stellar-mass black holes."
        ]

        doi_2 = "10.1000/183"
        chunks_2 = [
            "Quantum entanglement enables quantum key distribution protocols.",
            "BB84 protocol relies on single photon polarization states."
        ]

        self.assertTrue(add_paper_to_db(doi_1, chunks_1))
        self.assertTrue(add_paper_to_db(doi_2, chunks_2))

        # Query with DOI filter
        results_gw = query_db("LIGO black holes", k=2, doi_filter=doi_1)
        self.assertGreater(len(results_gw), 0)
        for item in results_gw:
            self.assertEqual(item["doi"], doi_1)

        # Query without DOI filter
        results_all = query_db("quantum photon", k=2)
        self.assertGreater(len(results_all), 0)


if __name__ == "__main__":
    unittest.main()
