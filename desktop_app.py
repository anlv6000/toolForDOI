"""Tkinter desktop interface for the academic research agent."""

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

PROJECT_DIR = Path(__file__).resolve().parent / "research_agent"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from graph import build_research_graph, clean_doi
from main import OUTPUT_DIR, save_synthesis, validate_environment


class ResearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Academic Research Agent")
        self.geometry("1050x720")
        self.minsize(780, 560)
        self._build_ui()
        self._prefill_single_local_pdf()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=(20, 18, 20, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Academic Research Agent", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Run a literature synthesis and save it by DOI", foreground="#555555").grid(row=1, column=0, sticky="w", pady=(4, 0))

        form = ttk.LabelFrame(self, text="Research task", padding=12)
        form.grid(row=1, column=0, sticky="ew", padx=20, pady=8)
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Base DOI").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        self.doi_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.doi_var).grid(row=0, column=1, sticky="ew", pady=5)
        ttk.Label(form, text="Research question").grid(row=1, column=0, sticky="nw", padx=(0, 10), pady=5)
        self.query_text = tk.Text(form, height=3, wrap="word")
        self.query_text.grid(row=1, column=1, sticky="ew", pady=5)
        self.run_button = ttk.Button(form, text="Run research", command=self._start_run)
        self.run_button.grid(row=2, column=1, sticky="e", pady=(8, 0))

        result_frame = ttk.LabelFrame(self, text="Final Research Synthesis", padding=10)
        result_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 12))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        self.result_text = ScrolledText(result_frame, wrap="word", font=("Consolas", 10))
        self.result_text.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(self, padding=(20, 0, 20, 16))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.open_folder_button = ttk.Button(footer, text="Open outputs folder", command=self._open_outputs, state="disabled")
        self.open_folder_button.grid(row=0, column=1, sticky="e")

    def _prefill_single_local_pdf(self):
        pdf_files = list((PROJECT_DIR / "PDF").glob("*.pdf"))
        if len(pdf_files) == 1:
            self.doi_var.set(pdf_files[0].stem)

    def _start_run(self):
        doi = clean_doi(self.doi_var.get())
        query = self.query_text.get("1.0", "end").strip()
        if not doi or not query:
            messagebox.showwarning("Missing input", "Please provide both a DOI and a research question.")
            return
        self.run_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.result_text.delete("1.0", "end")
        self.status_var.set("Running research... Please wait.")
        threading.Thread(target=self._run_worker, args=(doi, query), daemon=True).start()

    def _run_worker(self, doi, query):
        try:
            if not validate_environment():
                raise RuntimeError("Environment configuration is incomplete. Check research_agent/.env.")
            state = {
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
            final_state = build_research_graph().invoke(state)
            answer = final_state.get("final_answer", "No synthesis generated.")
            output_path = save_synthesis(
                doi,
                query,
                answer,
                final_state.get("current_dois", []),
                final_state.get("hop_count", 0),
            )
            self.after(0, self._run_complete, answer, output_path)
        except Exception as error:
            self.after(0, self._run_failed, str(error))

    def _run_complete(self, answer, output_path):
        self.result_text.insert("1.0", answer)
        self.status_var.set(f"Saved: {output_path}")
        self.run_button.configure(state="normal")
        self.open_folder_button.configure(state="normal")

    def _run_failed(self, error):
        self.status_var.set("Run failed")
        self.run_button.configure(state="normal")
        messagebox.showerror("Execution failed", error)

    def _open_outputs(self):
        import os
        os.startfile(OUTPUT_DIR)


if __name__ == "__main__":
    ResearchApp().mainloop()
