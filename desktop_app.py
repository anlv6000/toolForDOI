"""Tkinter desktop interface for the academic research agent."""

import os
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

PROJECT_DIR = Path(__file__).resolve().parent / "research_agent"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from graph import build_research_graph, clean_doi, generate_gap_question
from main import OUTPUT_DIR, create_run_output_dir, save_synthesis, validate_environment


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
        ttk.Label(form, text="Gemini model").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
        configured_model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.model_var = tk.StringVar(value=configured_model)
        self.model_combo = ttk.Combobox(
            form,
            textvariable=self.model_var,
            values=(
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-3-flash",
                "gemini-3.1-flash-lite",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-3.8-flash",
            ),
            state="normal",
        )
        self.model_combo.grid(row=2, column=1, sticky="ew", pady=5)
        self.run_button = ttk.Button(form, text="Run research", command=self._start_run)
        self.run_button.grid(row=3, column=1, sticky="e", pady=(8, 0))
        self.gap_button = ttk.Button(form, text="Generate gap question from citations", command=self._start_gap_question)
        self.gap_button.grid(row=3, column=0, sticky="w", pady=(8, 0))

        result_frame = ttk.LabelFrame(self, text="Final Research Synthesis", padding=10)
        result_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 12))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        self.result_text = ScrolledText(result_frame, wrap="word", font=("Consolas", 10))
        self.result_text.grid(row=0, column=0, sticky="nsew")

        artifact_frame = ttk.Frame(self, padding=(20, 0, 20, 8))
        artifact_frame.grid(row=3, column=0, sticky="ew")
        self.artifact_files = {}
        self.graph_button = ttk.Button(artifact_frame, text="Open Citation Graph", command=self._open_graph, state="disabled")
        self.graph_button.pack(side="left", padx=(0, 8))
        self.evidence_button = ttk.Button(artifact_frame, text="Open Evidence Matrix", command=self._open_evidence, state="disabled")
        self.evidence_button.pack(side="left", padx=(0, 8))
        self.bib_button = ttk.Button(artifact_frame, text="Export Zotero (.bib)", command=self._export_bib, state="disabled")
        self.bib_button.pack(side="left")

        footer = ttk.Frame(self, padding=(20, 0, 20, 16))
        footer.grid(row=4, column=0, sticky="ew")
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
        model = self.model_var.get().strip()
        if not doi or not query:
            messagebox.showwarning("Missing input", "Please provide both a DOI and a research question.")
            return
        if not model:
            messagebox.showwarning("Missing model", "Please select or enter a Gemini model.")
            return
        self.run_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self._set_artifact_buttons("disabled")
        self.result_text.delete("1.0", "end")
        self.status_var.set("Running research... Please wait.")
        threading.Thread(target=self._run_worker, args=(doi, query, model), daemon=True).start()

    def _start_gap_question(self):
        doi = clean_doi(self.doi_var.get())
        model = self.model_var.get().strip()
        if not doi:
            messagebox.showwarning("Missing DOI", "Please provide a base DOI first.")
            return
        if not model:
            messagebox.showwarning("Missing model", "Please select or enter a Gemini model.")
            return
        self.run_button.configure(state="disabled")
        self.gap_button.configure(state="disabled")
        self.status_var.set("Reading related-work citations and generating a gap question...")
        threading.Thread(target=self._gap_worker, args=(doi, model), daemon=True).start()

    def _gap_worker(self, doi, model):
        previous_model = os.environ.get("GEMINI_MODEL")
        try:
            os.environ["GEMINI_MODEL"] = model
            if not validate_environment():
                raise RuntimeError("Environment configuration is incomplete. Check research_agent/.env.")
            question = generate_gap_question(doi)
            self.after(0, self._gap_complete, question)
        except Exception as error:
            self.after(0, self._gap_failed, str(error))
        finally:
            self._restore_model(previous_model)

    def _gap_complete(self, question):
        self.query_text.delete("1.0", "end")
        self.query_text.insert("1.0", question)
        self.status_var.set("Gap question generated from citation evidence. Review it before running research.")
        self.run_button.configure(state="normal")
        self.gap_button.configure(state="normal")

    def _gap_failed(self, error):
        self.status_var.set("Gap-question generation failed")
        self.run_button.configure(state="normal")
        self.gap_button.configure(state="normal")
        messagebox.showerror("Gap question failed", error)

    def _run_worker(self, doi, query, model):
        previous_model = os.environ.get("GEMINI_MODEL")
        try:
            os.environ["GEMINI_MODEL"] = model
            if not validate_environment():
                raise RuntimeError("Environment configuration is incomplete. Check research_agent/.env.")
            state = {
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
            run_output_dir, report_number = create_run_output_dir(doi)
            state["run_output_dir"] = str(run_output_dir)
            state["report_number"] = report_number
            final_state = build_research_graph().invoke(state)
            answer = final_state.get("final_answer", "No synthesis generated.")
            output_path = save_synthesis(
                doi,
                query,
                answer,
                final_state.get("current_dois", []),
                final_state.get("hop_count", 0),
                output_dir=run_output_dir,
                report_number=report_number,
            )
            self.after(0, self._run_complete, answer, output_path, final_state.get("generated_files", {}))
        except Exception as error:
            self.after(0, self._run_failed, str(error))
        finally:
            self._restore_model(previous_model)

    @staticmethod
    def _restore_model(previous_model):
        if previous_model is None:
            os.environ.pop("GEMINI_MODEL", None)
        else:
            os.environ["GEMINI_MODEL"] = previous_model

    def _run_complete(self, answer, output_path, generated_files):
        self.result_text.insert("1.0", answer)
        self.artifact_files = generated_files
        self.status_var.set(f"Saved: {output_path}")
        self.run_button.configure(state="normal")
        self.open_folder_button.configure(state="normal")
        self._set_artifact_buttons("normal")

    def _run_failed(self, error):
        self.status_var.set("Run failed")
        self.run_button.configure(state="normal")
        self._set_artifact_buttons("disabled")
        messagebox.showerror("Execution failed", error)

    def _set_artifact_buttons(self, state):
        self.graph_button.configure(state=state if self.artifact_files else "disabled")
        self.evidence_button.configure(state=state if self.artifact_files else "disabled")
        self.bib_button.configure(state=state if self.artifact_files else "disabled")

    def _open_graph(self):
        path = self.artifact_files.get("html")
        if path:
            webbrowser.open(Path(path).resolve().as_uri())

    def _open_evidence(self):
        path = self.artifact_files.get("csv")
        if path:
            os.startfile(Path(path).resolve())

    def _export_bib(self):
        path = self.artifact_files.get("bib")
        if path:
            self.clipboard_clear()
            self.clipboard_append(Path(path).read_text(encoding="utf-8"))
            self.status_var.set("BibTeX copied to clipboard")

    def _open_outputs(self):
        import os
        os.startfile(OUTPUT_DIR)


if __name__ == "__main__":
    ResearchApp().mainloop()
