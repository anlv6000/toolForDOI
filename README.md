# Autonomous Academic Research Agent

An autonomous literature research agent built with **LangGraph**, **ChromaDB**, **PyMuPDF**, and **Google Gemini 1.5 Flash**.

Given a research question and a base paper DOI, the agent retrieves the Open Access PDF via **Unpaywall**, extracts and indexes its text into ChromaDB, evaluates whether the evidence is sufficient, and recursively explores reference papers using the **Semantic Scholar Graph API** (up to 2 hops) before synthesizing a final, academically cited report.

---

## 🏛️ Architecture & Workflow

```
[Start]
   │
   ▼
[node_process_base] ──> Fetch PDF via Unpaywall (or fallback to S2) ──> Parse with PyMuPDF ──> Index ChromaDB
   │
   ▼
[node_evaluate]     ──> Gemini 1.5 Flash evaluates: Is accumulated context sufficient? (Hop limit: 2)
   │
   ├── Status: "EXPLORE" (Insufficient & Hops < 2)
   │        │
   │        ▼
   │   [node_explore_refs] ──> Semantic Scholar Graph API (fetch references)
   │        │                   Gemini selects most relevant reference
   │        │                   Download PDF / Abstract into ChromaDB
   │        │                   Enrich context & increment hop_count
   │        └─────────────────> Return to [node_evaluate]
   │
   └── Status: "FINISH" (Sufficient or Hop Count == 2)
            │
            ▼
       [node_synthesize]   ──> Gemini 1.5 Flash writes comprehensive Markdown report with [DOI: ...] citations
            │
            ▼
          [END]
```

---

## 📁 File Structure

```
/g/toolForDOI
├── main.py                     # Root CLI launcher
├── requirements.txt            # Project dependencies
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore definitions
├── tests/                      # Automated test suite
│   ├── test_components.py
│   └── test_vector_store.py
└── research_agent/
    ├── __init__.py
    ├── .env                    # Your local environment variables
    ├── .env.example
    ├── requirements.txt
    ├── main.py                 # Main application CLI
    ├── graph.py                # LangGraph State & node definitions
    ├── tools.py                # Unpaywall, Semantic Scholar & PyMuPDF tools
    ├── vector_store.py         # ChromaDB persistence & similarity search
    └── db/                     # ChromaDB local vector storage (auto-generated)
```

---

## 🚀 Quickstart

### 1. Installation

Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration (`.env`)

Copy `.env.example` to `.env` in `research_agent/` (or root):
```bash
cp research_agent/.env.example research_agent/.env
```

Edit `research_agent/.env`:
```env
GOOGLE_API_KEY=your_google_api_key_here
UNPAYWALL_EMAIL=your_real_email@domain.com
SEMANTIC_SCHOLAR_API_KEY=your_optional_s2_key_here
```

> **Important Notes on Keys:**
> - `GOOGLE_API_KEY`: Get a free API key at [Google AI Studio](https://aistudio.google.com/).
> - `UNPAYWALL_EMAIL`: Unpaywall requires a real-looking email (e.g. `yourname@domain.com`). Test domains like `example.com` or `test@...` are rejected with HTTP 422.
> - `SEMANTIC_SCHOLAR_API_KEY`: Optional; Semantic Scholar works without a key for standard queries.

---

## 💻 Usage

### Command-Line Arguments:
```bash
python main.py --doi 10.1038/s41586-020-2649-2 --query "How was GW150914 analyzed and reproduced?"
```

### Interactive Mode:
Simply run without arguments to enter DOI and question interactively:
```bash
python main.py
```

You will be prompted:
```text
Enter base paper DOI (e.g. 10.1038/s41586-020-2649-2): 10.1038/s41586-020-2649-2
Enter research question: What methods were used for data reproducibility?
```

### Desktop App:
```bash
python desktop_app.py
```

The desktop app runs the same research workflow, displays the final synthesis, and saves each report as a UTF-8 text file under `research_agent/outputs/`, using the DOI as the filename.

---

## 🧪 Running Automated Tests

Run the test suite with:
```bash
python -m unittest discover -s tests
```
