# Research Agent

Ứng dụng desktop hỗ trợ đọc bài báo khoa học, dựng mạng citation, trích xuất evidence có cấu trúc và tạo bản tổng hợp có trích dẫn bằng Gemini.

PDF local được ưu tiên xử lý trước nên hỗ trợ cả PDF tải qua paywall. Nếu PDF local không tồn tại hoặc parse thất bại, agent mới thử Unpaywall, Semantic Scholar và abstract fallback.

---

## Kiến trúc và workflow

Luồng chính hiện tại:

1. Đặt PDF gốc trong `research_agent/PDF/`.
2. **CiteNet** lấy references từ Semantic Scholar, tạo citation graph HTML và thư viện BibTeX.
3. **SynthDesk** parse PDF bằng PyMuPDF, index ChromaDB, trích xuất methodology/dataset/findings và tạo evidence matrix CSV.
4. **IntroWri** dùng evidence pool để viết draft học thuật với citation dạng `[Author, Year]` hoặc DOI fallback.
5. Desktop app hiển thị draft và bật nút mở HTML/CSV hoặc copy BibTeX sau khi pipeline hoàn tất.

```
[Start]
   │
   ▼
[node_citenet] ──> Semantic Scholar references ──> citation_network ──> HTML + BibTeX
   │
   ▼
[node_synthdesk] ──> PDF/ChromaDB evidence extraction ──> evidence_pool ──> CSV
   │
   ▼
[node_introwri] ──> Gemini viết final_draft với citation học thuật
            │
            ▼
          [END]
```

---

## Cấu trúc thư mục

```
/g/toolForDOI
├── desktop_app.py              # Giao diện Tkinter
├── main.py                     # Launcher CLI
├── requirements.txt            # Dependencies
├── .env.example                # Mẫu biến môi trường
├── .gitignore
├── tests/                      # Bộ kiểm thử
│   ├── test_components.py
│   └── test_vector_store.py
└── research_agent/
    ├── __init__.py
      ├── .env                    # Biến môi trường local
    ├── .env.example
      ├── requirements.txt
      ├── main.py                 # CLI và lưu synthesis
      ├── graph.py                # LangGraph và gap question
      ├── tools.py                # PDF, Unpaywall, Semantic Scholar
      ├── vector_store.py         # Embedding và ChromaDB
      ├── PDF/                    # PDF local của người dùng
      ├── outputs/                # Báo cáo .txt theo DOI
      └── db/                     # ChromaDB persistence
```

---

## Cài đặt nhanh

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình (`.env`)

Sao chép `.env.example` thành `.env` trong `research_agent/`:
```bash
cp research_agent/.env.example research_agent/.env
```

Điền các biến trong `research_agent/.env`:
```env
GOOGLE_API_KEY=your_google_api_key_here
UNPAYWALL_EMAIL=your_real_email@domain.com
SEMANTIC_SCHOLAR_API_KEY=your_optional_s2_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_DEVICE=cuda
```

`GOOGLE_API_KEY` vẫn cần cho Gemini sinh câu trả lời. Embedding chạy local nên không dùng quota embedding của Gemini. `UNPAYWALL_EMAIL` và `SEMANTIC_SCHOLAR_API_KEY` chỉ phục vụ fallback online.

### Chạy embedding local trên GPU WSL2

Trong WSL, dùng đúng virtual environment Linux và cài các gói embedding:

```bash
cd /mnt/g/toolForDOI
source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers langchain-huggingface
```

Kiểm tra PyTorch nhìn thấy GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`LOCAL_EMBEDDING_DEVICE=cuda` sẽ dùng GPU. Nếu WSL chưa có CUDA hoặc muốn chạy CPU, đổi thành `LOCAL_EMBEDDING_DEVICE=cpu`. Lần chạy đầu tiên sẽ tải model `all-MiniLM-L6-v2` về cache HuggingFace.

---

## Cách chạy

### Chạy CLI với tham số
```bash
python main.py --doi 10.1038/s41586-020-2649-2 --query "How was GW150914 analyzed and reproduced?"
```

### Chạy CLI tương tác
```bash
python main.py
```

Chương trình sẽ hỏi DOI và câu hỏi nghiên cứu:
```text
Enter base paper DOI (e.g. 10.1038/s41586-020-2649-2): 10.1038/s41586-020-2649-2
Enter research question: What methods were used for data reproducibility?
```

### Chạy desktop app

`main.py` là CLI tương tác. Giao diện Tkinter nằm trong `desktop_app.py`.

Trên WSL2 có WSLg, cài Tkinter đúng với phiên bản Python của virtualenv. Ví dụ
nếu `.venv` dùng Python 3.12:

```bash
sudo apt update
sudo apt install -y python3.12-tk
```

Kiểm tra trước khi mở giao diện:

```bash
python -c "import tkinter; print('tkinter ok')"
```

Sau đó chạy:

```bash
cd /mnt/g/toolForDOI
source .venv/bin/activate
python desktop_app.py
```

Nếu chỉ muốn chạy terminal, dùng `python main.py`.

Trong app, `Generate gap question from citations` tạo câu hỏi từ citations của PDF đã index và điền vào ô Research question để người dùng xem lại. Dropdown `Gemini model` cho phép đổi model riêng cho từng lần chạy.

Các model text trong dropdown:

```text
gemini-2.5-flash
gemini-2.5-flash-lite
gemini-3-flash
gemini-3.1-flash-lite
gemini-3.5-flash
gemini-3.5-flash-lite
gemini-3.6-flash
gemini-3.7-flash
gemini-3.8-flash
```

Khi model hết quota, chọn model còn RPM/RPD trong Google AI Studio. Có thể nhập model khác trực tiếp vào ô chọn nếu API hỗ trợ.

Mỗi DOI có thư mục riêng. Mỗi câu hỏi/lần chạy được đánh số và chứa toàn bộ report cùng artifact:

```text
research_agent/outputs/10.1109_JBHI.2021.3119519/
└── 10.1109_JBHI.2021.3119519_1/
   ├── 10.1109_JBHI.2021.3119519_1.txt
   ├── 10.1109_JBHI.2021.3119519_network.html
   ├── 10.1109_JBHI.2021.3119519_evidence.csv
   └── 10.1109_JBHI.2021.3119519_refs.bib
```

Câu hỏi tiếp theo tạo `_2`, `_3`, ... và không ghi đè kết quả cũ.

Khi publisher chặn request tải PDF từ URL Unpaywall, SynthDesk sẽ thử theo thứ tự: PDF local của reference, URL open-access, rồi abstract từ Semantic Scholar. Vì vậy một lỗi anti-bot không làm mất hoàn toàn evidence của reference.

ChromaDB được tách collection theo embedding model để tránh lỗi dimension mismatch giữa collection cũ 384 chiều và Gemini embedding 3072 chiều.

---

## Kiểm thử

Chạy bộ test:
```bash
python -m unittest discover -s tests
```
