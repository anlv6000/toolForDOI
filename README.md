# Research Agent

Ứng dụng desktop hỗ trợ đọc bài báo khoa học, tìm research gap từ phần Related Work/citations và tạo bản tổng hợp có trích dẫn DOI bằng Gemini.

PDF local được ưu tiên xử lý trước nên hỗ trợ cả PDF tải qua paywall. Nếu PDF local không tồn tại hoặc parse thất bại, agent mới thử Unpaywall, Semantic Scholar và abstract fallback.

---

## Kiến trúc và workflow

Luồng chính hiện tại:

1. Đặt PDF gốc trong `research_agent/PDF/`.
2. Nếu thư mục chỉ có một PDF, app tự điền tên file làm DOI.
3. Parse PDF bằng PyMuPDF, chia chunk và index vào ChromaDB.
4. Nút `Generate gap question from citations` đọc Related Work, citations, limitations và future work đã index để tạo một câu hỏi gap.
5. Agent đánh giá context, có thể tìm thêm reference qua Semantic Scholar tối đa 2 hop.
6. Gemini tạo `Final Research Synthesis`, hiển thị trên app và lưu thành file UTF-8.

```
[Start]
   │
   ▼
[node_process_base] ──> Local PDF first; online fallback if needed ──> Parse ──> ChromaDB
   │
   ▼
[node_evaluate]     ──> Gemini evaluates context (maximum 2 hops)
   │
   ├── Status: "EXPLORE" (context chưa đủ và còn hop)
   │        │
   │        ▼
   │   [node_explore_refs] ──> Semantic Scholar lấy reference liên quan
   │        │                   Index PDF hoặc abstract của reference
   │        │                   Bổ sung context
   │        └─────────────────> Quay lại [node_evaluate]
   │
   └── Status: "FINISH" (đủ context hoặc đã đạt 2 hop)
            │
            ▼
      [node_synthesize]   ──> Gemini tạo báo cáo Markdown có citation DOI
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
GEMINI_EMBED_MODEL=models/gemini-embedding-001
```

> `GOOGLE_API_KEY` là bắt buộc. `UNPAYWALL_EMAIL` và `SEMANTIC_SCHOLAR_API_KEY` chỉ phục vụ fallback online.

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
```bash
python desktop_app.py
```

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

Mỗi lần chạy được lưu tại `research_agent/outputs/<doi>.txt`, ví dụ `research_agent/outputs/10.1109_JBHI.2021.3119519.txt`.

ChromaDB được tách collection theo embedding model để tránh lỗi dimension mismatch giữa collection cũ 384 chiều và Gemini embedding 3072 chiều.

---

## Kiểm thử

Chạy bộ test:
```bash
python -m unittest discover -s tests
```
