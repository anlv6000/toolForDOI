# YÊU CẦU NÂNG CẤP MÃ NGUỒN: toolForDOI v2.0 (Autonomous Research Workflow)

## 1. Tổng quan Kiến trúc Mới
Dự án cần được tái cấu trúc từ luồng LangGraph hiện tại (tìm PDF -> bóc tách -> tìm gap question) sang mô hình đường ống 3 trợ lý tuần tự: **CiteNet -> SynthDesk -> IntroWri**.
Mục tiêu là tạo ra các sản phẩm trung gian có thể kiểm chứng được (`.html`, `.csv`, `.bib`) thay vì chỉ in ra một text report duy nhất. Toàn bộ UI (Tkinter) và DB (ChromaDB) hiện tại giữ nguyên kiến trúc lõi nhưng mở rộng để hỗ trợ hiển thị/quản lý đa file output.

## 2. Nâng cấp Dependencies (Yêu cầu thêm vào `requirements.txt`)
Yêu cầu hệ thống cài đặt thêm các thư viện xử lý dữ liệu và trực quan hóa:
* `pyvis` (Dựng đồ thị HTML tương tác)
* `pandas` (Xuất ma trận CSV)
* `networkx` (Xử lý cấu trúc node/edge nếu cần)

## 3. Tái cấu trúc Module `tools.py` (Bổ sung Data Export)
Yêu cầu định nghĩa thêm 3 hàm tiện ích mới, chỉ viết logic xuất file, KHÔNG làm thay đổi các hàm xử lý Unpaywall/PyMuPDF cũ.

### 3.1. `export_citation_network(base_doi, references_list, output_dir)`
* **Input:** `base_doi` (str), `references_list` (list các dict chứa metadata từ Semantic Scholar: doi, title, authors, abstract).
* **Logic:** Sử dụng `pyvis.network.Network`. Tạo Node gốc màu cam. Duyệt qua `references_list` tạo các Node nhánh màu xanh. Thêm cạnh (Edge) nối từ gốc ra nhánh. Thêm Title và Abstract vào thuộc tính `title` (hover tooltip) của từng node.
* **Output:** Đường dẫn tới file `[base_doi]_network.html` lưu trong `research_agent/outputs/`.

### 3.2. `export_evidence_matrix(base_doi, extracted_claims, output_dir)`
* **Input:** `base_doi` (str), `extracted_claims` (list các dict với keys: `doi`, `title`, `methodology`, `dataset`, `key_findings`).
* **Logic:** Chuyển list thành `pandas.DataFrame` và gọi `.to_csv()`.
* **Output:** Đường dẫn tới file `[base_doi]_evidence.csv` chuẩn UTF-8.

### 3.3. `export_bibtex(base_doi, references_list, output_dir)`
* **Input:** Tương tự 3.1.
* **Logic:** Chuyển đổi siêu dữ liệu thành format chuỗi BibTeX chuẩn.
* **Output:** File `[base_doi]_refs.bib`.

## 4. Tái cấu trúc Module `graph.py` (Core Workflow)

### 4.1. Cập nhật `AgentState`
Thay thế State cũ bằng cấu trúc TypedDict mới:
* `base_doi` (str): DOI gốc do user nhập.
* `user_query` (str): Câu hỏi hoặc định hướng nghiên cứu.
* `citation_network` (list): Dữ liệu thô kéo từ Semantic Scholar.
* `evidence_pool` (list): Dữ liệu cấu trúc RAG bóc tách từ ChromaDB (Methodology, Findings).
* `final_draft` (str): Output cuối cùng của LLM.
* `generated_files` (dict): Lưu đường dẫn các file đã tạo `{"html": str, "csv": str, "bib": str}`.

### 4.2. Xóa các Node cũ và khởi tạo 3 Node mới
* **`node_citenet` (Khám phá):**
  * Hành động: Gọi API Semantic Scholar. Dùng data trả về gọi 2 hàm `export_citation_network` và `export_bibtex`.
  * Cập nhật State: Trả về `citation_network` và update `generated_files`.
* **`node_synthdesk` (Tổng hợp):**
  * Hành động: Nhận `citation_network`, lọc ra top DOI. Chạy logic xử lý PDF cũ (PyMuPDF) -> Lưu vào ChromaDB.
  * Truy vấn ChromaDB bằng prompt cấu trúc để ép Gemini nhả ra JSON bóc tách (Dataset, Method...). Đưa JSON vào `export_evidence_matrix`.
  * Cập nhật State: Trả về `evidence_pool` và update `generated_files`.
* **`node_introwri` (Viết nháp):**
  * Hành động: Cấp cho Gemini toàn bộ `evidence_pool`.
  * Ràng buộc Prompt (System Instruction): Phải dùng văn phong học thuật, bắt buộc trích dẫn chéo theo cú pháp `[Tác giả, Năm]` ở cuối mỗi luận điểm dựa trên dữ liệu từ `evidence_pool`.
  * Cập nhật State: Trả về `final_draft`.

### 4.3. Cập nhật Edges
* Workflow tĩnh, không cần conditional routing phức tạp: `START -> node_citenet -> node_synthdesk -> node_introwri -> END`.

## 5. Cập nhật Giao diện `desktop_app.py` (Tkinter)
Kiến trúc Tkinter hiện tại đã có, yêu cầu AI Coder bổ sung các widget sau trên giao diện chính:
* **Output Panel mới:** Ngay bên dưới khu vực hiển thị `Final Research Synthesis`, thêm một `Frame` chứa 3 nút bấm (Buttons).
* **Nút 1: "Open Citation Graph"**: Kích hoạt `webbrowser.open(path_to_html)`.
* **Nút 2: "Open Evidence Matrix"**: Dùng thư viện `os` để mở file CSV bằng trình đọc mặc định của hệ điều hành.
* **Nút 3: "Export Zotero (.bib)"**: Copy nội dung file bib vào clipboard hoặc mở thư mục chứa file.
* **Xử lý trạng thái:** Các nút này mặc định `state="disabled"`, chỉ được bật (`state="normal"`) khi luồng LangGraph chạy xong và trả về dict `generated_files`.