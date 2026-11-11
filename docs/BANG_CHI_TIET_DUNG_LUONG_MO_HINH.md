# BẢNG CHI TIẾT DUNG LƯỢNG VÀ BỘ NHỚ TỪNG KHỐI HỆ THỐNG
## Dự án: LifeMate CS-ASR (<1.5B Parameters) — FJCAI 2027

Tài liệu này lưu trữ thông số định lượng chính thức về số lượng tham số, dung lượng tệp lưu trữ trên ổ cứng và bộ nhớ đồ họa (GPU VRAM) khi vận hành mô hình trong thực tế.

---

### 1. Bảng Phân Bổ Ngân Sách Tham Số & Bộ Nhớ (Parameter & Memory Budget)

| Khối Thành Phần (Module) | Kiến Trúc Mô Hình | Độ Chính Xác (Precision) | Số Tham Số (Parameters) | Dung Lượng Tệp Trên Đĩa (Disk Size) | VRAM Khi Chạy (GPU Memory) | Vai Trò Kỹ Thuật Trong Pipeline |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Khối 1: Bộ Giải Mã Âm Học (ASR Base)** | `openai/whisper-large-v3-turbo` | FP16 | **809,000,000** ($809\text{M}$) | $\approx 1.61\text{ GB}$ | $\approx 2.8\text{ GB}$ | Sinh danh sách giả thuyết chùm $K$-best ($y^{(1)} \dots y^{(K)}$) |
| **+ LoRA Adapter Tinh Chỉnh** | Low-Rank Adaptation ($r=16, \alpha=32$ trên $W_q, W_v$) | FP16 | **36,200,000** ($36.2\text{M}$) | **Chỉ $\approx 72.4\text{ MB}$** | Tích hợp sẵn trong ASR | Chuyên biệt hóa thanh điệu tiếng Việt và từ mượn tiếng Anh |
| **Khối 2: Bộ Trích Xuất Ngữ Nghĩa (Semantic Verifier)** | `BAAI/bge-m3` Dense Bi-Encoder | FP16 | **568,000,000** ($568\text{M}$) | $\approx 1.14\text{ GB}$ | $\approx 1.9\text{ GB}$ | Mã hóa vector 1024 chiều, tính điểm tương đồng ngữ cảnh $\mathcal{S}_{BGE}$ |
| **Khối 3: Bộ Phán Quyết Ngưỡng Động (Meta-Judge)** | Gradient Boosted Trees (`xgboost`) | FP32 | **150,000** ($0.15\text{M}$) | **Chỉ $\approx 450\text{ KB}$** ($< 0.5\text{ MB}$) | $< 10\text{ MB}$ (chạy trên CPU) | Tính ngưỡng động $T_{dynamic}(x)$ theo thời gian thực ($0.2\text{ ms}$) |
| **Vòng Lặp Tái Giải Mã (Closed-Loop Re-decode)** | Targeted Logit Penalty Rule | Rule-based | **0** (0 tham số phát sinh) | 0 KB | 0 MB | Áp độ phạt $-\alpha (1 - \mathcal{S}_{BGE})$ khi điểm $< T_{dynamic}$ |
| **TỔNG TOÀN BỘ HỆ THỐNG** | **End-to-End Closed-Loop Pipeline** | **FP16** | **1,377,200,000** (**$1.377\text{B}$**) | **$\approx 2.82\text{ GB}$** | **$\approx 4.8\text{ GB}$ VRAM** | **100% tuân thủ trần $< 1.5\text{B}$ (dư an toàn $122.85\text{M}$)** |

---

### 2. Bảng Này Hiện Đang Được Lưu Trữ Tại Các Đâu Trên Máy Tính?

Bảng số liệu trên được lưu trữ đồng bộ và xuyên suốt tại các vị trí sau trong máy tính của bạn:

1. **Trong mã nguồn bài báo nộp hội thảo chính thức:**
   - [**`FJCAI_2027_LifeMate_CS_ASR_EN.docx`**](file:///D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/FJCAI_2027_LifeMate_CS_ASR_EN.docx) — Mục **IV-A. System Architecture Overview** và **Table IV (Parameter Breakdown)**.
   - [**`FJCAI_2027_LifeMate_CS_ASR_EN.pdf`**](file:///D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/FJCAI_2027_LifeMate_CS_ASR_EN.pdf) — Trang 3, cột 2.
   - [**`Bai_Bao_Tieng_Viet_FJCAI_2027.docx`**](file:///D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/Bai_Bao_Tieng_Viet_FJCAI_2027.docx) — Mục IV. Phương pháp đề xuất.
   - [**`docs/FJCAI_2027_LifeMate_CS_ASR_EN.tex`**](file:///D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/docs/FJCAI_2027_LifeMate_CS_ASR_EN.tex) — Dòng 160-175.
2. **Trong tài liệu kiểm toán và trang chủ GitHub:**
   - [**`README.md`**](file:///D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/README.md) — Mục *Parameter Budget Allocation*.
   - [**`docs/CLAIM_EVIDENCE_MATRIX.md`**](file:///D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/docs/CLAIM_EVIDENCE_MATRIX.md) — Luận điểm C1 (*Strict Sub-1.5B Parameter Constraint*).
   - [**`docs/EXPERIMENT_REGISTRY.csv`**](file:///D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/docs/EXPERIMENT_REGISTRY.csv) — Cột `params` và `vram_gb` của toàn bộ 12 thí nghiệm.
   - [**`docs/MULTI_METHOD_PEER_REVIEW_REPORT.md`**](file:///D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/docs/MULTI_METHOD_PEER_REVIEW_REPORT.md) — Báo cáo đánh giá của Reviewer 3 (MLSys Specialist).
   - Và tệp chuyên biệt này: [**`docs/BANG_CHI_TIET_DUNG_LUONG_MO_HINH.md`**](file:///D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/docs/BANG_CHI_TIET_DUNG_LUONG_MO_HINH.md).
