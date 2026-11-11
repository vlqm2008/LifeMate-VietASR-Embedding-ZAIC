# LifeMate CS-ASR (<1.5B): Vietnamese-English Code-Switching ASR via BGE-M3 Semantic Reranking and XGBoost Dynamic Thresholding

[![Conference](https://img.shields.io/badge/Conference-FJCAI%202027-blue.svg)](https://fjcai.org.vn/)
[![Submission Status](https://img.shields.io/badge/Status-Camera%20Ready%20Draft-green.svg)](https://cmt3.research.microsoft.com/FJCAI2027)
[![WER](https://img.shields.io/badge/WER-2.85%25%20(%3C%203.0%25)-brightgreen.svg)]()
[![Model Budget](https://img.shields.io/badge/Parameters-1.377B%20(%3C%201.5B)-orange.svg)]()
[![RTF](https://img.shields.io/badge/RTF-0.0415-purple.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1G4VUXKbCRRMOAuQ85uUR-vEStk6qVSXC)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Full%20Package%20ZIP-blue.svg)](https://drive.google.com/file/d/1PabqJVL2t8IQfixyAejQKRzfM08vrTtu/view?usp=drivesdk)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Công trình nghiên cứu toàn văn (Full Research Paper) nộp Hội thảo Quốc gia về Trí tuệ Nhân tạo (FJCAI 2027)**  
> **Chuyên đề (Track):** Xử lý ngôn ngữ & Tiếng Việt / Học máy & Học sâu  
> **Tác giả:**  
> - **Phùng Nguyễn Như Bách** — Khoa CNTT, Trường Đại học Sư phạm Quy Nhơn (QNU)  
> - **Đỗ Phú Hưng** — Khoa CNTT & Truyền thông, Trường ĐH Khoa học và Công nghệ Hà Nội (USTH)  
> - **Võ Lê Quang Minh** — Khoa Khoa học Máy tính, Trường ĐH CNTT & TT Việt – Hàn (VKU), Đại học Đà Nẵng  

---

## 📌 Tổng Quan Dự Án (Executive Summary)

Dự án đề xuất một kiến trúc nhận dạng tiếng nói tự động chuyển mã ngôn ngữ Việt – Anh (**Vietnamese-English Code-Switching ASR**) đột phá với quy mô được tối ưu hóa nghiêm ngặt **dưới 1.5 tỷ tham số ($1.377\text{B} < 1.5\text{B}$)**, vận hành mượt mà theo thời gian thực trên 01 GPU 6GB VRAM thương mại.

Hệ thống kết hợp hài hòa giữa:
1. **ASR Base:** Whisper-Large-v3-Turbo LoRA ($845\text{M}$ tham số) giải mã âm học đa ngữ tốc độ cao.
2. **Semantic Reranker:** BGE-M3 ($568\text{M}$ tham số) trích xuất biểu diễn ngữ nghĩa đa ngôn ngữ $1024$-chiều, triệt tiêu sự nhập nhằng âm vị tại các ranh giới chuyển mã (*code-switching boundaries*).
3. **Meta-Judge Ngưỡng Động:** Cây quyết định Gradient Boosting (XGBoost, $0.15\text{M}$ tham số) tự động ước lượng ngưỡng động $T_{dynamic}(x)$ theo từng câu thoại.
4. **Vòng lặp Phản hồi đóng (Closed-Loop Re-decoding):** Tự động phát hiện ảo giác suy diễn (hallucination) và tái giải mã có áp đặt độ phạt logit.

🏆 **Thành tựu cốt lõi:**
* Đạt **Word Error Rate (WER) 2.85%** trên bộ dữ liệu kiểm thử thực tế LifeMate CS (vượt chuẩn mục tiêu $< 3.0\%$).
* Giảm hơn **78.6% tỷ lệ lỗi trên từ mượn tiếng Anh (E-WER giảm từ $19.40\%$ xuống $4.15\%$)**.
* Hệ số thời gian thực **$RTF = 0.0415$** (độ trễ chỉ $\approx 145\text{ ms}$ cho câu thoại 3.5 giây).

---

## 🏗️ Sơ Đồ Kiến Trúc Hệ Thống (Closed-Loop Architecture)

```
+----------------------------------------------------------------------------------+
|                            CLOSED-LOOP CS-ASR PIPELINE                           |
|                                                                                  |
|   [ Audio Signal x ]                                                             |
|           |                                                                      |
|           v                                                                      |
|   +-----------------------+     N-best Hypotheses     +----------------------+   |
|   |  Whisper-Large-Turbo  | ------------------------> | BGE-M3 Multilingual  |   |
|   |  (LoRA Adapted ASR)   |   H_K = {y^(1)...y^(K)}   |  Semantic Reranker   |   |
|   +-----------------------+                           +----------------------+   |
|           ^                                                      |               |
|           |                                                      v               |
|    Re-decode Loop                                     [ 32-dim Feature Vector ]  |
|    (Logit Penalty)                                    z = [phi_ASR, phi_SEM]     |
|           |                                                      |               |
|           +----------------- Reject (Score < T_dyn) <----+       v               |
|                                                          |  +----------------+   |
|                                                          |  | XGBoost Meta-  |   |
|                                                          |  |  Judge Model   |   |
|                                                          |  +----------------+   |
|                                                          |       |               |
|   [ Final Verified Transcript y* ] <--- Accept (Pass) ---+-------+               |
+----------------------------------------------------------------------------------+
```

---

## 📊 Bảng Điểm Chuẩn Thực Nghiệm (Empirical Benchmarks)

### Bảng 1: Kiểm toán Ngân sách Tham số (Parameter Audit)
| Thành phần | Kiến trúc cụ thể | Số tham số | Tỷ lệ ngân sách |
| :--- | :--- | :---: | :---: |
| **ASR Base** | Whisper-Large-v3-Turbo + LoRA ($r=16$) | $845.000.000$ | $56.33\%$ |
| **Semantic Embedding** | BGE-M3 (Dense Context Head) | $568.000.000$ | $37.87\%$ |
| **Dynamic Meta-Judge** | XGBoost Gradient Boosted Trees | $150.000$ | $0.01\%$ |
| **TỔNG HỆ THỐNG** | **Proposed Pipeline** | **1.377.150.000 (1.377B)** | **91.81% (< 1.5B)** |

### Bảng 2: So sánh Điểm chuẩn trên tập LifeMate CS Test Set
| Hệ thống | Cấu hình suy luận | WER (%) | MER (%) | E-WER (%) | Ảo giác (%) | RTF |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1** | Whisper-Base Zero-shot | $34.20\%$ | $41.50\%$ | $38.90\%$ | $1.85\%$ | **0.014** |
| **Baseline 2** | Whisper-Large-v3-Turbo Zero-shot | $13.45\%$ | $18.20\%$ | $19.40\%$ | $0.92\%$ | $0.038$ |
| **Baseline 3** | Whisper-Turbo + Static Threshold ($\tau=0.85$) | $6.80\%$ | $8.50\%$ | $9.80\%$ | $0.45\%$ | $0.042$ |
| **Proposed System** | **Whisper-Turbo + BGE-M3 + XGBoost $T_{dynamic}$** | **2.85%** | **3.40%** | **4.15%** | **0.06%** | **0.0415** |

### Bảng 3: So sánh 5 Cặp đôi Mô hình (< 1.5B)
| STT | Cặp đôi Mô hình | Tham số | WER (%) | E-WER (%) | RTF | VRAM (GB) | Đạt WER < 3% |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Whisper-Turbo + BGE-M3 (Proposed Champion)** | **1.377B** | **2.85%** | **4.15%** | **0.0415** | **4.8 GB** | **ĐẠT** |
| 2 | Whisper-Small + Multilingual-E5-Base (Siêu nhẹ) | $0.522\text{B}$ | $6.42\%$ | $9.15\%$ | $0.0185$ | $2.1\text{ GB}$ | Chưa đạt |
| 3 | PhoWhisper-Small + Vietnamese-Bi-Encoder | $0.354\text{B}$ | $8.75\%$ | $15.30\%$ | $0.0162$ | $1.6\text{ GB}$ | Chưa đạt |
| 4 | Whisper-Medium + Multilingual-E5-Large | $1.329\text{B}$ | $3.75\%$ | $5.20\%$ | $0.0520$ | $5.4\text{ GB}$ | Gần đạt |
| 5 | PhoWhisper-Medium + BGE-M3 (Hybrid) | $1.337\text{B}$ | $4.10\%$ | $7.45\%$ | $0.0510$ | $5.2\text{ GB}$ | Chưa đạt |

---

## 🚀 Hướng Dẫn Chạy Nhanh (Quickstart)

### 1. Chạy Trực Tiếp Trên Google Colab (1-Click)
Mở file notebook [`LifeMate_CS_ASR_Ultra_1p5B.ipynb`](https://colab.research.google.com/drive/1G4VUXKbCRRMOAuQ85uUR-vEStk6qVSXC) trên Google Colab với GPU T4/A100 miễn phí. Notebook tự động tải dữ liệu, fine-tune LoRA, huấn luyện BGE-M3 và xuất checkpoint `.pth`.

### 2. Cài Đặt Môi Trường Cục Bộ
```bash
git clone https://github.com/AIVIETNAM-AIO-MinhHocAyAi/LifeMate-VietASR-Embedding-ZAIC.git
cd LifeMate-VietASR-Embedding-ZAIC
pip install -r requirements.txt
```

### 3. Chạy Kiểm Thử Tự Động & Trình Diễn Tương Tác
```bash
# Chạy bộ unit test kiểm chứng độ tin cậy khoa học
python tests/test_pipeline.py

# Chạy công cụ CLI tương tác mô phỏng quy trình Closed-loop
python demo_cli.py --demo

# Hoặc chạy kiểm tra toàn bộ chỉ với 1 click (Windows)
quick_test.bat
```

### 4. Sử Dụng CLI Phiên Mã Tệp Âm Thanh
```bash
python cs_asr.py transcribe --audio path/to/audio.wav --output transcript.txt
```

---

## 📁 Cấu Trúc Kho Lưu Trữ (Repository Structure)

```
LifeMate-VietASR-Embedding-ZAIC/
├── docs/
│   ├── FJCAI_2027_LifeMate_CS_ASR.tex     # Bản thảo LaTeX chuẩn IEEE 8 trang
│   ├── FJCAI_2027_LifeMate_CS_ASR.md      # Bài báo khoa học toàn văn định dạng Markdown
│   ├── PEER_REVIEW_REBUTTAL_GUIDE.md      # Cẩm nang phản biện bảo vệ bài báo
│   ├── FJCAI_2027_PRESENTATION_SLIDES.md  # Slide thuyết trình 15 phút tại hội thảo
│   ├── DATASET_SUMMARY.md                 # Thống kê chi tiết các tập dữ liệu
│   └── ZAIC_LEGAL_COMPLIANCE.md           # Tuân thủ bản quyền nghiên cứu ZAIC 2026
├── src/
│   ├── cs_asr.py                          # Pipeline CLI mã nguồn gốc
│   ├── train_asr_whisper.py               # Module huấn luyện ASR LoRA
│   ├── finetune_bge_m3.py                 # Module huấn luyện BGE-M3
│   └── text_normalizer.py                 # Chuẩn hóa văn bản chuyển mã
├── tests/
│   └── test_pipeline.py                   # Bộ kiểm thử đơn vị tự động (100% Pass)
├── configs/
│   └── default_config.yaml                # Cấu hình siêu tham số tập trung
├── LifeMate_CS_ASR_Ultra_1p5B.ipynb       # Colab Notebook huấn luyện & xuất .pth
├── compare_model_pairs.py                 # Thực nghiệm đối sánh 5 cặp đôi mô hình
├── demo_cli.py                            # CLI demo tương tác minh họa re-decoding
├── run_real_audio_experiment.py           # Thực nghiệm kín trên âm thanh .wav thực tế
├── experiment_results.json                # Kết quả kiểm chuẩn chi tiết
├── real_audio_experiment_trace.json       # Log chẩn đoán từng mẫu âm thanh
├── quick_test.bat                         # Trình chạy kiểm tra 1-click Windows
└── README.md                              # Giới thiệu tổng quan dự án
```

---


---

## 📄 Bản Thảo Camera-Ready Hoàn Chỉnh (Exactly 6 Pages IEEE Double-Column)

Bản thảo bài báo chính thức nộp hội thảo **FJCAI 2027** đã được biên soạn và định dạng chuẩn xác theo khuôn mẫu **IEEE Double-Column (2 cột)** đạt chuẩn mật độ học thuật cao nhất của các bài báo accepted tại FJCAI:

| Định dạng tệp | Đường dẫn tệp tin | Kích thước | Trạng thái |
|---|---|---|---|
| **Microsoft Word (.doc)** | [`docs/FJCAI_2027_LifeMate_CS_ASR_EN.doc`](docs/FJCAI_2027_LifeMate_CS_ASR_EN.doc) | 528.0 KB | ✅ Hoàn thiện chuẩn nộp ban tổ chức |
| **Microsoft Word (.docx)** | [`docs/FJCAI_2027_LifeMate_CS_ASR_EN.docx`](docs/FJCAI_2027_LifeMate_CS_ASR_EN.docx) | 408.1 KB | ✅ Bản gốc OpenXML 6 trang |
| **Compiled PDF (.pdf)** | [`docs/FJCAI_2027_LifeMate_CS_ASR_EN.pdf`](docs/FJCAI_2027_LifeMate_CS_ASR_EN.pdf) | 413.4 KB | ✅ Xuất trực tiếp từ Word COM Engine |
| **LaTeX Source (.tex)** | [`docs/FJCAI_2027_LifeMate_CS_ASR_EN.tex`](docs/FJCAI_2027_LifeMate_CS_ASR_EN.tex) | 30.2 KB | ✅ IEEEtran template 6 trang |

### 📸 Ảnh Chụp Toàn Bộ 6 Trang Bài Báo (Visual Page Previews)

<p align="center">
  <img src="docs/page_1.png" width="31%" alt="Page 1: Title, Abstract, Intro"/>
  <img src="docs/page_2.png" width="31%" alt="Page 2: Related Work, Hypotheses, Methodology"/>
  <img src="docs/page_3.png" width="31%" alt="Page 3: Algorithm 1, 32-dim Features, Setup"/>
</p>
<p align="center">
  <img src="docs/page_4.png" width="31%" alt="Page 4: Benchmark Table I, Ablation Table II, Figures"/>
  <img src="docs/page_5.png" width="31%" alt="Page 5: Generalization Table III, Case Studies, Latency Table V"/>
  <img src="docs/page_6.png" width="31%" alt="Page 6: Conclusion, Ethical, Appendix A Table VI, References"/>
</p>

---

## 🚀 Trải Nghiệm Ứng Dụng Web Tương Tác (Gradio Web UI)

Dự án cung cấp giao diện Web UI tương tác bằng **Gradio** cho phép thử nghiệm trực quan:
- **Tab 1: Live Inference & Closed-Loop Re-decoding:** Tải lên file âm thanh `.wav` hoặc chọn các mẫu thoại thực tế (`samples/sample1_meeting.wav`, `sample2_deploy.wav`, v.v.), hệ thống sẽ hiển thị bản dịch Whisper gốc, phân tích điểm ngữ nghĩa BGE-M3, ngưỡng động XGBoost $T_{dynamic}$, và tự động kích hoạt Re-decoding nếu phát hiện nhập nhằng âm vị.
- **Tab 2: Multi-Model Benchmark Comparison:** Bảng đối sánh trực tiếp 7 kiến trúc mô hình (WER, SyER, E-WER, Hallucination, RTF, Tham số).
- **Tab 3: Architecture & Parameter Budget Inspector:** Bảng phân bổ chi tiết tham số và bộ nhớ VRAM cho từng module (Tổng cộng $1.377\text{B} < 1.5\text{B}$).
- **Tab 4: Paper Artifacts & Downloads:** Tải trực tiếp các tệp `.doc`, `.docx`, `.pdf`, `.tex`.

```bash
# Khởi chạy ứng dụng Web trên cổng localhost:7860
python app.py
```

## 📖 Trích Dẫn Bài Báo (Citation)

Nếu bạn sử dụng mã nguồn, dữ liệu hoặc kiến trúc này trong nghiên cứu, vui lòng trích dẫn theo định dạng IEEE:

```bibtex
@inproceedings{bach2027lifemate,
  author    = {Phung Nguyen Nhu Bach and Do Phu Hung and Vo Le Quang Minh},
  title     = {Vietnamese-English Code-Switching Speech Recognition Under 1.5B Parameters via BGE-M3 Semantic Reranking and XGBoost Dynamic Thresholding},
  booktitle = {Proceedings of the National Conference on Artificial Intelligence (FJCAI)},
  year      = {2027},
  pages     = {1--8},
  address   = {Vietnam}
}
```
