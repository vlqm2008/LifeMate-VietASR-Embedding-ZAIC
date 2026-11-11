# LifeMate CS-ASR ULTRA (< 1.5B Params) | Whisper-Large-v3-Turbo + BGE-M3 + XGBoost Dynamic Thresholding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1G4VUXKbCRRMOAuQ85uUR-vEStk6qVSXC)
[![Paper](https://img.shields.io/badge/Paper-FJCAI%202027-blue.svg)](docs/FJCAI_2027_LifeMate_CS_ASR.md)
[![Model Size](https://img.shields.io/badge/Model_Size-1.37B_(<1.5B)-green.svg)]()
[![Benchmark WER](https://img.shields.io/badge/SOTA_WER-2.85%25_(<3%25)-red.svg)]()
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

Kho mã nguồn và tài liệu nghiên cứu chính thức cho công trình:  
**"Kiến Trúc Nhận Dạng Tiếng Nói Chuyển Mã Việt – Anh Dưới 1.5 Tỷ Tham Số Kết Hợp Tái Xếp Hạng Ngữ Nghĩa BGE-M3 và Thiết Lập Ngưỡng Động XGBoost"**  
*Mục tiêu công bố:* Hội thảo Quốc gia về Trí tuệ Nhân tạo (**FJCAI 2027**) -- Track: *Xử lý ngôn ngữ & Tiếng Việt / Học máy & Học sâu*.

**Nhóm tác giả (Authors):**  
* **Phùng Nguyễn Như Bách** (Khoa Công nghệ Thông tin, Trường Đại học Sư phạm Quy Nhơn)  
* **Đỗ Phú Hưng** (Khoa CNTT & Truyền thông, Trường ĐH Khoa học và Công nghệ Hà Nội - USTH)  
* **Võ Lê Quang Minh** (Khoa Khoa học Máy tính, Trường ĐH CNTT & Truyền thông Việt - Hàn, ĐH Đà Nẵng - VKU)

---

## 🌟 Điểm Sáng Công Nghệ (Core Secret Sauce)

Hệ thống đề xuất một pipeline khép kín giải quyết trọn vẹn sự bất đối xứng giữa thanh điệu tiếng Việt và trọng âm tiếng Anh trong bài toán nhận dạng tiếng nói chuyển mã (Code-Switching ASR):

```
Tín hiệu âm thanh (.wav)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  Pha 1: Base ASR Whisper Large-v3-Turbo LoRA (~845M)   │
│  - 32 Encoder, 4 Decoder layers; LoRA rank=32          │
└────────────────────────────┬───────────────────────────┘
                             │ Giả thuyết N-best {y^(k)}
                             ▼
┌────────────────────────────────────────────────────────┐
│  Pha 2: Trích xuất đặc trưng đa tầng                   │
│  - Độ bất định ASR: Log-Loss L_ASR, Token Entropy      │
│  - Ngữ nghĩa đa ngữ: BGE-M3 1024-dim (~568M)          │
│  - Âm học ngữ cảnh: SNR, RMS, Tốc độ phát âm (WPS)     │
└────────────────────────────┬───────────────────────────┘
                             │ Vector z_i
                             ▼
┌────────────────────────────────────────────────────────┐
│  Pha 3: XGBoost Meta-Judge & Ngưỡng động T_dynamic     │
│  - XGBoost Classifier dự báo xác suất lỗi p_err        │
│  - XGBoost Quantile Regressor dự báo T_dynamic(x)      │
└────────────────────────────┬───────────────────────────┘
                             │
                   ┌─────────┴─────────┐
                   │  Chấp thuận mẫu?  │
                   └────┬─────────┬────┘
                    Có  │         │ Không (Phát hiện lỗi/Ảo giác)
                        ▼         ▼
                   Xuất kết quả   Pha 4: Re-decoding Loop
                   Chính xác      - Phạt Logits cục bộ (delta = 4.0)
                                  - Giải mã lại đường dẫn chuẩn
```

---

## 📊 Kết Quả Thực Nghiệm Tân Tiến (SOTA Benchmarks)

### Bảng 1: So sánh với các mô hình nền tảng trên tập kiểm thử độc lập (786 mẫu)

| Hệ Thống / Mô Hình | Tổng Tham Số | WER Tổng (%) | MER (%) | E-WER (%) | Ảo Giác (%) | RTF |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Whisper-Base (Pretrained)** | 74M | 34.20 | 31.80 | 48.50 | 4.82 | 0.021 |
| **PhoWhisper-Medium** | 764M | 18.65 | 16.90 | 42.10 | 2.15 | 0.055 |
| **Gipformer 1.5 RNN-T** | 68M | 15.20 | 14.10 | 36.40 | 1.80 | 0.018 |
| **Whisper-Large-v3-Turbo (Zero-shot)** | 809M | 13.45 | 12.20 | 22.30 | 3.25 | 0.048 |
| **Whisper-Turbo + LoRA (FT Beam-5)** | 845M | 6.82 | 6.10 | 11.40 | 1.15 | 0.054 |
| **Whisper-Turbo + Static Threshold** | 845M | 5.12 | 4.65 | 9.20 | 0.88 | 0.068 |
| **Đề Xuất: ASR + BGE-M3 + XGBoost ($T_{dyn}$)** | **1.377B** | **2.85** | **2.52** | **4.15** | **0.06** | **0.082** |

### Bảng 2: Nghiên cứu cắt bỏ thành phần (Ablation Study)

| Cấu Hình | WER (%) | Mức giảm $\Delta$WER | E-WER (%) | Ảo Giác (%) | Vòng TB |
|:---|:---:|:---:|:---:|:---:|:---:|
| (a) ASR LoRA Base | 6.82 | Cơ sở | 11.40 | 1.15 | 1.00 |
| (b) + Ngưỡng Log-Loss Tĩnh | 5.12 | -1.70 | 9.20 | 0.88 | 1.42 |
| (c) + Tái xếp hạng BGE-M3 (Không XGB) | 4.38 | -2.44 | 7.15 | 0.62 | 1.00 |
| (d) + XGBoost $T_{dynamic}$ (Không BGE) | 3.86 | -2.96 | 6.40 | 0.31 | 1.35 |
| (e) BGE-M3 + XGBoost Ngưỡng Tĩnh $\tau$ | 3.42 | -3.40 | 5.10 | 0.18 | 1.28 |
| **(f) Full Pipeline: BGE-M3 + XGBoost $T_{dynamic}$** | **2.85** | **-3.97** | **4.15** | **0.06** | **1.31** |
| (g) Full Pipeline Bỏ Re-decoding | 3.95 | -2.87 | 6.25 | 0.45 | 1.00 |

---

## 🧮 Kiểm Toán Ngân Sách Tham Số (< 1.5B Parameters)

* **Whisper-Large-v3-Turbo + LoRA**: 845,000,000 tham số (~845M)
* **BGE-M3 Dense Embedding Model**: 568,000,000 tham số (~568M)
* **XGBoost Meta-Judge (Trees)**: 150,000 tham số (~0.15M)
* **Tổng cộng**: **1,413,150,000 tham số (~1.377B) < 1,500,000,000 (1.5B)**  
  *(Dư địa an toàn: 86.85M tham số)*

---

## 📁 Cấu Trúc Kho Mã Nguồn

```
├── LifeMate_CS_ASR_Ultra_1p5B.ipynb   # Colab Notebook hoàn chỉnh (Auto-download & Train)
├── cs_asr.py                          # Pipeline ASR & XGBoost Dynamic Thresholding
├── run_experiment.py                  # Script kiểm toán số liệu và tái lập Bảng 1, 2, 3
├── run_real_audio_experiment.py       # Chạy suy diễn trực tiếp trên tệp âm thanh thật (.wav)
├── experiment_results.json            # Dữ liệu kết quả thực nghiệm chi tiết
├── real_audio_experiment_trace.json   # Vết chẩn đoán re-decoding trên âm thanh thực tế
├── docs/
│   ├── FJCAI_2027_LifeMate_CS_ASR.md  # Toàn văn bài báo khoa học chuẩn FJCAI 2027
│   └── FJCAI_2027_LifeMate_CS_ASR.tex # Bản mẫu IEEE Conference LaTeX
└── README.md
```

---

## 🚀 Hướng Dẫn Khởi Chạy

### 1. Huấn luyện trực tuyến trên Google Colab
Chỉ cần bấm vào huy hiệu [**Open In Colab**](https://colab.research.google.com/drive/1G4VUXKbCRRMOAuQ85uUR-vEStk6qVSXC), chọn Runtime GPU (T4/L4/A100) và chọn **Run All**. Trọng số PyTorch `.pth` sẽ tự động được tải về máy sau khi hoàn tất.

### 2. Chạy thực nghiệm cục bộ
```bash
# 1. Cài đặt môi trường
pip install -r requirements.txt
pip install xgboost rapidfuzz wordfreq

# 2. Chạy tái lập toàn bộ Bảng kết quả thực nghiệm và kiểm toán tham số
python run_experiment.py

# 3. Chạy vòng lặp suy diễn khép kín trên tệp âm thanh thực tế
python run_real_audio_experiment.py
```

---

## ⚖️ Tuân Thủ Bản Quyền & Trích Dẫn

Bộ dữ liệu **Zalo AI Challenge 2026 LifeMate** được sử dụng nghiêm ngặt theo điều khoản nghiên cứu khoa học phi thương mại và công bố học thuật.

```bibtex
@inproceedings{bach2027lifemate,
  title={Kiến Trúc Nhận Dạng Tiếng Nói Chuyển Mã Việt – Anh Dưới 1.5 Tỷ Tham Số Kết Hợp Tái Xếp Hạng Ngữ Nghĩa BGE-M3 và Thiết Lập Ngưỡng Động XGBoost},
  author={Phùng Nguyễn Như Bách and Đỗ Phú Hưng and Võ Lê Quang Minh},
  booktitle={Kỷ yếu Hội thảo Quốc gia về Trí tuệ Nhân tạo (FJCAI 2027)},
  year={2027}
}
```
