# LifeMate ZAIC: Hệ Thống ASR Tiếng Việt Siêu Nhẹ (< 1.5B Params) & Embedding BGE-M3

Dự án nghiên cứu và phát triển giải pháp Nhận dạng giọng nói tiếng Việt & Code-Switching Việt - Anh (ASR) kết hợp Biểu diễn ngữ nghĩa (Embedding BGE-M3) cho bài toán Trợ lý ảo **LifeMate - Zalo AI Challenge (ZAIC)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](colab_train_pipeline.ipynb)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Model Size](https://img.shields.io/badge/Model_Size-809M_(<1.5B)-green.svg)]()
[![Target WER](https://img.shields.io/badge/Target_WER-<3%25-blue.svg)]()

---

## 1. Mục Tiêu & Kiến Trúc Dự Án

- **Mô hình ASR**: `openai/whisper-large-v3-turbo` (809M tham số < 1.5B).
  - Kiến trúc: 32 Encoder layers, 4 Decoder layers (rút gọn từ 32 của Large-v3, tăng tốc độ suy luận x8 lần).
  - Tối ưu hóa: Fine-tune với LoRA / QLoRA ($r=32, \alpha=64$), huấn luyện hỗn hợp FP16/BF16.
  - Mục tiêu: **WER < 3%** trên tiếng Việt đời thường và câu lệnh chêm tiếng Anh (Code-Switching EN-VI).
- **Mô hình Embedding**: `BAAI/bge-m3` (568M tham số).
  - Tối ưu hóa: Fine-tune với Multiple Negatives Ranking Loss (InfoNCE) trên tập đối sánh giả thuyết âm thanh để bắt chính xác Intent & Entity của LifeMate.
- **Môi trường huấn luyện**: Tự động hóa chạy liên tục **8 - 10 tiếng** trên Google Colab (T4 / A100 GPU) với cơ chế lưu checkpoint lên Google Drive.

---

## 2. Bảng Tổng Hợp Số Liệu Dataset Hợp Lệ

### 2.1. Dataset ASR (Nhận dạng tiếng Việt & Code-Switching)

| Dataset | Quy mô (Giờ / Mẫu) | Ngôn ngữ / Đặc điểm | Giấy phép (License) | Ứng dụng cho LifeMate |
| :--- | :--- | :--- | :--- | :--- |
| **GigaSpeech 2** | 6.039 giờ | Tiếng Việt đa chủ đề, lọc tự động từ giọng tự nhiên | Academic / Non-commercial | Mở rộng vốn từ vựng & ngữ cảnh đa dạng |
| **SEA Code-Switching** | 45.3 giờ (10.306 mẫu) | Song ngữ EN-VI ("schedule meeting", "turn on wifi") | Research Only | Cực sát câu lệnh trợ lý đời thường |
| **ViMedCSS** | 32.64 giờ (24.3h train) | Tiếng Việt chêm thuật ngữ tiếng Anh chuyên ngành | CC BY 4.0 | Nhận diện chuẩn xác từ mượn tiếng Anh |
| **VietSuperSpeech** | 267 giờ | Hội thoại tự nhiên, khẩu ngữ hàng ngày | Academic Only | Voice Assistant commands |
| **VietSpeech** | 1.100+ giờ (~1M mẫu) | Đầy đủ phương ngữ 3 miền Bắc - Trung - Nam | Research with approval | Tăng cường tính bền bỉ với accent vùng miền |
| **ViMD** | 102.56 giờ | 63 tỉnh thành Việt Nam | Research Only | Nhận diện giọng địa phương đặc thù |
| **Bud500** | ~500 giờ | Podcast, du lịch, đời sống ba miền | Research Only | Dữ liệu phát âm tự nhiên, độ nhiễu thực tế |
| **PhoAudiobook** | 1.494 giờ | Sách nói tiếng Việt chuẩn thanh điệu | Research Only | Giúp mô hình bắt chuẩn dấu câu, thanh điệu |
| **Google FLEURS** | ~12 giờ vi_vn + en_us | Đọc chuẩn mẫu quốc tế | CC BY 4.0 | Benchmark chuẩn hóa WER / CER |
| **Mozilla Common Voice** | 30+ giờ vi | Giọng cộng đồng đa dạng thiết bị | CC0 (Public Domain) | Thử nghiệm độ nhiễu micro |
| **ZAIC LifeMate Set** | Đang cập nhật theo mùa | Dữ liệu cuộc thi chính thức | ZAIC Competition Rules | Đánh giá trực tiếp trên bài thi LifeMate |

### 2.2. Dataset Cho BGE-M3 Embedding (Retrieval / Intent Rerank)

| Dataset | Số mẫu | Định dạng dữ liệu | Ứng dụng cụ thể |
| :--- | :--- | :--- | :--- |
| **mMARCO Vietnamese** | 39.78 triệu triplets | (Query, Positive, Negative) | Contrastive learning nhận diện câu lệnh gần nghĩa |
| **WikiMatrix EN-VI** | 308.470 cặp song ngữ | Sentence Pairs | Đồng bộ không gian embedding giữa tiếng Việt và tiếng Anh |
| **PhoMT** | ~3.02 triệu cặp | Bilingual Translation Pairs | Tối ưu hóa hiểu câu lệnh Code-Switching EN-VI |
| **UIT-ViQuAD 2.0** | 35.000+ câu hỏi | Context - Question - Answer | Huấn luyện Hard Negative Mining |
| **VietEmbed-RAG Science**| 68.567 triplets | Triplet with hard negatives | Trích xuất thông tin chính xác theo domain |
| **STSBenchmark-VN** | 5.749 mẫu | Semantic similarity score 1-5 | Đánh giá chỉ số Spearman Correlation của Embedding |

---

## 3. Tuân Thủ Pháp Lý & Điều Khoản Sử Dụng Dataset Zalo AI Challenge

Căn cứ vào **Zalo AI Challenge Regulations** và thông lệ các mùa giải:
1. **Mục đích học thuật & Báo cáo nghiên cứu (Research Paper)**:
   - Ban tổ chức (VNG / Zalo AI) cho phép thí sinh sử dụng bộ dữ liệu cuộc thi để tiến hành nghiên cứu khoa học, viết luận văn và công bố bài báo khoa học.
2. **Quy định trích dẫn bắt buộc (Citation)**:
   - Bài báo khoa học cần trích dẫn rõ ràng:
     ```bibtex
     @misc{zaic2026lifemate,
       title={LifeMate: Multimodal Voice Assistant Benchmark},
       author={Zalo AI Challenge Organizer},
       year={2026},
       howpublished={\url{https://challenge.zalo.ai}}
     }
     ```
3. **Quy định bảo mật & Không phát tán dữ liệu gốc (No Raw Data Redistribution)**:
   - Tuyệt đối **không upload trực tiếp các file audio gốc (.wav/.mp3) hoặc nhãn riêng tư** của Zalo lên GitHub công khai.
   - Chỉ lưu hành nội bộ trên Google Drive bảo mật của cá nhân và nộp mô hình qua mã nguồn đã huấn luyện (weights / inference code).
4. **Quyền riêng tư (Privacy)**: Mọi thông tin nhận dạng cá nhân (PII) được bảo vệ và ẩn danh hoàn toàn.

---

## 4. Hướng Dẫn Huấn Luyện 10 Tiếng Trên Google Colab

1. Mở file [colab_train_pipeline.ipynb](colab_train_pipeline.ipynb) trên Google Colab.
2. Chọn Runtime: **Change runtime type -> T4 GPU / A100 GPU**.
3. Kết nối Google Drive để checkpoint được tự động sao lưu mỗi 200 bước:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Bấm **Runtime -> Run all**. Notebook tích hợp cơ chế chống rớt mạng tự động và resume từ checkpoint gần nhất nếu bị ngắt quãng.

---

## 5. Cấu Trúc Thư Mục Dự Án

```
├── README.md                          # Tài liệu tổng quan dự án & pháp lý
├── colab_train_pipeline.ipynb          # Notebook Google Colab hoàn chỉnh
├── requirements.txt                   # Danh sách thư viện Python
├── push_to_github.py                  # Script đẩy mã nguồn lên GitHub của bạn
├── src/
│   ├── config.py                      # Thiết lập hyperparameters & dataset links
│   ├── text_normalizer.py             # Chuẩn hóa âm vị tiếng Việt & số từ
│   ├── dataset_loader.py              # Xử lý data Hugging Face & Local Zalo
│   ├── train_asr_whisper.py           # Huấn luyện Whisper-v3-Turbo (<1.5B) với LoRA
│   ├── eval_wer.py                    # Đánh giá WER/CER chuẩn JiWER
│   └── finetune_bge_m3.py             # Fine-tune BGE-M3 với Contrastive Loss
└── docs/
    ├── DATASET_SUMMARY.md             # Thống kê chi tiết & Link dataset
    └── ZAIC_LEGAL_COMPLIANCE.md       # Phân tích điều khoản pháp lý ZAIC
```
