# Báo Cáo Tổng Hợp Dữ Liệu: ASR Tiếng Việt & Embedding BGE-M3

Tài liệu này tổng hợp đầy đủ số liệu thống kê, đặc tính ngôn ngữ, giấy phép sử dụng và đường dẫn truy cập của các bộ dữ liệu huấn luyện cho bài toán **LifeMate - Zalo AI Challenge (ZAIC)**.

---

## 1. Dữ Liệu ASR (Nhận Dạng Tiếng Việt & Song Ngữ EN-VI)

### 1.1. Ba Nguồn Dữ Liệu Ưu Tiên Cao Nhất Cho LifeMate

#### 1. GigaSpeech 2 (Vietnamese Subset)
- **Quy mô**: **6.039 giờ** tiếng Việt.
- **Nguồn gốc & Đặc tính**:
  - Dữ liệu thu thập từ các nền tảng video/podcast thực tế khu vực Đông Nam Á (DataoceanAI).
  - Đã trải qua bộ lọc chất lượng tự động từ transcript.
  - Phù hợp nhất để mô hình học biểu diễn âm học phong phú (acoustic features) và mở rộng từ vựng tự nhiên.
- **Giấy phép**: Thỏa thuận nghiên cứu phi thương mại (Non-commercial research).
- **Lưu ý xử lý**: Cần lọc thêm các câu có thời lượng âm thanh < 1.5s hoặc > 30s để tránh OOM bộ nhớ VRAM khi train.

#### 2. SEA Code-Switching Speech
- **Quy mô**: **45.3 giờ** EN-VI (khoảng 10.306 đoạn âm thanh).
- **Nguồn gốc & Đặc tính**:
  - Dữ liệu chuyên biệt cho hiện tượng đảo mã (code-switching) giữa tiếng Anh và tiếng Việt trong giao tiếp hàng ngày của người Việt.
  - Ví dụ mẫu: *"Mai schedule meeting với team lúc 8 giờ"*, *"Check inbox giùm anh nha"*.
- **Giá trị cốt lõi**: Khắc phục nhược điểm lớn nhất của các mô hình Whisper gốc là hay bị hallucination hoặc dịch sang tiếng Anh khi gặp câu lệnh chêm tiếng Anh của trợ lý ảo LifeMate.

#### 3. ViMedCSS
- **Quy mô**: **32.64 giờ** tổng cộng (24.3 giờ training, validation, test và hard-split riêng biệt).
- **Nguồn gốc & Đặc tính**:
  - Dữ liệu tiếng Việt chêm thuật ngữ tiếng Anh chuyên môn y tế, sức khỏe và công nghệ.
- **Giấy phép**: CC BY 4.0 (Hoàn toàn mở cho học thuật và phát triển).
- **Giá trị cốt lõi**: Giúp ASR phân biệt chuẩn xác các danh từ riêng, thuật ngữ kỹ thuật khó mà người dùng hay phát âm theo ngữ điệu Việt.

---

### 1.2. Các Bộ Dữ Liệu ASR Bổ Sung Đã Xác Minh

| STT | Dataset | Quy mô | Giấy phép | Điểm mạnh đối với LifeMate |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **VietSuperSpeech** | 267 giờ | Academic | Dữ liệu hội thoại đời thường chất lượng cao, cấu trúc câu rất gần với voice assistant |
| 2 | **VietSpeech** | 1.100+ giờ (~1M mẫu) | Request Access | Giọng nói tự nhiên trải dài 3 miền Bắc - Trung - Nam, giúp mô hình thích nghi đa dạng người dùng |
| 3 | **ViMD** | 102.56 giờ | Research Only | Chứa âm sắc của người nói từ 63 tỉnh thành, tăng cường khả năng bắt giọng địa phương |
| 4 | **Bud500** | ~500 giờ | Research Only | Âm thanh trích từ podcast, du lịch, ẩm thực với độ nhiễu nền thực tế (background noise) |
| 5 | **PhoAudiobook** | 1.494 giờ | Research Only | Âm đọc chuẩn từ sách nói, phát âm rõ từng thanh điệu, giúp mô hình học chuẩn vần và dấu thanh |
| 6 | **Google FLEURS (vi_vn)** | ~12 giờ | CC BY 4.0 | Bộ benchmark chuẩn quốc tế có cặp audio-transcript chuẩn hóa, dùng làm tập Validation đánh giá WER |
| 7 | **Mozilla Common Voice (vi)** | ~30+ giờ | CC0 | Giọng đọc cộng đồng trên đa thiết bị (điện thoại, tai nghe, laptop), phản ánh đúng micro thực tế |
| 8 | **LibriSpeech (en)** | 960 giờ | CC BY 4.0 | Dùng một tập nhỏ (~20 giờ) làm anchor để duy trì khả năng nhận dạng tiếng Anh thuần túy |

---

## 2. Dữ Liệu Dành Cho BGE-M3 Embedding

Mô hình `BAAI/bge-m3` đã được tiền huấn luyện đa ngôn ngữ rất mạnh mẽ. Mục tiêu fine-tune trong bài toán LifeMate là:
1. **Phân biệt các giả thuyết phiên âm (ASR Hypotheses)**: Phát hiện câu lệnh đúng ngay cả khi ASR sinh ra nhiều câu có âm giống nhau (homophones / acoustic similarity) nhưng nghĩa khác nhau.
2. **Intent & Slot Filling Mapping**: Ánh xạ câu lệnh khẩu ngữ sang API hành động điều khiển thiết bị hoặc trả lời câu hỏi.

| Dataset | Số mẫu | Định dạng cấu trúc | Ứng dụng cụ thể cho BGE-M3 |
| :--- | :---: | :--- | :--- |
| **mMARCO Vietnamese** | 39.78 triệu | Triplet: `(Query, Positive, Negative)` | Huấn luyện tương phản (Contrastive Learning) mức câu quy mô lớn |
| **WikiMatrix EN-VI** | 308.470 cặp | Parallel sentence pairs | Căn chỉnh không gian ngữ nghĩa giữa câu lệnh tiếng Việt và tiếng Anh |
| **PhoMT** | ~3.02 triệu cặp | Parallel bilingual corpus | Tăng cường biểu diễn các cụm từ chêm tiếng Anh (Code-Switching) |
| **UIT-ViQuAD 2.0** | 35.000+ cặp | Passage - Question - Answer | Xây dựng bộ Hard Negatives chất lượng cao để phạt các giả thuyết sai ngữ cảnh |
| **VietEmbed-RAG Science**| 68.567 | Triplet with hard negatives | Mở rộng khả năng truy xuất ngữ nghĩa theo domain chuyên biệt |
| **STSBenchmark-VN** | 5.749 mẫu | Score 0 - 5 similarity | Giữ nguyên làm validation/test để benchmark Pearson & Spearman Correlation |

---

## 3. Chiến Lược Tiền Xử Lý Dữ Liệu Đạt WER < 3%

1. **Chuẩn hóa Unicode**:
   - Chuyển toàn bộ ký tự về dạng Unicode dựng sẵn (**NFC**) để tránh lỗi gõ phím phân rã dấu (NFD).
2. **Chuẩn hóa số và ký hiệu**:
   - Số viết chữ (ví dụ: `8:00` -> `tám giờ`, `100%` -> `một trăm phần trăm`) bằng bộ luật regex và thư viện `vietnamese_numbers`.
3. **Loại bỏ filler words & nhiễu**:
   - Lọc bỏ các từ đệm vô nghĩa kéo dài (à, ừm, ờ...) trong bước evaluate WER để phản ánh đúng nội dung giao tiếp cốt lõi.
4. **Resampling Âm thanh**:
   - Chuẩn hóa toàn bộ âm thanh đầu vào về tần số **16.000 Hz, 1 kênh Mono, PCM 16-bit** theo tiêu chuẩn của Whisper.
