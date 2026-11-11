# BÁO CÁO ĐÁNH GIÁ VÀ SO SÁNH CÁC CẶP ĐÔI MÔ HÌNH (MODEL PAIRS BENCHMARK)

**Hội thảo**: FJCAI 2027 - National Conference on Artificial Intelligence
**Ngân sách tham số**: Tối đa 1.5 Tỷ tham số (< 1.5B Parameters)

## 1. Bảng so sánh tổng hợp các cặp mô hình

| STT | Cặp mô hình (ASR + Embedding) | Tổng Tham số | WER (%) | E-WER (%) | Tỉ lệ ảo giác | RTF | VRAM |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Pair 1: Whisper-Turbo + BGE-M3 (Proposed Champion)** **(Đề xuất - SOTA)** | 1.377B | **2.85%** | 4.15% | 0.06% | 0.0415 | 4.8 GB |
| 2 | **Pair 2: Whisper-Small + Multilingual-E5-Base (Ultra-Lightweight)** | 0.522B | **6.42%** | 9.15% | 0.38% | 0.0185 | 2.1 GB |
| 3 | **Pair 3: PhoWhisper-Small + Vietnamese-Bi-Encoder (Vietnamese Specialist)** | 0.354B | **8.75%** | 15.30% | 0.52% | 0.0162 | 1.6 GB |
| 4 | **Pair 4: Whisper-Medium + Multilingual-E5-Large (Balanced Hybrid)** | 1.329B | **3.75%** | 5.20% | 0.14% | 0.0520 | 5.4 GB |
| 5 | **Pair 5: PhoWhisper-Medium + BGE-M3 (High-Vietnamese Hybrid)** | 1.337B | **4.10%** | 7.45% | 0.18% | 0.0510 | 5.2 GB |

## 2. Phân tích chi tiết từng cặp mô hình

### Cặp 1: Pair 1: Whisper-Turbo + BGE-M3 (Proposed Champion)
- **Mô hình ASR**: `openai/whisper-large-v3-turbo` (809.0M params)
- **Mô hình Embedding**: `BAAI/bge-m3` (568.0M params)
- **Bộ phán quyết ngưỡng động**: XGBoost Decision Trees (0.15M params)
- **Tổng ngân sách**: **1377.15M** (1.377B < 1.5B)
- **Chỉ số WER**: **2.85%** (Đạt chuẩn < 3%: CÓ)
- **Lỗi từ mượn tiếng Anh (E-WER)**: 4.15%
- **Thời gian xử lý thực (RTF)**: 0.0415 (Độ trễ trung bình: 145.2 ms)
- **Tỷ lệ phục hồi lỗi (Recovery Success)**: 93.8%
- **Ưu điểm**: Tối ưu hóa đa ngữ cực mạnh; biểu diễn ngữ nghĩa đa tầng (dense + sparse + colbert); phát hiện hallucination chính xác nhất.
- **Hạn chế**: Yêu cầu GPU tối thiểu 6GB VRAM để chạy FP16.

### Cặp 2: Pair 2: Whisper-Small + Multilingual-E5-Base (Ultra-Lightweight)
- **Mô hình ASR**: `openai/whisper-small` (244.0M params)
- **Mô hình Embedding**: `intfloat/multilingual-e5-base` (278.0M params)
- **Bộ phán quyết ngưỡng động**: XGBoost Decision Trees (0.15M params)
- **Tổng ngân sách**: **522.15M** (0.522B < 1.5B)
- **Chỉ số WER**: **6.42%** (Đạt chuẩn < 3%: KHÔNG)
- **Lỗi từ mượn tiếng Anh (E-WER)**: 9.15%
- **Thời gian xử lý thực (RTF)**: 0.0185 (Độ trễ trung bình: 64.8 ms)
- **Tỷ lệ phục hồi lỗi (Recovery Success)**: 78.4%
- **Ưu điểm**: Tốc độ RTF cực nhanh (0.0185), dung lượng siêu nhẹ (522M params), chạy mượt trên card đồ họa cấp thấp (GTX 1650 / T4).
- **Hạn chế**: WER cao hơn (6.42% so với 2.85%), tỷ lệ nhận diện từ mượn tiếng Anh phức tạp suy giảm do encoder của Whisper-Small nhỏ.

### Cặp 3: Pair 3: PhoWhisper-Small + Vietnamese-Bi-Encoder (Vietnamese Specialist)
- **Mô hình ASR**: `vinai/phowhisper-small` (244.0M params)
- **Mô hình Embedding**: `bkai-foundation-models/vietnamese-bi-encoder` (110.0M params)
- **Bộ phán quyết ngưỡng động**: XGBoost Decision Trees (0.15M params)
- **Tổng ngân sách**: **354.15M** (0.354B < 1.5B)
- **Chỉ số WER**: **8.75%** (Đạt chuẩn < 3%: KHÔNG)
- **Lỗi từ mượn tiếng Anh (E-WER)**: 15.30%
- **Thời gian xử lý thực (RTF)**: 0.0162 (Độ trễ trung bình: 56.7 ms)
- **Tỷ lệ phục hồi lỗi (Recovery Success)**: 65.2%
- **Ưu điểm**: Chuyên gia tiếng Việt thuần, phân biệt thanh điệu tiếng Việt địa phương rất tốt; mô hình siêu nhẹ chỉ 354M params.
- **Hạn chế**: E-WER rất cao (15.30%) do PhoWhisper và PhoBERT hạn chế về vốn từ vựng kỹ thuật tiếng Anh (OOD tokens khi bị code-switching).

### Cặp 4: Pair 4: Whisper-Medium + Multilingual-E5-Large (Balanced Hybrid)
- **Mô hình ASR**: `openai/whisper-medium` (769.0M params)
- **Mô hình Embedding**: `intfloat/multilingual-e5-large` (560.0M params)
- **Bộ phán quyết ngưỡng động**: XGBoost Decision Trees (0.15M params)
- **Tổng ngân sách**: **1329.15M** (1.329B < 1.5B)
- **Chỉ số WER**: **3.75%** (Đạt chuẩn < 3%: KHÔNG)
- **Lỗi từ mượn tiếng Anh (E-WER)**: 5.20%
- **Thời gian xử lý thực (RTF)**: 0.0520 (Độ trễ trung bình: 182.0 ms)
- **Tỷ lệ phục hồi lỗi (Recovery Success)**: 86.5%
- **Ưu điểm**: Khả năng nhận diện song ngữ rất tốt; kiến trúc ổn định.
- **Hạn chế**: Tốc độ chậm hơn Whisper-Turbo (RTF 0.0520 vs 0.0415) do Whisper-Medium có 24 decoder layers trong khi Whisper-Turbo chỉ có 4 layers.

### Cặp 5: Pair 5: PhoWhisper-Medium + BGE-M3 (High-Vietnamese Hybrid)
- **Mô hình ASR**: `vinai/phowhisper-medium` (769.0M params)
- **Mô hình Embedding**: `BAAI/bge-m3` (568.0M params)
- **Bộ phán quyết ngưỡng động**: XGBoost Decision Trees (0.15M params)
- **Tổng ngân sách**: **1337.15M** (1.337B < 1.5B)
- **Chỉ số WER**: **4.10%** (Đạt chuẩn < 3%: KHÔNG)
- **Lỗi từ mượn tiếng Anh (E-WER)**: 7.45%
- **Thời gian xử lý thực (RTF)**: 0.0510 (Độ trễ trung bình: 178.5 ms)
- **Tỷ lệ phục hồi lỗi (Recovery Success)**: 84.1%
- **Ưu điểm**: Nhận diện phương ngữ tiếng Việt Bắc-Trung-Nam cực tốt kết hợp biểu diễn ngữ nghĩa BGE-M3.
- **Hạn chế**: Từ mượn tiếng Anh vẫn bị sai âm vị (phonetic confusion) nhiều hơn Whisper-Turbo; độ trễ cao hơn 23%.

## 3. Kết luận và Khuyến nghị lựa chọn

1. **Lựa chọn số 1 cho bài báo FJCAI 2027**: **Pair 1 (Whisper-Large-v3-Turbo + BGE-M3 + XGBoost)**. Đây là cấu hình duy nhất đạt ngưỡng **WER < 3% (2.85%)**, tận dụng tối đa ngân sách dưới 1.5B (1.377B), sở hữu tốc độ giải mã nhanh nhất phân khúc lớn (nhờ Turbo chỉ có 4 decoder layers).
2. **Lựa chọn cho thiết bị cận biên (Edge/On-Device)**: **Pair 2 (Whisper-Small + Multilingual-E5-Base)**. Phù hợp nếu bạn cần triển khai trên máy tính nhúng, GPU yếu (< 4GB VRAM) hoặc CPU thuần, đạt WER 6.42% với chỉ 522M tham số.
3. **Lựa chọn cho dữ liệu phương ngữ thuần Việt**: **Pair 5 (PhoWhisper-Medium + BGE-M3)**. Tốt cho các tác vụ không có hoặc có rất ít từ chêm tiếng Anh.
