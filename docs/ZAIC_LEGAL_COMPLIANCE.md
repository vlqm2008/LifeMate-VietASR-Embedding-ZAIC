# Quy Định Pháp Lý & Điều Khoản Sử Dụng Dữ Liệu Zalo AI Challenge

Tài liệu hướng dẫn tuân thủ các quy định pháp lý, bản quyền và điều khoản sử dụng của **Zalo AI Challenge (ZAIC)** khi sử dụng tập dữ liệu bài thi (ví dụ: LifeMate) cho mục đích nghiên cứu khoa học và viết báo cáo khoa học (Research Paper).

---

## 1. Cơ Sở Pháp Lý & Tinh Thần Quy Chế Cuộc Thi

Theo quy chế chính thức của Zalo AI Challenge (Regulations & Terms of Service):
1. **Mục đích hỗ trợ cộng đồng AI**:
   - BTC Zalo AI Challenge khuyến khích các đội thi và các nhà nghiên cứu khai thác dữ liệu sau cuộc thi để phục vụ nghiên cứu khoa học, cải tiến các mô hình mã nguồn mở và đóng góp cho cộng đồng trí tuệ nhân tạo Việt Nam.
2. **Quyền sở hữu trí tuệ**:
   - Bản quyền dữ liệu gốc, các bản ghi âm thanh và nhãn gán thuộc về Ban tổ chức Zalo AI Challenge và các đối tác đồng hành.
   - Thí sinh được cấp quyền sử dụng có giới hạn (non-exclusive, revocable license) cho mục đích học thuật và phi thương mại.

---

## 2. Các Điều Kiện Đi Kèm Bắt Buộc Khi Viết Báo Cáo / Paper

Để công bố báo cáo khoa học hoặc bài báo nghiên cứu hợp lệ mà không vi phạm quy chế cuộc thi:

### 2.1. Nghĩa vụ Trích dẫn & Ghi nhận (Citation & Acknowledgement)
- Trong phần **Acknowledgements** hoặc **Data Description** của bài báo, bắt buộc phải ghi rõ nguồn gốc dữ liệu từ Zalo AI Challenge:
  > *"The authors would like to thank Zalo AI Challenge 2026 for providing the dataset for the LifeMate task. This research was conducted using the competition dataset strictly for academic benchmarking."*
- Nếu bài thi có xuất bản technical report từ BTC, nhóm tác giả cần trích dẫn trực tiếp bài báo đó trong danh mục tham khảo.

### 2.2. Tuyệt Đối Không Phát Tán Nguyên Trạng Dữ Liệu Gốc (No Raw Data Redistribution)
- **Nghiêm cấm**:
  - Upload toàn bộ hoặc một phần file âm thanh gốc (`.wav`, `.mp3`) hoặc file nhãn gốc của ban tổ chức lên các kho lưu trữ công cộng (GitHub public, Google Drive chia sẻ công khai, Hugging Face Dataset public không cài gated access).
- **Được phép**:
  - Công bố mô hình đã huấn luyện (model weights / checkpoint sau fine-tuning).
  - Công bố mã nguồn xử lý pipeline, công thức tính toán và số liệu đo đạc (WER, CER, F1-Score, Loss curve).
  - Cung cấp script tái lập kết quả (reproducibility script) yêu cầu người dùng phải tự tải data từ nguồn chính thức của Zalo AI Challenge.

### 2.3. Bảo Vệ Dữ Liệu Cá Nhân & Quyền Riêng Tư (Privacy & Anonymization)
- Mọi thông tin nhận dạng cá nhân (PII - Personally Identifiable Information) nếu vô tình xuất hiện trong âm thanh (tên người dùng cụ thể, số điện thoại, địa chỉ nhà) phải được xử lý làm mờ, gắn nhãn ẩn danh (ví dụ: `<PERSON>`, `<PHONE>`) trong các trích dẫn mẫu của báo cáo.

### 2.4. Tính Phi Thương Mại (Non-commercial Use)
- Nghiêm cấm đem bán trực tiếp tập dữ liệu hoặc sử dụng trực tiếp mô hình huấn luyện độc quyền trên tập data Zalo để kinh doanh sinh lời mà chưa có thỏa thuận thương mại bằng văn bản với đơn vị chủ quản.

---

## 3. Khuyến Nghị Khi Đẩy Code Lên GitHub

1. Luôn thêm file `.gitignore` để tự động loại bỏ các thư mục chứa dữ liệu thô:
   ```gitignore
   # Data and checkpoints
   data/
   *.wav
   *.mp3
   *.flac
   checkpoints/
   *.pt
   *.bin
   *.safetensors
   ```
2. Trong file README của GitHub, chỉ chia sẻ mã nguồn, liên kết checkpoint đã huấn luyện (Hugging Face) và tài liệu học thuật.
