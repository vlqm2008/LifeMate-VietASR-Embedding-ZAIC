# THƯ GỬI ĐỒNG TÁC GIẢ & ĐỒNG NGHIỆP — BẢN THẢO BÀI BÁO FJCAI 2027

**Kính gửi:**
- TS. Phùng Nguyễn Như Bách (Trường Đại học Sư phạm Quy Nhơn - QNU)
- ThS. Đỗ Phú Hưng (Trường Đại học Khoa học và Công nghệ Hà Nội - USTH)
- Cùng quý Thầy Cô trong nhóm nghiên cứu,

**Người gửi:** Võ Lê Quang Minh (Trường ĐH CNTT & TT Việt – Hàn, Đại học Đà Nẵng)  
**Tiêu đề:** [FJCAI 2027] Gửi bản thảo Camera-Ready hoàn thiện (Chuẩn IEEE 6 trang, Đo thật trên GPU, Cam kết Liêm chính học thuật QĐ 2557) để rà soát lần cuối trước khi nộp cổng CMT3.

---

### Kính gửi các anh / các thầy,

Nhóm nghiên cứu đã hoàn tất việc tinh chỉnh toàn diện bài báo khoa học gửi Hội thảo Khoa học Quốc gia về Trí tuệ Nhân tạo 2027 (**FJCAI 2027 - Đà Nẵng, 8-9/4/2027**). Em xin gửi bản thảo hoàn chỉnh cùng các số liệu thực nghiệm đo đạc thực tế tại chỗ để các anh xem xét, phản hồi và phê duyệt trước khi chính thức bấm nộp trên cổng CMT3.

---

### 1. Tóm Tắt Các Điểm Cải Tiến & Cập Nhật Lớn

1. **Tuyệt đối tuân thủ Liêm chính Học thuật (Quyết định số 2557/QĐ-BKHCN của Bộ KH&CN):**
   - Loại bỏ hoàn toàn mọi số liệu ước lượng/giả lập.
   - Toàn bộ kết quả Baseline đã được đo đạc thực tế trực tiếp trên card đồ họa máy trạm cá nhân (**NVIDIA GeForce GTX 1660 Super**, 6.0 GB VRAM, CUDA 12.4, PyTorch 2.6.0) với 60 tệp âm thanh thực tế từ tập dữ liệu cuộc thi Zalo AI Challenge LifeMate.
   - Kết quả đo thật: Baseline Zero-shot Whisper-Large-v3-Turbo đạt **WER 28.68%** (thời gian trễ $RTF = 0.710$). Mô hình đề xuất đạt **WER 2.85%** ($E\text{-}WER = 4.15\%$), giảm đến **90.1%** lỗi thực tế.
   - Nhật ký đo đạc thô từng tệp được lưu vết minh bạch tại tệp đính kèm `FJCAI_DEFENSE_REAL_BENCHMARK.json`.
2. **Bổ sung Phân tích Âm vị Thực chiến (Empirical Qualitative Case Studies):**
   - Phân tích chi tiết 4 ca lỗi thực tế từ dữ liệu LifeMate:
     - `train_1687`: Ranh giới tên riêng phần mềm *"Brightpath"* bị biến âm thành *"Pride Park"*.
     - `train_1686`: Hiện tượng ép âm từ mượn đời thường *"London"* $\rightarrow$ *"lên đình"*, *"sale"* $\rightarrow$ *"sau"*.
     - `train_1474`: Lỗi hỏng hậu tố tên hoạt chất dược phẩm y tế *"Omeprazole"* $\rightarrow$ *"Omeprazone"*.
     - `train_0236`: Lỗi nuốt âm từ viết tắt và danh xưng *"sếp Dũng"* $\rightarrow$ *"xem dũ"*.
3. **Định dạng Chuẩn IEEE Camera-Ready Đúng 6 Trang:**
   - Bản PDF và Word tiếng Anh đã được dàn trang tự động chính xác **đúng 6 trang** (không thừa, không thiếu, không rớt dòng), bảo đảm 100% quy chuẩn ngặt nghèo của IEEE và ban tổ chức FJCAI.
4. **Bản Toàn Văn Tiếng Việt:**
   - Được định dạng lại chuyên nghiệp với 5 bảng số liệu Word chuẩn, đồ thị sắc nét và đầy đủ điều khoản đạo đức nghiên cứu, rất thuận tiện để đọc và trao đổi chuyên môn.

---

### 2. Danh Mục Tài Liệu Gửi Kèm (Trong Gói ZIP)

1. `FJCAI_2027_LifeMate_CS_ASR_EN.pdf`: Bản PDF nộp chính thức (Chuẩn IEEE 2 cột, đúng 6 trang).
2. `FJCAI_2027_LifeMate_CS_ASR_EN.docx`: Bản Word OpenXML tiếng Anh (để chỉnh sửa nếu cần).
3. `FJCAI_2027_LifeMate_CS_ASR_EN.doc`: Bản Word 97-2003 (tương thích các cổng nộp bài cũ).
4. `Bai_Bao_Tieng_Viet_FJCAI_2027.docx`: Bản toàn văn tiếng Việt có định dạng bảng biểu và hình ảnh.
5. `FJCAI_2027_PRESENTATION_SLIDES.md`: Khung slide báo cáo 15 trang phục vụ thuyết trình tại hội thảo.
6. `FJCAI_DEFENSE_REAL_BENCHMARK.json`: File nhật ký đo kiểm thô 60 mẫu trên GPU thực tế.
7. `CLAIM_EVIDENCE_MATRIX.md`: Bảng ma trận đối soát 100% luận điểm khoa học và bằng chứng thực nghiệm.

---

### 3. Nhờ Các Anh Rà Soát & Xác Nhận (Checklist Cần Phản Hồi)

Kính nhờ anh Bách và anh Hưng kiểm tra giúp em các thông tin sau:
- [ ] **Thứ tự và định dạng đồng tác giả:**
  - 1. Phung-Nguyen-Nhu Bach
  - 2. Do-Phu Hung
  - 3. Vo-Le-Quang Minh (Tác giả liên hệ)
  *(Các chi tiết email và đơn vị công tác cá nhân đã được ẩn trên bản thảo công khai để bảo vệ tính ẩn danh phản biện)*
- [ ] **Mục Lời cảm ơn (Acknowledgment):** Các đơn vị tài trợ tính toán (PTN Điện toán hiệu năng cao QNU, Khoa CNTT&TT USTH, Khoa KHMT VKU).
- [ ] **Phê duyệt bản nộp:** Xác nhận đồng thuận nội dung bài báo và cam kết liêm chính khoa học theo Quyết định 2557/QĐ-BKHCN để tiến hành nộp bài.

*Kho lưu trữ mã nguồn mở:*  
https://github.com/AIVIETNAM-AIO-MinhHocAyAi/LifeMate-VietASR-Embedding-ZAIC.git

Kính chúc các anh sức khỏe và công tác tốt!  
Trân trọng,  
**Võ Lê Quang Minh**
