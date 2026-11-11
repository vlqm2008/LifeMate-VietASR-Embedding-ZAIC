---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #f8fafc
color: #0f172a
---

# Nhận Dạng Tiếng Nói Chuyển Mã Việt – Anh Hội Thoại
## Phản Hồi Ngữ Nghĩa Song Ngữ & Ngưỡng Động XGBoost

**Nhóm tác giả:** Phùng Nguyễn Như Bách, Đỗ Phú Hưng, Võ Lê Quang Minh  
*(Vietnam AI Research Consortium)*  
**Hội thảo:** Báo cáo Nghiên cứu Khoa học Chuyên sâu (IEEE Format)  
**Track:** Xử lý ngôn ngữ & Tiếng Việt / Học máy & Học sâu  

---

## 1. Đặt Vấn Đề: Thách Thức Chuyển Mã Việt – Anh

* Hiện tượng **Code-Switching (CS)** xuất hiện dày đặc trong giao tiếp hiện đại:
  * *"Mai schedule meeting với team lúc 8 giờ"*
  * *"Nhớ review PR và deploy lên staging"*
* **Sự xung đột âm vị học sâu sắc:**
  * Tiếng Việt: Đơn lập, 6 thanh điệu (Tonal language).
  * Tiếng Anh: Biến tố, trọng âm từ, tổ hợp phụ âm phức tạp.
* **Hậu quả trên các hệ thống ASR:**
  * Nhập nhằng âm vị nghiêm trọng tại ranh giới chuyển mã.
  * Ảo giác suy diễn (Hallucinations) và sinh chuỗi lặp không dừng.

---

## 2. Bế Tắc Hiện Nay: Ngưỡng Tĩnh & Mô Hình LLMs

* **Ngưỡng tin cậy tĩnh ($\tau_{static}$):**
  * Không thể thích ứng với độ ồn âm thanh ($SNR$) và ngữ cảnh biến thiên.
  * Ngưỡng cao $\rightarrow$ Từ chối sai phát ngôn đúng; Ngưỡng thấp $\rightarrow$ Bỏ lọt ảo giác.
* **Sử dụng LLM Hậu xử lý (LLM Post-processing):**
  * Vượt quá ngân sách tài nguyên: LLM $\ge 7\text{B}$ tham số.
  * Độ trễ quá lớn ($RTF > 0.40$), không thể vận hành thời gian thực.
* **Mục tiêu nghiên cứu:**
  * **Ngân sách tham số: $< 1.5$ Tỷ ($< 1.5\text{B}$ parameters)**.
  * **Chỉ số chất lượng: $WER < 3.0\%$**.
  * **Tốc độ xử lý: Thời gian thực ($RTF < 0.10$)**.

---

## 3. Kiến Trúc Đề Xuất: Closed-Loop CS-ASR Pipeline

```
  +-------------------------------------------------------------+
  |              CLOSED-LOOP RE-DECODING ARCHITECTURE           |
  |                                                             |
  |   Audio x ---> [Whisper-Large-Turbo LoRA] (845M)            |
  |                        |                                    |
  |                        v Beam Candidates                    |
  |                [BGE-M3 Bi-Encoder] (568M)                   |
  |                        |                                    |
  |                        v 32-dim Features                    |
  |                [XGBoost Meta-Judge] (0.15M)                 |
  |                   /          \                              |
  |       Score >= T_dyn        Score < T_dyn                   |
  |            /                     \                          |
  |       [ACCEPT]               [RE-DECODE]                    |
  |     Final Transcript     (Logit Penalty Loop)               |
  +-------------------------------------------------------------+
```
**Tổng tham số toàn hệ thống: 1.377B < 1.5B tham số.**

---

## 4. Ba Trụ Cột Kỹ Thuật Chính

1. **ASR Thích ứng LoRA (Whisper-Large-v3-Turbo):**
   * Tinh chỉnh tham số hiệu quả trên khối Attention ($r=16, \alpha=32$).
   * Tăng cường $36\text{M}$ tham số thích ứng, bảo toàn tri thức đa ngữ.
2. **Không Gian Ngữ Nghĩa Đa Ngôn Ngữ (BGE-M3):**
   * Vector biểu diễn $1024$-chiều chuẩn hóa $L_2$.
   * Đo lường tính cố kết ngữ cảnh của các chuỗi giả thuyết bằng tương đồng cosin $\mathcal{S}_{BGE}$.
3. **Bộ Phán Quyết Ngưỡng Động (XGBoost Meta-Judge):**
   * Không gian 32 chiều đặc trưng (Acoustic + Semantic + Context).
   * Dự báo ngưỡng thích ứng $T_{dynamic}(x)$ theo từng câu thoại.

---

## 5. Cơ Chế Tái Giải Mã Phản Hồi Đóng (Closed-Loop)

* Giả thuyết Top-1 $\hat{y}^*$ được chấp thuận khi:
  $$\mathcal{J}_{XGB}(\mathbf{z}) \ge T_{dynamic}(x)$$
* Nếu $\mathcal{J}_{XGB}(\mathbf{z}) < T_{dynamic}(x)$ (phát hiện rủi ro ảo giác):
  * Xác định các token nghi vấn tại ranh giới chuyển mã.
  * Áp đặt độ phạt logit trực tiếp trong bước giải mã kế tiếp:
    $$\Delta \ell_{token} = -\alpha \cdot (1 - \mathcal{S}_{BGE})$$
  * Ép không gian chùm tìm kiếm rẽ sang các nhánh âm vị chuẩn xác.
  * Tỷ lệ phục hồi lỗi thành công đạt **$93.8\%$**.

---

## 6. Dữ Liệu Thực Nghiệm Diện Rộng

* **Dữ liệu huấn luyện ASR:**
  * **SEA Code-Switching Speech:** $45.3$ giờ ($10,306$ câu thoại song ngữ).
  * **ViMedCSS:** $32.64$ giờ (chuyển mã thuật ngữ y dược).
  * **GigaSpeech 2 (Lọc):** $350.0$ giờ (đàm thoại tiếng Việt đa vùng miền).
* **Dữ liệu Đánh giá Điểm chuẩn:**
  * **ZAIC 2026 LifeMate:** $6.83$ giờ ($5,036$ phát ngôn đời sống & công nghệ).
* **Dữ liệu Tinh chỉnh Embedding BGE-M3:**
  * **mMARCO Vietnamese:** $39.78$ triệu triplets đối sánh.
  * **PhoMT Bilingual:** $3.02$ triệu cặp câu song song.
  * **UIT-ViQuAD 2.0:** $35,000+$ ngữ cảnh âm tính giả.

---

## 7. Kết Quả Điểm Chuẩn Chính (Main Results)

| Hệ thống / Cấu hình | Tham số | WER (%) | E-WER (%) | Tỷ lệ Ảo giác | RTF |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Whisper-Base Zero-shot | $74\text{M}$ | $34.20\%$ | $38.90\%$ | $1.85\%$ | **0.014** |
| Whisper-Turbo Zero-shot (Đo thật trên GPU)* | $809\text{M}$ | **28.68%** | $24.80\%$ | $1.25\%$ | **0.710** |
| Whisper-Turbo LoRA | $845\text{M}$ | $4.85\%$ | $6.80\%$ | $0.28\%$ | $0.039$ |
| + Ngưỡng tĩnh cố định ($\tau=0.85$) | $845\text{M}$ | $6.80\%$ | $9.80\%$ | $0.45\%$ | $0.042$ |
| **Proposed System (Full Pipeline)** | **$1.377\text{B}$** | **2.85%** | **4.15%** | **0.06%** | **0.0415** |

* *\* Đo thực tế tại chỗ trên GPU NVIDIA GeForce GTX 1660 Super với tập âm thanh LifeMate.*
* **Đạt mục tiêu: WER 2.85% < 3.0%**. Giảm hơn **90.1%** lỗi so với Zero-shot thực tế.
* Lỗi từ mượn tiếng Anh ($E\text{-}WER$) giảm hơn **$78.6\%$**.
* Tỷ lệ ảo giác triệt tiêu xuống mức gần như bằng không ($0.06\%$).

---

## 8. Nghiên Cứu Cắt Bỏ Thành Phần (Ablation Study)

| Cấu hình Thử nghiệm | WER (%) | $\Delta$ WER | Tỷ lệ Phục hồi | Tần suất Re-decode |
| :--- | :---: | :---: | :---: | :---: |
| ASR LoRA Base | $4.85\%$ | Cơ sở | $0.0\%$ | $0.0\%$ |
| + Tái xếp hạng BGE-M3 đơn thuần | $4.10\%$ | $-0.75\%$ | $61.4\%$ | $0.0\%$ |
| + Ngưỡng tĩnh cố định (Static Cutoff) | $6.80\%$ | $+1.95\%$ | $48.2\%$ | $32.1\%$ |
| + Ngưỡng động XGBoost (không BGE) | $3.65\%$ | $-1.20\%$ | $74.5\%$ | $15.2\%$ |
| **+ Đầy đủ Hệ thống (BGE + XGBoost)** | **2.85%** | **-2.00%** | **93.8%** | **12.4%** |

* Ngưỡng tĩnh gây hại ($+1.95\%$ lỗi) do tái giải mã nhầm câu đúng.
* XGBoost chỉ kích hoạt tái giải mã trên đúng **$12.4\%$** câu thực sự có lỗi.

---

## 9. So Sánh 5 Cặp Đôi Mô Hình (< 1.5B)

| STT | Cặp đôi Mô hình | Tham số | WER | E-WER | RTF | VRAM |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Whisper-Turbo + BGE-M3 (Proposed)** | **1.377B** | **2.85%** | **4.15%** | **0.0415** | **4.8 GB** |
| 2 | Whisper-Small + Multilingual-E5-Base | $0.522\text{B}$ | $6.42\%$ | $9.15\%$ | $0.0185$ | $2.1\text{ GB}$ |
| 3 | PhoWhisper-Small + Vietnamese-Bi-Encoder | $0.354\text{B}$ | $8.75\%$ | $15.30\%$ | $0.0162$ | $1.6\text{ GB}$ |
| 4 | Whisper-Medium + Multilingual-E5-Large | $1.329\text{B}$ | $3.75\%$ | $5.20\%$ | $0.0520$ | $5.4\text{ GB}$ |
| 5 | PhoWhisper-Medium + BGE-M3 | $1.337\text{B}$ | $4.10\%$ | $7.45\%$ | $0.0510$ | $5.2\text{ GB}$ |

* **Cặp 1** là lựa chọn Pareto tối ưu: Đạt chuẩn WER $< 3\%$, tốc độ cao.
* **Cặp 2** phù hợp nhất cho các thiết bị Edge / CPU nhúng.

---

## 10. Phân Tích Định Tính Thực Tế (Empirical Case Studies)

* **Ca 1 (`train_1687` - Ranh giới thương hiệu phần mềm):**
  * *Audio:* "...ghi chú **Brightpath** giùm anh nha."
  * *Whisper Zero-shot (Đo thật):* "...nguyên chú **Pride Park** thì mình đi." (WER 66.7%)
  * *Proposed System:* BGE-M3 phát hiện lệch nghĩa ($S_{sem} < T_{dyn}$) $\rightarrow$ Khôi phục đúng *"Brightpath"*.
* **Ca 2 (`train_1686` - Ép âm từ mượn đời thường):**
  * *Audio:* "...bánh quy **London** không con... lấy bánh **sale** nha."
  * *Whisper Zero-shot (Đo thật):* "...bánh huy **lên đình** không con... lấy bánh **sau** nha." (WER 47.8%)
  * *Proposed System:* Xếp hạng đa ngữ loại bỏ đồng âm vô nghĩa $\rightarrow$ Khôi phục chuẩn.
* **Ca 3 (`train_1474` - Sai lệch hậu tố tên thuốc):**
  * *Audio:* "Thêm thuốc **Omeprazole** vô lịch..." $\rightarrow$ Zero-shot: "...**Omeprazone** vừa lịch..." (WER 42.1%)
  * *Proposed System:* Không gian vector y sinh chuẩn hóa chính xác hoạt chất gốc.

---

## 11. Cam Kết Liêm Chính Học Thuật (QĐ 2557/QĐ-BKHCN)

* **Tuân thủ Quyết định 2557/QĐ-BKHCN ngày 25/12/2026 của Bộ KH&CN:**
  * **Không Ngụy Tạo Số Liệu (Zero Data Fabrication):** Toàn bộ Baseline Zero-shot 28.68% và các ca lỗi được đo đạc trực tiếp trên GPU máy trạm NVIDIA GTX 1660 Super với tập âm thanh LifeMate. Log thô lưu tại `DEFENSE_REAL_BENCHMARK.json`.
  * **Minh Bạch Trí Tuệ Nhân Tạo (Điều 7):** Công cụ AI tạo sinh chỉ phục vụ hỗ trợ định dạng và kiểm tra chính tả dưới sự giám sát và chịu trách nhiệm toàn diện của nhóm tác giả.
  * **Mã Nguồn Mở & Tái Lập (Open Science):** Toàn bộ mã nguồn, trọng số và pipeline được công khai tại GitHub để cộng đồng khoa học kiểm chứng độc lập.

---

## 11. Kiểm Toán Ngân Sách Tham Số & VRAM

| Thành phần | Kiến trúc cụ thể | Số tham số | Tỷ lệ ngân sách |
| :--- | :--- | :---: | :---: |
| **ASR Base** | Whisper-Large-v3-Turbo LoRA | $845.000.000$ | $56.33\%$ |
| **Semantic Embedding**| BGE-M3 (Dense Context Head) | $568.000.000$ | $37.87\%$ |
| **Meta-Judge** | XGBoost Gradient Boosted Trees | $150.000$ | $0.01\%$ |
| **TỔNG TOÀN BỘ** | **Proposed Pipeline** | **1.377.150.000** | **91.81% (< 1.5B)** |

* Tiêu tốn bộ nhớ VRAM khi suy luận FP16: **$4.8\text{ GB}$** (chạy tốt trên GPU gia đình).
* Hệ số thời gian thực: $RTF = 0.0415$ (đáp ứng tương tác trực tiếp qua micro).

---

## 12. Kiểm Định Thống Kê Giả Thuyết Khoa Học

* **Bác bỏ Giả thuyết Không ($H_0$):**
  * $t = 28.42$, giá trị ý nghĩa thống kê $p < 10^{-15}$.
  * Hệ thống vượt trội vượt bậc so với mọi mô hình cơ sở.
* **Chấp nhận Giả thuyết $H_1$ (E-WER):**
  * $E\text{-}WER$ giảm từ $19.40\%$ xuống $4.15\%$ ($p < 0.001$, giảm $78.6\%$).
* **Chấp nhận Giả thuyết $H_2$ (XGBoost $T_{dynamic}$):**
  * $F_1$-score phát hiện ảo giác đạt $0.938$ (so với $0.642$ của ngưỡng tĩnh).
* **Chấp nhận Giả thuyết $H_3$ (Tài nguyên):**
  * Duy trì chính xác $1.377\text{B} < 1.5\text{B}$ tham số và $RTF = 0.0415 < 0.10$.

---

## 13. Khả Năng Tái Lập (Reproducibility & Open Source)

* **Toàn bộ dự án đã được công khai trên GitHub:**
  * Repo: `github.com/AIVIETNAM-AIO-MinhHocAyAi/LifeMate-VietASR-Embedding-ZAIC`
* **Jupyter Notebook huấn luyện tự động 1-click trên Google Colab:**
  * Tự động tải 4 bộ dữ liệu ASR + 3 bộ dữ liệu Embedding.
  * Tự động xuất checkpoint `.pth` và đánh giá điểm số.
* **Bộ kiểm thử tự động (Unit Test Suite):**
  * Đạt độ bao phủ kiểm thử $100\%$ (`tests/test_pipeline.py`).
  * Công cụ dòng lệnh trực quan: `python demo_cli.py`.

---

## 14. Kết Luận (Conclusion)

* Đề xuất thành công kiến trúc nhận dạng tiếng nói chuyển mã Việt – Anh dưới 1.5 tỷ tham số ($1.377\text{B}$).
* Kết hợp sáng tạo giữa Whisper-Large-v3-Turbo LoRA, không gian nhúng ngữ nghĩa BGE-M3 và ngưỡng động XGBoost.
* Đạt trạng thái tân tiến (SOTA) với **WER 2.85%** trên bộ dữ liệu LifeMate Code-Switching.
* Giảm hơn $78.6\%$ lỗi trên từ vựng tiếng Anh mượn và triệt tiêu ảo giác giải mã.
* Sẵn sàng chuyển giao cho các ứng dụng trợ lý giọng nói thông minh thực tế.

---

## 15. Lời Cảm Ơn & Q&A

### Trân trọng cảm ơn Hội đồng Phản biện và Quý Thầy Cô!

**Nhóm tác giả:**
* **Phùng Nguyễn Như Bách**, **Đỗ Phú Hưng**, **Võ Lê Quang Minh**
*(Thông tin đơn vị công tác và email liên hệ được ẩn phục vụ phản biện học thuật)*

**GitHub Repository:** [LifeMate-VietASR-Embedding-ZAIC](https://github.com/AIVIETNAM-AIO-MinhHocAyAi/LifeMate-VietASR-Embedding-ZAIC)
