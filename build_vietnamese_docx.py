import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_vietnamese_docx():
    src_md = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR.md"
    out_docx = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\Bai_Bao_Tieng_Viet_FJCAI_2027.docx"
    root_docx = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\Bai_Bao_Tieng_Viet_FJCAI_2027.docx"
    sec_docx = r"D:\Dataset\LifeMate_CS_ASR\docs\Bai_Bao_Tieng_Viet_FJCAI_2027.docx"
    
    with open(src_md, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = docx.Document()
    
    # Page setup A4
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
        # Header
        header = section.header
        p_hdr = header.paragraphs[0]
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_hdr = p_hdr.add_run("Hội thảo khoa học Quốc gia về Trí tuệ nhân tạo 2027 (FJCAI 2027) - Đà Nẵng, 8-9/4/2027")
        r_hdr.font.name = "Times New Roman"
        r_hdr.font.size = Pt(8.5)
        r_hdr.font.italic = True
        r_hdr.font.color.rgb = RGBColor(100, 100, 100)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("KIẾN TRÚC NHẬN DẠNG TIẾNG NÓI CHUYỂN MÃ VIỆT – ANH DƯỚI 1.5 TỶ THAM SỐ KẾT HỢP TÁI XẾP HẠNG NGỮ NGHĨA BGE-M3 VÀ THIẾT LẬP NGƯỠNG ĐỘNG XGBOOST")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(14)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0, 32, 96)

    # Subtitle / Note
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run("(Bản Toàn Văn Tiếng Việt Phục Vụ Thẩm Định Chuyên Gia & Hội Đồng Cố Vấn)")
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(10)
    r_sub.font.italic = True

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(4)
    r_auth = p_auth.add_run("Phùng Nguyễn Như Bách¹*, Đỗ Phú Hưng²*, Võ Lê Quang Minh³*")
    r_auth.font.name = "Times New Roman"
    r_auth.font.size = Pt(11)
    r_auth.font.bold = True

    p_aff = doc.add_paragraph()
    p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_aff.paragraph_format.space_after = Pt(14)
    r_aff = p_aff.add_run(
        "¹ Khoa Công nghệ Thông tin, Trường Đại học Sư phạm Quy Nhơn (QNU)\n"
        "² Khoa Công nghệ Thông tin & Truyền thông, Trường ĐH Khoa học và Công nghệ Hà Nội (USTH)\n"
        "³ Khoa Khoa học Máy tính, Trường ĐH CNTT & TT Việt – Hàn (VKU), Đại học Đà Nẵng\n"
        "Email: bachpnn@qnu.edu.vn, hung.dp@usth.edu.vn, minhvlq.26itb@vku.udn.vn"
    )
    r_aff.font.name = "Times New Roman"
    r_aff.font.size = Pt(9.5)

    # Process markdown paragraphs
    lines = md_text.splitlines()
    in_abstract = False
    
    for l in lines:
        stripped = l.strip()
        if not stripped:
            continue
        if stripped.startswith("# ") or stripped.startswith("---"):
            continue
        if stripped.startswith("## Tóm tắt"):
            in_abstract = True
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run("TÓM TẮT")
            r.font.name = "Times New Roman"
            r.font.size = Pt(11)
            r.font.bold = True
            continue
        if stripped.startswith("## "):
            sec_title = stripped.replace("## ", "").upper()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(sec_title)
            r.font.name = "Times New Roman"
            r.font.size = Pt(11.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0, 32, 96)
            continue
        if stripped.startswith("### "):
            subsec_title = stripped.replace("### ", "")
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(subsec_title)
            r.font.name = "Times New Roman"
            r.font.size = Pt(10.5)
            r.font.bold = True
            r.font.italic = True
            continue
        if stripped.startswith("* ") or stripped.startswith("- "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run("• " + stripped[2:])
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            continue
        if stripped.startswith("1. ") or stripped.startswith("2. ") or stripped.startswith("3. ") or stripped.startswith("4. "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run(stripped)
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            continue
        
        # Standard paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(stripped)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)

    # Insert Figures
    p_fig1_h = doc.add_paragraph()
    p_fig1_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_fig1_h.paragraph_format.space_before = Pt(10)
    p_fig1_h.paragraph_format.space_after = Pt(4)
    r_f1 = p_fig1_h.add_run("HÌNH ẢNH THỰC NGHIỆM VÀ ĐỒ THỊ KHOA HỌC")
    r_f1.font.name = "Times New Roman"
    r_f1.font.size = Pt(11)
    r_f1.font.bold = True
    
    fig1_path = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\fig1_benchmark.png"
    if os.path.exists(fig1_path):
        doc.add_picture(fig1_path, width=Inches(6.2))
        p_cap1 = doc.add_paragraph()
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap1.paragraph_format.space_after = Pt(10)
        r_c1 = p_cap1.add_run("Hình 1: Đánh giá chuẩn hiệu năng tổng thể trên tập kiểm thử LifeMate CS và biên giới Pareto tối ưu.")
        r_c1.font.name = "Times New Roman"
        r_c1.font.size = Pt(9)
        r_c1.font.italic = True

    fig2_path = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\fig2_redecoding.png"
    if os.path.exists(fig2_path):
        doc.add_picture(fig2_path, width=Inches(6.2))
        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap2.paragraph_format.space_after = Pt(10)
        r_c2 = p_cap2.add_run("Hình 2: Mặt phẳng phân định ngưỡng động XGBoost và phân phối khôi phục lỗi qua vòng lặp phản hồi kín.")
        r_c2.font.name = "Times New Roman"
        r_c2.font.size = Pt(9)
        r_c2.font.italic = True

    doc.save(out_docx)
    doc.save(root_docx)
    doc.save(sec_docx)
    print(f"[OK] Generated Vietnamese Word documents: {out_docx}")

if __name__ == '__main__':
    create_vietnamese_docx()
