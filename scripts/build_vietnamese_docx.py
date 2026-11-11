import os
import re
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def clean_latex(text):
    if not text:
        return ""
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathcal\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\tau', r'τ', text)
    text = re.sub(r'\\sigma', r'σ', text)
    text = re.sub(r'\\Delta', r'Δ', text)
    text = re.sub(r'\\times', r'×', text)
    text = re.sub(r'\\alpha', r'α', text)
    text = re.sub(r'\\beta', r'β', text)
    text = re.sub(r'\\leq', r'≤', text)
    text = re.sub(r'\\geq', r'≥', text)
    text = re.sub(r'\\%', r'%', text)
    text = text.replace('$', '')
    return text.strip()

def set_cell_margins(cell, top=40, bottom=40, left=60, right=60):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}/>')
    for side, border in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        if border:
            sz = border.get('sz', '4')
            val = border.get('val', 'single')
            color = border.get('color', '000000')
            el = parse_xml(f'<w:{side} {nsdecls("w")} w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>')
            tcBorders.append(el)
        else:
            el = parse_xml(f'<w:{side} {nsdecls("w")} w:val="none"/>')
            tcBorders.append(el)
    tcPr.append(tcBorders)

def render_table(doc, table_lines):
    rows_data = []
    for line in table_lines:
        line = line.strip()
        if not line.startswith('|') or not line.endswith('|'):
            continue
        parts = [c.strip() for c in line.split('|')[1:-1]]
        if all(re.match(r'^:?-+:?$', p) for p in parts if p):
            continue
        rows_data.append(parts)

    if not rows_data:
        return

    num_cols = max(len(r) for r in rows_data)
    # Normalize rows
    for r in rows_data:
        while len(r) < num_cols:
            r.append("")

    tbl = doc.add_table(rows=len(rows_data), cols=num_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    for r_idx, row in enumerate(rows_data):
        is_header = (r_idx == 0)
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            set_cell_margins(cell, top=30, bottom=30, left=45, right=45)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            
            clean_val = clean_latex(val)
            clean_val = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_val)
            r = p.add_run(clean_val)
            r.font.name = "Times New Roman"
            r.font.size = Pt(8.5)

            if is_header:
                r.bold = True
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_cell_shading(cell, "EBF1F5")
                set_cell_borders(cell, top={'sz': '8', 'color': '002060'}, bottom={'sz': '6', 'color': '002060'})
            else:
                if any(val.startswith(x) for x in ['**', 'Whisper-Turbo + BGE-M3', 'Proposed', '+ Toàn bộ']):
                    r.bold = True
                if c_idx == 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                is_last = (r_idx == len(rows_data) - 1)
                b_border = {'sz': '8', 'color': '002060'} if is_last else {'sz': '2', 'color': 'E0E0E0'}
                set_cell_borders(cell, bottom=b_border)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(4)

def create_vietnamese_docx():
    project_root = Path(__file__).resolve().parent.parent
    src_md = project_root / "docs" / "paper" / "FJCAI_2027_LifeMate_CS_ASR.md"
    out_docx = project_root / "docs" / "paper" / "Bai_Bao_Tieng_Viet_FJCAI_2027.docx"
    sub_docx = project_root / "submission" / "Bai_Bao_Tieng_Viet_FJCAI_2027.docx"
    sec_docx = Path("D:/Dataset/LifeMate_CS_ASR/docs/Bai_Bao_Tieng_Viet_FJCAI_2027.docx")
    
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
    r_title = p_title.add_run("KIẾN TRÚC NHẬN DẠNG TIẾNG NÓI CHUYỂN MÃ VIỆT – ANH HỘI THOẠI QUA CƠ CHẾ PHẢN HỒI NGỮ NGHĨA SONG NGỮ VÀ ĐỊNH NGƯỠNG ĐỘNG")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(14)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0, 32, 96)

    # Subtitle / Note
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("(Bản Toàn Văn Tiếng Việt Phục Vụ Thẩm Định Chuyên Gia & Hội Đồng Cố Vấn)")
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(10)
    r_sub.font.italic = True

    # Process markdown lines
    raw_lines = md_text.splitlines()
    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []

    for l in raw_lines:
        stripped = l.strip()

        # Code block handling
        if stripped.startswith("```"):
            if in_code_block:
                # Flush code block
                p_code = doc.add_paragraph()
                p_code.paragraph_format.left_indent = Inches(0.25)
                p_code.paragraph_format.right_indent = Inches(0.25)
                p_code.paragraph_format.space_before = Pt(3)
                p_code.paragraph_format.space_after = Pt(6)
                code_text = "\n".join(code_lines)
                r_c = p_code.add_run(code_text)
                r_c.font.name = "Consolas"
                r_c.font.size = Pt(8.5)
                r_c.font.color.rgb = RGBColor(40, 40, 40)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(l)
            continue

        # Table block handling
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(stripped)
            continue
        else:
            if in_table:
                render_table(doc, table_lines)
                table_lines = []
                in_table = False

        if not stripped:
            continue
        if stripped.startswith("# ") or stripped.startswith("---"):
            continue

        if stripped.startswith("## Tóm tắt"):
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
            text_body = clean_latex(stripped[2:])
            # bold sub-parts
            parts = re.split(r'(\*\*[^*]+\*\*)', text_body)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    r = p.add_run(part[2:-2])
                    r.bold = True
                else:
                    r = p.add_run(part)
                r.font.name = "Times New Roman"
                r.font.size = Pt(10)
            continue

        if re.match(r'^\d+\.\s', stripped):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            text_body = clean_latex(stripped)
            parts = re.split(r'(\*\*[^*]+\*\*)', text_body)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    r = p.add_run(part[2:-2])
                    r.bold = True
                else:
                    r = p.add_run(part)
                r.font.name = "Times New Roman"
                r.font.size = Pt(10)
            continue

        # Standard paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        text_body = clean_latex(stripped)
        parts = re.split(r'(\*\*[^*]+\*\*)', text_body)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                r = p.add_run(part[2:-2])
                r.bold = True
            else:
                r = p.add_run(part)
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)

    # If document ended with table
    if in_table:
        render_table(doc, table_lines)

    # Insert Figures
    p_fig1_h = doc.add_paragraph()
    p_fig1_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_fig1_h.paragraph_format.space_before = Pt(12)
    p_fig1_h.paragraph_format.space_after = Pt(4)
    r_f1 = p_fig1_h.add_run("HÌNH ẢNH THỰC NGHIỆM VÀ ĐỒ THỊ KHOA HỌC")
    r_f1.font.name = "Times New Roman"
    r_f1.font.size = Pt(11)
    r_f1.font.bold = True
    r_f1.font.color.rgb = RGBColor(0, 32, 96)
    
    fig1_path = str(project_root / "docs" / "figures" / "fig1_benchmark.png")
    if os.path.exists(fig1_path):
        doc.add_picture(fig1_path, width=Inches(6.2))
        p_cap1 = doc.add_paragraph()
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap1.paragraph_format.space_after = Pt(10)
        r_c1 = p_cap1.add_run("Hình 1: Đánh giá chuẩn hiệu năng tổng thể trên tập kiểm thử LifeMate CS và biên giới Pareto tối ưu.")
        r_c1.font.name = "Times New Roman"
        r_c1.font.size = Pt(9)
        r_c1.font.italic = True

    fig2_path = str(project_root / "docs" / "figures" / "fig2_redecoding.png")
    if os.path.exists(fig2_path):
        doc.add_picture(fig2_path, width=Inches(6.2))
        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap2.paragraph_format.space_after = Pt(10)
        r_c2 = p_cap2.add_run("Hình 2: Mặt phẳng phân định ngưỡng động XGBoost và phân phối khôi phục lỗi qua vòng lặp phản hồi kín.")
        r_c2.font.name = "Times New Roman"
        r_c2.font.size = Pt(9)
        r_c2.font.italic = True

    doc.save(str(out_docx))
    doc.save(str(sub_docx))
    if sec_docx.parent.exists():
        doc.save(str(sec_docx))
    print(f"[OK] Generated Vietnamese Word documents: {out_docx}")

if __name__ == '__main__':
    create_vietnamese_docx()
