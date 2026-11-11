import fitz # PyMuPDF
import os

pdf_path = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR_EN.pdf"
doc = fitz.open(pdf_path)
print(f"Rendering {len(doc)} pages...")

docs_dir = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs"
sec_docs_dir = r"D:\Dataset\LifeMate_CS_ASR\docs"
art_dir = r"C:\Users\Admin\.gemini\antigravity\brain\d96ec7f7-0030-4eae-b228-2dcd50fcbb50"

for idx, page in enumerate(doc, 1):
    pix = page.get_pixmap(dpi=150)
    out_name = f"page_{idx}.png"
    p1 = os.path.join(docs_dir, out_name)
    p2 = os.path.join(sec_docs_dir, out_name)
    p3 = os.path.join(art_dir, out_name)
    pix.save(p1)
    pix.save(p2)
    pix.save(p3)
    print(f"Saved {out_name} ({pix.width}x{pix.height})")

print("Done rendering pages.")
