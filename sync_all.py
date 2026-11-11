import shutil
import os

src_dir = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC"
dest_dir = r"D:\Dataset\LifeMate_CS_ASR"

src_docs = os.path.join(src_dir, "docs")
dest_docs = os.path.join(dest_dir, "docs")

# 1. Sync camera-ready files
exts = [".docx", ".doc", ".pdf", ".tex"]
for ext in exts:
    for name in ["FJCAI_2027_LifeMate_CS_ASR_EN", "FJCAI_2027_LifeMate_CS_ASR"]:
        f = f"{name}{ext}"
        s = os.path.join(src_docs, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dest_docs, f))
            print(f"Synced {f} -> {dest_docs}")

# 2. Sync root deliverables
for f in ["FJCAI_2027_LifeMate_CS_ASR_EN.pdf", "FJCAI_2027_LifeMate_CS_ASR_EN.docx", "FJCAI_2027_LifeMate_CS_ASR_EN.doc", "Bai_Bao_Tieng_Viet_FJCAI_2027.docx", "README.md"]:
    s = os.path.join(src_dir, f)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(dest_dir, f))
        print(f"Synced root {f} -> {dest_dir}")

# 3. Sync docs documentation
for f in ["CLAIM_EVIDENCE_MATRIX.md", "EXPERIMENT_REGISTRY.csv", "MULTI_METHOD_PEER_REVIEW_REPORT.md", "PEER_REVIEW_REBUTTAL_GUIDE.md", "Bai_Bao_Tieng_Viet_FJCAI_2027.docx"]:
    s = os.path.join(src_docs, f)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(dest_docs, f))
        print(f"Synced doc {f} -> {dest_docs}")

print("[SUCCESS] All files synchronized to secondary repository D:\\Dataset\\LifeMate_CS_ASR")
