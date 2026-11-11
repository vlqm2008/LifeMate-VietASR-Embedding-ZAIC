import shutil
import os

src_dir = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs"
dest_dir = r"D:\Dataset\LifeMate_CS_ASR\docs"

exts = [".docx", ".doc", ".pdf", ".tex"]

for ext in exts:
    src_en = os.path.join(src_dir, f"FJCAI_2027_LifeMate_CS_ASR_EN{ext}")
    src_std = os.path.join(src_dir, f"FJCAI_2027_LifeMate_CS_ASR{ext}")
    shutil.copy2(src_en, src_std)
    print(f"Copied {src_en} -> {src_std}")
    
    dest_en = os.path.join(dest_dir, f"FJCAI_2027_LifeMate_CS_ASR_EN{ext}")
    dest_std = os.path.join(dest_dir, f"FJCAI_2027_LifeMate_CS_ASR{ext}")
    shutil.copy2(src_en, dest_en)
    shutil.copy2(src_std, dest_std)
    print(f"Synced to {dest_dir}")

# Also copy scripts if any changed
shutil.copy2(r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\build_fjcai_ieee_2col.py",
             r"D:\Dataset\LifeMate_CS_ASR\build_fjcai_ieee_2col.py")
print("Synced build_fjcai_ieee_2col.py")

# Also copy figures
for f in ["fig1_benchmark.png", "fig2_redecoding.png"]:
    shutil.copy2(os.path.join(src_dir, f), os.path.join(dest_dir, f))
print("Synced figures")
