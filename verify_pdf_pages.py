import fitz # PyMuPDF
doc = fitz.open(r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\docs\FJCAI_2027_LifeMate_CS_ASR_EN.pdf")
print("PDF page count:", len(doc))
for i in range(len(doc)):
    page = doc[i]
    print(f"Page {i+1} rect:", page.rect)
