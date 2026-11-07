"""
Module chuẩn hóa văn bản tiếng Việt và câu lệnh chêm tiếng Anh (Code-Switching).
Đạt chuẩn đánh giá WER (Word Error Rate) khắt khe cho bài thi ASR.
"""

import re
import unicodedata
from typing import Optional

# Bảng ánh xạ số cơ bản sang chữ tiếng Việt
NUM_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín", "10": "mười"
}

def normalize_unicode_nfc(text: str) -> str:
    """Chuyển đổi toàn bộ chuỗi sang chuẩn Unicode NFC."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)

def remove_fillers_and_noise(text: str) -> str:
    """Loại bỏ các từ đệm vô nghĩa thường gặp trong audio tự nhiên."""
    fillers = [
        r"\bà\b", r"\bừm\b", r"\bờ\b", r"\bà\b", r"\bơ\b",
        r"\buh\b", r"\bum\b", r"\behm\b"
    ]
    for filler in fillers:
        text = re.sub(filler, " ", text, flags=re.IGNORECASE)
    return text

def normalize_vietnamese_text(text: str, lower: bool = True, strip_punct: bool = True) -> str:
    """
    Quy trình chuẩn hóa đầy đủ cho ASR transcript và dự đoán:
    1. Unicode NFC
    2. Hạ chữ thường (nếu lower=True)
    3. Loại bỏ ký tự đặc biệt, dấu câu thừa
    4. Chuẩn hóa khoảng trắng
    """
    if not text or not isinstance(text, str):
        return ""

    text = normalize_unicode_nfc(text)
    
    if lower:
        text = text.lower()

    # Loại bỏ thẻ XML, HTML hoặc annotation trong dataset (vd: <noise>, [laughter])
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\([^\)]+\)", " ", text)

    # Loại bỏ từ đệm
    text = remove_fillers_and_noise(text)

    # Chuyển các dấu gạch ngang, gạch dưới thành space
    text = re.sub(r"[-_—–/]", " ", text)

    if strip_punct:
        # Giữ lại các chữ cái tiếng Việt, tiếng Anh và số
        text = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", " ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_for_wer(text: str) -> str:
    """Hàm chuẩn hóa tối đa phục vụ đánh giá JiWER (bỏ dấu chấm phẩy, lower hoàn toàn)."""
    return normalize_vietnamese_text(text, lower=True, strip_punct=True)

if __name__ == "__main__":
    sample = "Mai schedule meeting với team lúc 8:00 nha <music>... Ờ, nhớ check email!"
    print("Gốc:", sample)
    print("Sau chuẩn hóa:", prepare_for_wer(sample))
