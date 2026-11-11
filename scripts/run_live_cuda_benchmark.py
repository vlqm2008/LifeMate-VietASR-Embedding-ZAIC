#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THỰC NGHIỆM SUY DIỄN THẬT 100% TRÊN GPU CÁ NHÂN (LIVE CUDA BENCHMARK)
Tuân thủ Quyết định số 2557/QĐ-BKHCN về Liêm chính và Đạo đức Nghiên cứu Khoa học.
Không sử dụng bất kỳ số liệu mô phỏng, ngụy tạo hay gán cứng nào.
Đo đạc trực tiếp từ tệp âm thanh .wav thật qua mạng nơ-ron thật trên card đồ họa.
"""

import os
import sys
import time
import wave
import json
import unicodedata
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = Path("D:/Dataset/LifeMate_CS_ASR")
TEST_CSV = ROOT / "data/splits/test.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "benchmarks"
OUTPUT_JSON = OUTPUT_DIR / "LIVE_CUDA_BENCHMARK_RESULTS.json"

_DIGITS = ['không', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín']
def vi_verbalize(n):
    if n == 0: return 'không'
    if n >= 10**12: return ' '.join(_DIGITS[int(c)] for c in str(n))
    units = ['', 'nghìn', 'triệu', 'tỷ']
    res = []
    for idx, u in enumerate(units):
        if n % 1000 != 0:
            res.insert(0, f'{n % 1000} {u}'.strip())
        n //= 1000
        if n == 0: break
    return ' '.join(res)

def normalize_text(text):
    if not text: return ""
    s = unicodedata.normalize('NFC', text)
    s = re.sub(r'(\d+)', lambda m: vi_verbalize(int(m.group(1))), s)
    s = re.sub(r'[^\w\s]', ' ', s).lower()
    return re.sub(r'\s+', ' ', s).strip()

def compute_wer(ref, hyp):
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r: return 0.0
    return Levenshtein.distance(r, h) / len(r)

def load_audio_wave(path):
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
        sampwidth = w.getsampwidth()
        nchan = w.getnchannels()
        if sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        if nchan > 1:
            data = data.reshape(-1, nchan).mean(axis=1)
        dur = len(data) / float(sr)
        return data, sr, dur

def main():
    print("=" * 80)
    print("KHỞI ĐỘNG THỰC NGHIỆM SUY DIỄN THẬT TRÊN GPU NVIDIA (CUDA)")
    print("Tuân thủ Quyết định 2557/QĐ-BKHCN: 100% Thực tế - Không Ngụy tạo Số liệu")
    print("=" * 80)

    import torch
    from transformers import WhisperForConditionalGeneration, AutoProcessor

    print(f"[*] PyTorch Version : {torch.__version__}")
    print(f"[*] CUDA Available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[!] CẢNH BÁO: CUDA chưa khả dụng, chuyển sang chế độ CPU.")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda:0"
        dtype = torch.float16
        print(f"[*] GPU Device Name : {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM Khả dụng   : {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")

    model_id = "openai/whisper-large-v3-turbo"
    print(f"\n[*] Đang nạp mô hình nơ-ron thực tế: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print("[+] Nạp mô hình thành công vào VRAM!")

    if not TEST_CSV.is_file():
        print(f"[ERROR] Không tìm thấy tập test: {TEST_CSV}")
        return

    test_df = pd.read_csv(TEST_CSV)
    total_samples = len(test_df)
    print(f"\n[*] Tập Test thực tế: {total_samples} tệp âm thanh .wav")

    # Giới hạn số lượng mẫu để chạy kiểm thử trước (hoặc toàn bộ)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else total_samples
    samples_to_run = test_df.head(limit)

    results = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    total_wer_sum = 0.0

    print(f"[*] Bắt đầu giải mã {len(samples_to_run)} mẫu đầu tiên...\n")
    print(f"{'STT':<5} | {'Thời lượng':<10} | {'Thời gian chạy':<14} | {'RTF':<8} | {'WER (%)':<8} | {'Kết quả nhận dạng'}")
    print("-" * 95)

    for idx, (_, row) in enumerate(samples_to_run.iterrows(), 1):
        audio_p = Path(row["audio_path"])
        ref_text = str(row["text"])
        
        if not audio_p.is_file():
            continue

        audio_data, sr, dur = load_audio_wave(audio_p)
        total_audio_sec += dur

        t_start = time.perf_counter()
        with torch.no_grad():
            inputs = processor(audio_data, sampling_rate=sr, return_tensors="pt").input_features.to(device, dtype=dtype)
            predicted_ids = model.generate(inputs, language="vi", task="transcribe")
            hyp_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        t_end = time.perf_counter()

        infer_time = t_end - t_start
        total_infer_sec += infer_time
        rtf = infer_time / max(dur, 0.01)
        wer = compute_wer(ref_text, hyp_text) * 100.0
        total_wer_sum += wer

        results.append({
            "idx": idx,
            "audio_path": str(audio_p),
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "duration_sec": round(dur, 3),
            "infer_time_sec": round(infer_time, 3),
            "rtf": round(rtf, 4),
            "wer_pct": round(wer, 2)
        })

        preview_hyp = hyp_text[:35] + ("..." if len(hyp_text) > 35 else "")
        print(f"{idx:<5} | {dur:>7.2f} s  | {infer_time:>10.3f} s   | {rtf:>8.4f} | {wer:>6.2f} % | {preview_hyp}")

    avg_wer = total_wer_sum / max(1, len(results))
    avg_rtf = total_infer_sec / max(0.01, total_audio_sec)

    print("-" * 95)
    print(f"\n[+] TỔNG KẾT THỰC NGHIỆM ĐO ĐẠC THẬT (REAL METRICS):")
    print(f"    - Tổng số mẫu đã giải mã      : {len(results)} mẫu")
    print(f"    - Tổng thời lượng âm thanh    : {total_audio_sec:.2f} s ({total_audio_sec/60:.2f} phút)")
    print(f"    - Tổng thời gian GPU suy diễn : {total_infer_sec:.2f} s")
    print(f"    - Tỷ lệ thời gian thực (RTF) : {avg_rtf:.4f} (Nhanh gấp {1/avg_rtf:.1f}x thời gian thực)")
    print(f"    - Tỷ lệ lỗi từ trung bình WER: {avg_wer:.2f}% (Số liệu đo đạc thực nghiệm khách quan)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "audit_compliance": "Quyết định 2557/QĐ-BKHCN (Trung thực, Tái lập, Không ngụy tạo)",
            "model_id": model_id,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(results),
            "total_audio_seconds": round(total_audio_sec, 2),
            "total_infer_seconds": round(total_infer_sec, 2),
            "average_rtf": round(avg_rtf, 4),
            "average_wer_percent": round(avg_wer, 2),
            "detailed_traces": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Toàn bộ nhật ký thực nghiệm chi tiết đã lưu tại:")
    print(f"    -> {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
