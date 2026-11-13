#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REAL GIPFORMER ASR TRANSCRIPTION OF ALL 634 PUBLIC_TEST SAMPLES
Decodes real audio .wav on CPU using fine-tuned Gipformer 1.5 RNN-T ONNX.
"""

import sys, os, json, time, re, unicodedata
from pathlib import Path
import soundfile as sf
import sherpa_onnx

ROOT = Path("D:/Dataset")
DATA_DIR = ROOT / "LifeMate_CS_ASR/data/train+public_test"
PUB_TEST_DIR = DATA_DIR / "public_test"
PUB_TEST_JSONL = PUB_TEST_DIR / "public_test.jsonl"
OUT_FILE = ROOT / "CHALLENGE_TRACK1_SUBMISSIONS/public_test_real_transcripts.json"

print("="*75)
print("BƯỚC 1: GIẢI MÃ ÂM THANH THẬT 100% BẰNG GIPFORMER ASR TRÊN CPU")
print("="*75)

# Load Recognizer
m = ROOT / "models/lifemate-final/asr-gipformer"
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=str(m / 'encoder.onnx'), decoder=str(m / 'decoder.onnx'), joiner=str(m / 'joiner.onnx'),
    tokens=str(m / 'tokens.txt'), num_threads=8, sample_rate=16000, feature_dim=80,
    decoding_method='greedy_search', provider='cpu'
)

def decode_audio(rel_path):
    p = PUB_TEST_DIR / rel_path
    if not p.exists():
        return ""
    wav, rate = sf.read(str(p), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    s = rec.create_stream()
    s.accept_waveform(rate, wav)
    rec.decode_streams([s])
    text = s.result.text.strip().lower()
    return text

samples = [json.loads(line) for line in open(PUB_TEST_JSONL, encoding="utf-8") if line.strip()]
print(f"[*] Bắt đầu giải mã {len(samples)} mẫu audio thật...")

transcripts = {}
t0 = time.perf_counter()

for i, item in enumerate(samples):
    s_id = item["id"]
    main_audio = item["audio"]
    main_text = decode_audio(main_audio)
    
    # Also decode history audio if any
    hist_decoded = []
    for h in item.get("history", []):
        if h.get("audio"):
            h_text = decode_audio(h["audio"])
            hist_decoded.append({"role": h.get("role", "user"), "audio": h["audio"], "text": h_text})
        else:
            hist_decoded.append(h)
            
    transcripts[s_id] = {
        "audio": main_audio,
        "transcript": main_text,
        "history": hist_decoded
    }
    
    if (i + 1) % 50 == 0 or (i + 1) == len(samples):
        elapsed = time.perf_counter() - t0
        speed = (i + 1) / elapsed
        rem = (len(samples) - (i + 1)) / speed
        print(f"    -> Đã giải mã {i+1}/{len(samples)} mẫu ({speed:.1f} mẫu/s, còn ~{rem:.0f}s)... Mẫu gần nhất: '{main_text[:50]}'")

total_time = time.perf_counter() - t0
print(f"\n[OK] Hoàn tất giải mã 100% {len(transcripts)} file audio trong {total_time:.1f} giây!")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(transcripts, f, ensure_ascii=False, indent=2)

print(f"[*] Đã lưu toàn bộ bản dịch âm thanh thực tế vào: {OUT_FILE}")
