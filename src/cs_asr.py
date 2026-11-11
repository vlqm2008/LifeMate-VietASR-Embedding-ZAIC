#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LifeMate CS-ASR ULTRA | Whisper large-v3-turbo (LoRA) + BGE-M3 + XGBoost Dynamic Threshold
==========================================================================================
Vietnamese / English / Vietnamese-English code-switching ASR, total budget <= 1.5B params
(ASR ~809M + BGE-M3 ~568M + XGBoost trees).

Pipeline (each step is a sub-command, every step writes files so Colab can resume):
  collect-lifemate  : ZAIC 2026 LifeMate TRAIN audio (main turn + history user turns) -> manifest
  collect-hf        : external HF corpora (SEA-CS en-vi, ViMedCSS, GigaSpeech2-vi, ...) -> manifest
  prepare           : group-aware split train / val / judge / test + Table 1
  train-asr         : LoRA fine-tune Whisper (source-balanced sampling, SpecAugment)
  merge             : merge LoRA -> HF folder + asr_merged_fp16.pth
  nbest             : decode "candidate ladder" (round0 beam N-best, round1 positional penalty,
                      round2 stronger penalty + temperature sampling) with token log-probs
  build-emb-data    : mMARCO-vi / WikiMatrix / PhoMT / ViQuAD2 + ASR hard negatives
  train-emb         : fine-tune BGE-M3 (Cached MNRL) + discrimination eval
  features          : log-loss / confidence / repetition / semantic (BGE-M3) features per candidate
  train-judge       : XGBoost error classifier (binary log-loss) + XGBoost quantile model
                      T_dynamic(x); vectorised threshold search on out-of-fold predictions
  evaluate          : Table 2 / Table 3 (WER, CER, MER, E-WER, V-WER, hallucination, RTF)
  transcribe        : online loop: decode -> judge -> re-decode (penalise suspicious tokens)
  budget / export   : parameter audit (<1.5B) and .pth export

No metric is hard-coded: every number in the tables is computed from held-out data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import unicodedata
import warnings
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MAX_TOTAL_PARAMS = 1_500_000_000
DEFAULT_ASR = "openai/whisper-large-v3-turbo"
DEFAULT_EMB = "BAAI/bge-m3"
FIELDS = ["audio_path", "text", "language", "source", "split_hint", "record_id", "group_id", "duration"]
TEXT_CANDIDATES = ["text", "raw_transcription", "transcription", "transcript", "sentence", "segment_text",
                   "normalized_text", "sentence_norm", "content", "script", "label"]
GROUP_CANDIDATES = ["youtube_id", "video_id", "speaker_id", "speaker", "client_id", "channel_id", "source_id",
                    "audio_id", "file_id"]


# =============================================================================
# Generic utils
# =============================================================================
def log(*a):
    print(*a, flush=True)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.is_file():
        return out
    with p.open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def append_jsonl(path, obj):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s or "")).replace("\ufeff", "")).strip()


def stable_hash(s, mod=None):
    h = int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:12], 16)
    return h % mod if mod else h


def parse_kv(s, cast=float):
    out = {}
    for part in (s or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = cast(v.strip())
    return out


# =============================================================================
# Vietnamese text normalisation (applied symmetrically to REF and HYP for scoring)
# =============================================================================
_DIG = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _read3(n, full):
    tr, ch, dv = n // 100, (n // 10) % 10, n % 10
    out = []
    if full or tr:
        out += [_DIG[tr], "trăm"]
    if ch == 0:
        if dv and (full or tr):
            out.append("linh")
    elif ch == 1:
        out.append("mười")
    else:
        out += [_DIG[ch], "mươi"]
    if dv:
        if dv == 1 and ch >= 2:
            out.append("mốt")
        elif dv == 5 and ch >= 1:
            out.append("lăm")
        else:
            out.append(_DIG[dv])
    return out


def vi_number(n: int) -> str:
    if n == 0:
        return "không"
    if n >= 10 ** 12:
        return " ".join(_DIG[int(c)] for c in str(n))
    units = ["", "nghìn", "triệu", "tỷ"]
    groups = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    out = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        out += _read3(g, full=i < len(groups) - 1)
        if units[i]:
            out.append(units[i])
    return " ".join(out)


def verbalize_numbers(s):
    s = re.sub(r"(?<=\d)[.,](?=\d{3}(?:\D|$))", "", s)          # 1.200.000 -> 1200000
    s = re.sub(r"(\d{1,2}):(\d{2})", lambda m: f"{int(m.group(1))} giờ {int(m.group(2))}", s)
    s = re.sub(r"(\d),(\d)", r"\1 phẩy \2", s)
    s = s.replace("%", " phần trăm ")
    return re.sub(r"\d+", lambda m: " " + vi_number(int(m.group())) + " " if len(m.group()) <= 12
                  else " " + " ".join(_DIG[int(c)] for c in m.group()) + " ", s)


def metric_text(s):
    s = verbalize_numbers(norm(s)).casefold()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE).replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


# =============================================================================
# Code-switching: token language identification (VI vs EN)
# =============================================================================
VI_DIACRITIC = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
_ONSET = "(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|b|c|d|g|h|k|l|m|n|p|r|s|t|v|x)?"
_NUCLEI = ["oai", "oay", "oao", "oeo", "uai", "uay", "uoi", "uou", "ieu", "yeu", "uya", "uye", "uyu",
           "ai", "ao", "au", "ay", "eo", "eu", "ia", "iu", "oa", "oe", "oi", "ua", "ui", "uy", "ue", "uo",
           "ie", "ye", "oo", "uu", "a", "e", "i", "o", "u", "y"]
_VI_SYL = re.compile("^" + _ONSET + "(?:" + "|".join(_NUCLEI) + ")(?:ch|ng|nh|c|m|n|p|t)?$")
try:
    from wordfreq import zipf_frequency as _zipf
except Exception:  # pragma: no cover
    _zipf = None


def is_english_token(tok):
    t = tok.casefold()
    if not t or not t.isalpha() or VI_DIACRITIC.search(t) or not t.isascii():
        return False
    if not _VI_SYL.match(t):
        return True                     # violates Vietnamese phonotactics -> foreign word
    if _zipf is not None:
        en, vi = _zipf(t, "en"), _zipf(t, "vi")
        return en >= 3.5 and en - vi >= 1.5
    return False


def token_langs(tokens):
    return ["en" if is_english_token(t) else "vi" for t in tokens]


def utt_language(text):
    toks = metric_text(text).split()
    if not toks:
        return "vi"
    langs = token_langs(toks)
    en = sum(l == "en" for l in langs) / len(langs)
    if en >= 0.8:
        return "en"
    return "mixed" if en > 0 else "vi"


def switch_stats(tokens):
    langs = token_langs(tokens)
    if not langs:
        return 0.0, 0
    return sum(l == "en" for l in langs) / len(langs), sum(a != b for a, b in zip(langs, langs[1:]))


# =============================================================================
# Metrics: WER, CER, MER, E-WER (English tokens), V-WER (Vietnamese tokens), hallucination
# =============================================================================
def _editops_py(r, h):
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            if r[i - 1] != h[j - 1]:
                ops.append(("replace", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("delete", i - 1, j))
            i -= 1
        else:
            ops.append(("insert", i, j - 1))
            j -= 1
    return ops[::-1]


def editops(r, h):
    try:
        from rapidfuzz.distance import Levenshtein
        return [(o.tag, o.src_pos, o.dest_pos) for o in Levenshtein.editops(r, h)]
    except Exception:
        return _editops_py(r, h)


def edit_distance(a, b):
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except Exception:
        return len(_editops_py(list(a), list(b)))


def max_ngram_repeat(words, n=3):
    if len(words) < n:
        return 0
    return max(Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1)).values())


def utt_errors(ref_text, hyp_text):
    r, h = metric_text(ref_text).split(), metric_text(hyp_text).split()
    ops = editops(r, h)
    rl, hl = token_langs(r), token_langs(h)
    c = Counter()
    for tag, i, j in ops:
        c[tag] += 1
        lang = hl[j] if tag == "insert" else rl[i]
        c["err_" + lang] += 1
    rs, hs = "".join(r), "".join(h)
    out = {"errors": len(ops), "n": len(r), "S": c["replace"], "D": c["delete"], "I": c["insert"],
           "en_err": c["err_en"], "en_n": sum(l == "en" for l in rl),
           "vi_err": c["err_vi"], "vi_n": sum(l == "vi" for l in rl),
           "char_err": edit_distance(rs, hs), "chars": len(rs)}
    out["halluc"] = int((len(r) == 0 and len(h) > 0) or c["insert"] >= max(3, 0.5 * len(r))
                        or max_ngram_repeat(h) >= 3 > max_ngram_repeat(r))
    return out


def corpus_metrics(refs, hyps):
    tot = Counter()
    for r, h in zip(refs, hyps):
        e = utt_errors(r, h)
        for k, v in e.items():
            tot[k] += v
        tot["utts"] += 1
        tot["sent_err"] += int(e["errors"] > 0)
    n = max(1, tot["n"])
    return {"WER": tot["errors"] / n,
            # Vietnamese orthographic words are syllables, English tokens are words: the mixed-unit
            # error rate (MER, SEAME convention) therefore coincides with WER on this tokenisation.
            "MER": tot["errors"] / n,
            "CER": tot["char_err"] / max(1, tot["chars"]),
            "E_WER": tot["en_err"] / max(1, tot["en_n"]), "V_WER": tot["vi_err"] / max(1, tot["vi_n"]),
            "SER": tot["sent_err"] / max(1, tot["utts"]), "HALLUC": tot["halluc"] / max(1, tot["utts"]),
            "S": tot["S"], "D": tot["D"], "I": tot["I"], "words": tot["n"], "en_words": tot["en_n"],
            "utts": tot["utts"]}


# =============================================================================
# Audio
# =============================================================================
def to_16k_mono(wave, sr):
    from scipy.signal import resample_poly
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = math.gcd(int(sr), SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return wave


def get_audio_duration(path):
    try:
        import soundfile as sf
        return sf.info(str(path)).duration
    except Exception:
        import wave
        with wave.open(str(path), 'rb') as w:
            return w.getnframes() / float(w.getframerate())


def load_audio(path):
    try:
        import soundfile as sf
        wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return to_16k_mono(wave, sr)
    except Exception:
        try:
            import wave
            with wave.open(str(path), 'rb') as w:
                frames = w.readframes(w.getnframes())
                sr = w.getframerate()
                sampwidth = w.getsampwidth()
                nchan = w.getnchannels()
                if sampwidth == 2:
                    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                if nchan > 1:
                    data = data.reshape(-1, nchan)
                return to_16k_mono(data, sr)
        except Exception:
            import subprocess
            p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar",
                                "16000", "-f", "f32le", "pipe:1"], stdout=subprocess.PIPE, check=True)
            return np.frombuffer(p.stdout, dtype="<f4").copy()


def audio_stats(wave):
    dur = len(wave) / SAMPLE_RATE
    if len(wave) < 400:
        return dur, -90.0, 0.0
    hop, win = 160, 400
    n = 1 + (len(wave) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    e = 10 * np.log10(np.mean(wave[idx] ** 2, axis=1) + 1e-10)
    rms_db = float(10 * np.log10(np.mean(wave ** 2) + 1e-10))
    return dur, rms_db, float(np.percentile(e, 90) - np.percentile(e, 10))


# =============================================================================
# Manifest IO
# =============================================================================
def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["duration"] = float(r.get("duration") or 0)
    return rows


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def keep_utterance(text, dur):
    if not text or not (0.6 <= dur <= 29.5):
        return False
    cps = len(text) / dur
    if not (2.0 <= cps <= 32.0):
        return False
    return not re.search(r"[\u0E00-\u0E7F\u3040-\u30ff\u4e00-\u9fff\u0900-\u097F]", text)


# =============================================================================
# collect-lifemate
# =============================================================================
def cmd_collect_lifemate(a):
    train_dir = Path(a.train_dir)
    jsonl = train_dir / "train.jsonl"
    assert jsonl.is_file(), f"Missing {jsonl}"
    clean = {}
    if a.clearvoice_manifest and Path(a.clearvoice_manifest).is_file():
        for r in read_manifest(a.clearvoice_manifest):
            rid = r.get("record_id") or Path(r["audio_path"]).stem
            if Path(r["audio_path"]).is_file():
                clean[rid] = r["audio_path"]
        log("ClearVoice cache reused for", len(clean), "LifeMate recordings")
    rows, miss = {}, Counter()
    for rec in read_jsonl(jsonl):
        sid = rec["id"]
        items = [(sid, rec.get("audio"), rec.get("audio_text"))]
        for h in rec.get("history") or []:
            if h.get("role") == "user" and h.get("audio"):
                items.append((Path(h["audio"]).stem, h.get("audio"), h.get("audio_text")))
        for rid, ap, txt in items:
            txt = norm(txt)
            if not txt or not ap:
                miss["no_text"] += 1
                continue
            p = train_dir / ap
            if not p.is_file():
                miss["no_audio"] += 1
                continue
            p = Path(clean.get(rid, p))
            rows[rid] = {"audio_path": str(p.resolve()), "text": txt, "language": utt_language(txt),
                         "source": "lifemate", "split_hint": "", "record_id": rid, "group_id": sid,
                         "duration": round(get_audio_duration(p), 3)}
    rows = list(rows.values())
    write_manifest(a.out, rows)
    log(f"LifeMate recordings: {len(rows)} | {sum(r['duration'] for r in rows)/3600:.2f} h | skipped {dict(miss)}")
    log("Language mix:", Counter(r["language"] for r in rows))


# =============================================================================
# collect-hf : external corpora (each source independently; failures are skipped, never faked)
# =============================================================================
SOURCES = {
    "sea_cs_envi": dict(repo="gijs/sea_speech", config="en-vi", kind="hf", cs=True, gated=True),
    "vimedcss": dict(repo="tensorxt/ViMedCSS", config=None, kind="hf", cs=True),
    "gigaspeech2_vi": dict(repo="speechcolab/gigaspeech2", kind="giga2", gated=True),
    "vimd": dict(repo="nguyendv02/ViMD_Dataset", config=None, kind="hf"),
    "vietsuperspeech": dict(repo="thanhnew2001/VietSuperSpeech", config=None, kind="hf"),
    "bud500": dict(repo="linhtran92/viet_bud500", config=None, kind="hf", gated=True),
    "phoaudiobook": dict(repo="thivux/phoaudiobook", config=None, kind="hf", gated=True),
    "fleurs_vi": dict(repo="google/fleurs", config="vi_vn", kind="hf", trust=True),
    "fleurs_en": dict(repo="google/fleurs", config="en_us", kind="hf", trust=True),
    "librispeech_en": dict(repo="openslr/librispeech_asr", config="clean", kind="hf",
                           train_splits=["train.100"], test_splits=["test"]),
}


def classify_split(name):
    n = name.lower()
    if "train" in n:
        return "train"
    if "test" in n or "hard" in n:
        return "test"
    if "valid" in n or "dev" in n:
        return "dev"
    return "train"


def _pick_col(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def iter_hf(spec, split, token, seed):
    from datasets import Audio, load_dataset
    ds = load_dataset(spec["repo"], spec.get("config"), split=split, streaming=True, token=token or None,
                      trust_remote_code=bool(spec.get("trust")))
    first = next(iter(ds))
    cols = list(first.keys())
    audio_col = None
    feats = getattr(ds, "features", None) or {}
    for c in cols:
        if isinstance(feats.get(c), Audio) or (isinstance(first[c], dict) and
                                               ({"array", "bytes", "path"} & set(first[c].keys()))):
            audio_col = c
            break
    text_col = _pick_col(cols, TEXT_CANDIDATES)
    group_col = _pick_col(cols, GROUP_CANDIDATES)
    if audio_col is None or text_col is None:
        raise RuntimeError(f"Cannot detect audio/text columns in {spec['repo']}: {cols}")
    try:
        ds = ds.cast_column(audio_col, Audio(sampling_rate=SAMPLE_RATE))
    except Exception as e:
        warnings.warn(f"cast_column failed ({e}); resampling manually")
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    log(f"  columns: audio={audio_col} text={text_col} group={group_col}")
    for i, ex in enumerate(ds):
        au = ex[audio_col]
        if au is None:
            continue
        if isinstance(au, dict) and au.get("array") is not None:
            wave = to_16k_mono(au["array"], au.get("sampling_rate", SAMPLE_RATE))
        elif isinstance(au, dict) and au.get("bytes"):
            import soundfile as sf
            w, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32", always_2d=True)
            wave = to_16k_mono(w, sr)
        else:
            continue
        txt = ex.get(text_col)
        if isinstance(txt, (list, tuple)):
            txt = " ".join(map(str, txt))
        uid = str(ex.get("id") or ex.get("key") or ex.get("audio_id") or f"{split}_{i}")
        grp = str(ex.get(group_col)) if group_col else uid
        yield wave, norm(txt), uid, grp


def iter_giga2(spec, token, seed, max_shards, which="train"):
    from huggingface_hub import HfApi, hf_hub_download
    import soundfile as sf
    repo = spec["repo"]
    files = HfApi().list_repo_files(repo, repo_type="dataset", token=token or None)
    vi = [f for f in files if f.startswith("vi/")]
    if which == "train":
        tsv = next((f for f in vi if f.endswith(".tsv") and "train_refined" in f), None) or \
            next((f for f in vi if f.endswith(".tsv") and "train" in f), None)
        shards = sorted([f for f in vi if "/train/" in f and f.endswith(".tar.gz")],
                        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])
        random.Random(seed).shuffle(shards)
        shards = shards[:max_shards]
    else:
        tsv = next((f for f in vi if f.endswith(f"{which}.tsv")), None)
        shards = [f for f in vi if f.endswith(f"{which}.tar.gz")]
    if not tsv or not shards:
        raise RuntimeError(f"GigaSpeech2 vi files not found ({which}); check access on the HF page")
    tmp = Path("/tmp/giga2_dl")
    tmp.mkdir(parents=True, exist_ok=True)
    tpath = hf_hub_download(repo, tsv, repo_type="dataset", token=token or None, local_dir=str(tmp))
    text = {}
    with open(tpath, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                text[parts[0]] = parts[-1]
    log(f"  GigaSpeech2 {which}: {len(text)} transcripts, {len(shards)} shard(s)")
    for sh in shards:
        lp = hf_hub_download(repo, sh, repo_type="dataset", token=token or None, local_dir=str(tmp))
        with tarfile.open(lp, "r:gz") as tf:
            for m in tf:
                if not m.isfile() or not m.name.lower().endswith((".wav", ".flac", ".mp3", ".opus", ".ogg")):
                    continue
                uid = Path(m.name).stem
                if uid not in text:
                    continue
                try:
                    w, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32", always_2d=True)
                except Exception:
                    continue
                grp = uid.rsplit("-", 1)[0] if "-" in uid else uid.split("_")[0]
                yield to_16k_mono(w, sr), norm(text[uid]), uid, grp
        try:
            os.remove(lp)
        except OSError:
            pass


def cmd_collect_hf(a):
    import soundfile as sf
    out_dir = Path(a.out_dir)
    hours = parse_kv(a.hours)
    test_hours = float(a.test_hours)
    token = a.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    report = {}
    for name in [s.strip() for s in a.sources.split(",") if s.strip()]:
        spec = SOURCES.get(name)
        if spec is None:
            log("Unknown source", name)
            continue
        man = out_dir / f"manifest_{name}.csv"
        if man.is_file() and not a.overwrite:
            rows = read_manifest(man)
            report[name] = {"status": "cached", "rows": len(rows), "hours": sum(r["duration"] for r in rows) / 3600}
            log(f"[{name}] cached: {len(rows)} rows")
            continue
        log(f"[{name}] {spec['repo']} {spec.get('config') or ''} (gated={spec.get('gated', False)})")
        rows, stats = [], Counter()
        try:
            plan = []
            if spec["kind"] == "giga2":
                plan = [("train", "train"), ("test", "test")]
            else:
                if spec.get("train_splits"):
                    plan = [(s, "train") for s in spec["train_splits"]] + [(s, "test") for s in spec.get("test_splits", [])]
                else:
                    from datasets import get_dataset_split_names
                    names = get_dataset_split_names(spec["repo"], spec.get("config"), token=token or None,
                                                    trust_remote_code=bool(spec.get("trust")))
                    plan = [(s, classify_split(s)) for s in names]
            for split, hint in plan:
                cap = (hours.get(name, 10.0) if hint == "train" else test_hours) * 3600
                if cap <= 0:
                    continue
                got = 0.0
                it = (iter_giga2(spec, token, a.seed, int(a.giga2_shards), which=split) if spec["kind"] == "giga2"
                      else iter_hf(spec, split, token, a.seed))
                for wave, txt, uid, grp in it:
                    dur = len(wave) / SAMPLE_RATE
                    if not keep_utterance(txt, dur):
                        stats["filtered"] += 1
                        continue
                    hid = hashlib.md5(f"{name}|{split}|{uid}".encode()).hexdigest()
                    path = out_dir / "audio" / name / hid[:2] / f"{hid}.flac"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    peak = float(np.max(np.abs(wave))) if len(wave) else 0
                    if peak > 0.999:
                        wave = wave * (0.98 / peak)
                    sf.write(str(path), wave, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
                    rows.append({"audio_path": str(path), "text": txt, "language": utt_language(txt),
                                 "source": name, "split_hint": hint, "record_id": f"{name}_{hid[:16]}",
                                 "group_id": f"{name}_{grp}", "duration": round(dur, 3)})
                    got += dur
                    stats[hint] += 1
                    if got >= cap:
                        break
                log(f"  split={split} ({hint}): {got/3600:.2f} h")
            write_manifest(man, rows)
            report[name] = {"status": "ok", "rows": len(rows), "hours": sum(r["duration"] for r in rows) / 3600,
                            "stats": dict(stats)}
        except Exception as e:  # gated / renamed / network: skip honestly
            log(f"  !! {name} skipped: {type(e).__name__}: {str(e)[:300]}")
            if rows:
                write_manifest(man, rows)
            report[name] = {"status": f"failed: {type(e).__name__}", "rows": len(rows),
                            "hours": sum(r["duration"] for r in rows) / 3600}
    save_json(out_dir / "collect_report.json", report)
    log(json.dumps(report, ensure_ascii=False, indent=1))


# =============================================================================
# prepare : group-aware splits + Table 1
# =============================================================================
def cmd_prepare(a):
    rows = []
    for m in a.manifests:
        if Path(m).is_file():
            rows += read_manifest(m)
    assert rows, "No manifests"
    test_ids = set(read_json(a.lifemate_test_ids)) if a.lifemate_test_ids and Path(a.lifemate_test_ids).is_file() else set()
    cs_sources = {k for k, v in SOURCES.items() if v.get("cs")}
    for r in rows:
        g, src = r["group_id"], r["source"]
        h = stable_hash("split|" + g, 10000) / 10000
        if src == "lifemate":
            if test_ids and g in test_ids:
                sp = "test"
            elif not test_ids and h < 0.13:
                sp = "test"
            elif h < 0.13 + a.val_frac if not test_ids else h < a.val_frac:
                sp = "val"
            elif stable_hash("judge|" + g, 10000) / 10000 < a.judge_frac:
                sp = "judge"
            else:
                sp = "train"
        elif r["split_hint"] == "test":
            sp = "test"
        elif r["split_hint"] == "dev":
            sp = "judge"
        else:
            t = a.cs_test_frac if src in cs_sources and not any(
                x["source"] == src and x["split_hint"] == "test" for x in rows[:0]) else 0.0
            if src in cs_sources and h < t:
                sp = "test"
            elif src in cs_sources and h < t + a.cs_val_frac:
                sp = "val"
            elif h < t + a.cs_val_frac + a.judge_frac:
                sp = "judge"
            else:
                sp = "train"
        r["split"] = sp
    # sources with an official test split do not need a carved one: move carved CS test back to train
    official = {r["source"] for r in rows if r["split_hint"] == "test"}
    for r in rows:
        if r["split"] == "test" and r["source"] in official and r["split_hint"] != "test" and r["source"] != "lifemate":
            r["split"] = "train"
    # leakage check
    seen = defaultdict(set)
    for r in rows:
        seen[r["group_id"]].add(r["split"])
    leak = [g for g, s in seen.items() if len(s) > 1]
    assert not leak, f"Group leakage across splits: {leak[:5]}"
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for sp in ["train", "val", "judge", "test"]:
        part = [r for r in rows if r["split"] == sp]
        cap = {"val": a.max_val, "judge": a.max_judge, "test": a.max_test}.get(sp, 0)
        if cap and len(part) > cap:   # stratified by source, deterministic
            by = defaultdict(list)
            for r in part:
                by[r["source"]].append(r)
            keep = []
            for s, lst in by.items():
                rng.shuffle(lst)
                keep += lst[:max(1, int(round(cap * len(lst) / len(part))))]
            part = keep
        write_manifest(out / f"{sp}.csv", part)
        for r in part:
            stats[r["source"]][sp][0] += 1
            stats[r["source"]][sp][1] += r["duration"] / 3600
    tab = ["| Nguồn | Train (giờ / câu) | Val | Judge (XGBoost) | Test | % câu chêm EN (train) |",
           "|---|---|---|---|---|---|"]
    train_rows = read_manifest(out / "train.csv")
    for s in sorted(stats):
        mix = [r for r in train_rows if r["source"] == s]
        cs = 100 * sum(r["language"] == "mixed" for r in mix) / max(1, len(mix))
        cells = [f"{stats[s][k][1]:.2f} / {stats[s][k][0]:,}" for k in ["train", "val", "judge", "test"]]
        tab.append(f"| {s} | " + " | ".join(cells) + f" | {cs:.1f} |")
    tot = [sum(stats[s][k][1] for s in stats) for k in ["train", "val", "judge", "test"]]
    tab.append("| **Tổng** | " + " | ".join(f"**{x:.2f} h**" for x in tot) + " | |")
    (out / "table1_dataset.md").write_text("\n".join(tab), encoding="utf-8")
    save_json(out / "signature.json", {"stats": {s: dict(v) for s, v in stats.items()}, "seed": a.seed})
    log("\n".join(tab))


# =============================================================================
# ASR model helpers
# =============================================================================
def select_dtype(name="auto"):
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_whisper(path, dtype):
    from transformers import WhisperForConditionalGeneration
    m = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True,
                                                        attn_implementation="sdpa")
    m.generation_config.forced_decoder_ids = None
    m.config.forced_decoder_ids = None
    m.generation_config.task = "transcribe"
    return m


def whisper_lang(language):
    return "english" if language == "en" else "vietnamese"


def augment_wave(wave, strength):
    from scipy.signal import resample_poly
    if strength <= 0:
        return wave
    wave = wave.copy()
    if random.random() < 0.25 * strength:
        up, down = random.choice([(20, 19), (20, 21)])
        if len(wave) * up / down < 29.5 * SAMPLE_RATE:
            wave = resample_poly(wave, up, down).astype(np.float32)
    if random.random() < 0.5:
        wave *= 10.0 ** (random.uniform(-4, 4) / 20.0)
    if random.random() < 0.2 * strength:
        rms = float(np.sqrt(np.mean(wave * wave) + 1e-9))
        noise = np.random.standard_normal(len(wave)).astype(np.float32)
        wave += noise * rms / 10 ** (random.uniform(15, 35) / 20)
    peak = float(np.max(np.abs(wave)) + 1e-9)
    return (wave * (0.99 / peak) if peak > 0.99 else wave).astype(np.float32)


def spec_augment(feat, frames, strength):
    if strength <= 0:
        return
    nm, nf = feat.shape
    usable = max(1, min(nf, frames))
    for _ in range(2):
        w = random.randint(0, min(int(25 * strength), usable))
        if w:
            s = random.randint(0, usable - w)
            feat[:, s:s + w] = 0
    fw = random.randint(0, min(int(8 * strength), nm))
    if fw:
        s = random.randint(0, nm - fw)
        feat[s:s + fw, :] = 0


class ASRDataset:
    def __init__(self, rows, processor, train, strength=0.0, max_label_tokens=220):
        tok = processor.tokenizer
        self.fe = processor.feature_extractor
        self.train, self.strength = train, strength
        sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.rows, self.labels = [], []
        for r in rows:
            tok.set_prefix_tokens(language=whisper_lang(r["language"]), task="transcribe", predict_timestamps=False)
            ids = tok(r["text"]).input_ids
            if ids and ids[0] == sot:
                ids = ids[1:]
            if 4 < len(ids) <= max_label_tokens:
                self.rows.append(r)
                self.labels.append(np.asarray(ids, dtype=np.int64))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        wave = load_audio(self.rows[i]["audio_path"])
        if self.train:
            wave = augment_wave(wave, self.strength)
        feat = self.fe(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features[0]
        if self.train:
            spec_augment(feat, int(math.ceil(len(wave) / 160)), self.strength)
        return feat, torch.from_numpy(self.labels[i]), i


def collate(items):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    return {"input_features": torch.stack([x[0] for x in items]),
            "labels": pad_sequence([x[1] for x in items], batch_first=True, padding_value=-100),
            "indices": [x[2] for x in items]}


def quick_eval(model, processor, rows, bs=16, max_new_tokens=192):
    """Greedy decoding, used for model selection during training."""
    import torch
    model.eval()
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    hyps = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["language"] == "en", rows[i]["duration"]))
    with torch.inference_mode():
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            langs = {whisper_lang(rows[i]["language"]) for i in idx}
            groups = [[i for i in idx if whisper_lang(rows[i]["language"]) == l] for l in langs]
            for grp in groups:
                waves = [load_audio(rows[i]["audio_path"]) for i in grp]
                feats = processor.feature_extractor(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
                out = model.generate(input_features=feats.to(dev, dt), language=whisper_lang(rows[grp[0]]["language"]),
                                     task="transcribe", num_beams=1, do_sample=False, max_new_tokens=max_new_tokens)
                for i, t in zip(grp, processor.batch_decode(out, skip_special_tokens=True)):
                    hyps[i] = norm(t)
    m = corpus_metrics([r["text"] for r in rows], hyps)
    return m, hyps


# =============================================================================
# train-asr
# =============================================================================
def cmd_train_asr(a):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup, set_seed
    set_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    assert torch.cuda.is_available(), "GPU required"
    dtype = select_dtype(a.dtype)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    tr = read_manifest(data / "train.csv")
    va = read_manifest(data / "val.csv")
    if a.only_sources:
        keep = {s.strip() for s in a.only_sources.split(",")}
        tr = [r for r in tr if r["source"] in keep]
    if a.eval_max and len(va) > a.eval_max:
        va = random.Random(a.seed).sample(va, a.eval_max)
    processor = AutoProcessor.from_pretrained(a.model)
    base = load_whisper(a.model, dtype)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    resume = a.resume and (out / "last_adapter" / "adapter_config.json").is_file() and (out / "last_state.pt").is_file()
    if resume:
        model = PeftModel.from_pretrained(base, str(out / "last_adapter"), is_trainable=True)
    elif a.init_adapter:
        model = PeftModel.from_pretrained(base, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=a.dropout,
                                                target_modules=[t.strip() for t in a.targets.split(",")],
                                                bias="none"))
    model.to("cuda")
    total = sum(p.numel() for p in model.parameters())
    log(f"ASR params (base+LoRA): {total:,}")
    assert total <= a.max_params, "ASR exceeds its parameter budget"
    model.print_trainable_parameters()
    tds = ASRDataset(tr, processor, train=True, strength=a.augment)
    log(f"train rows {len(tds)} | val rows {len(va)}")
    boosts = parse_kv(a.boost)
    cnt = Counter(r["source"] for r in tds.rows)
    mass = {s: (n * boosts.get(s, 1.0)) ** a.alpha for s, n in cnt.items()}
    z = sum(mass.values())
    log("Sampling share per source:", {s: round(m / z, 3) for s, m in mass.items()})
    weights = torch.as_tensor([mass[r["source"]] / z / cnt[r["source"]] for r in tds.rows], dtype=torch.double)
    per_epoch = a.samples_per_epoch or len(tds)
    steps_epoch = math.ceil(per_epoch / a.batch / a.accum)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr, weight_decay=0.01,
                            betas=(0.9, 0.98))
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(a.warmup * steps_epoch * a.epochs)), steps_epoch * a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    start, best, bad = 0, float("inf"), 0
    if resume:
        st = torch.load(out / "last_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"]); scaler.load_state_dict(st["scaler"])
        start, best, bad = st["epoch"], st["best"], st["bad"]
        log("Resume from epoch", start + 1, "best WER", best)
    else:
        save_json(out / "training_config.json", vars(a))
        if a.baseline:
            m0, _ = quick_eval(model, processor, va)
            save_json(out / "baseline_metrics.json", m0)
            log("Baseline (pre-finetune) val WER:", round(m0["WER"] * 100, 2))
    for ep in range(start, a.epochs):
        g = torch.Generator().manual_seed(a.seed + ep)
        loader = DataLoader(tds, batch_size=a.batch, sampler=WeightedRandomSampler(weights, per_epoch, True, generator=g),
                            num_workers=a.workers, collate_fn=collate, pin_memory=True,
                            persistent_workers=a.workers > 0)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, run_loss = time.time(), 0.0
        for step, b in enumerate(tqdm(loader, desc=f"epoch {ep+1}/{a.epochs}")):
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                loss = model(input_features=b["input_features"].to("cuda", dtype),
                             labels=b["labels"].to("cuda")).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            scaler.scale(loss / a.accum).backward()
            run_loss += float(loss.detach())
            if (step + 1) % a.accum == 0 or step + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
            if a.log_every and (step + 1) % a.log_every == 0:
                log(f"  step {step+1} loss {run_loss/(step+1):.4f} lr {sch.get_last_lr()[0]:.2e}")
        m, hyps = quick_eval(model, processor, va)
        improved = m["WER"] < best - 1e-5
        if improved:
            best, bad = m["WER"], 0
            model.save_pretrained(out / "best_adapter")
            processor.save_pretrained(out / "best_adapter")
            save_json(out / "best_metrics.json", m)
            with (out / "best_predictions.jsonl").open("w", encoding="utf-8") as f:
                for r, h in zip(va, hyps):
                    e = utt_errors(r["text"], h)
                    f.write(json.dumps({"record_id": r["record_id"], "source": r["source"], "audio_path": r["audio_path"],
                                        "reference": r["text"], "prediction": h,
                                        "WER": e["errors"] / max(1, e["n"])}, ensure_ascii=False) + "\n")
        else:
            bad += 1
        model.save_pretrained(out / "last_adapter")
        torch.save({"opt": opt.state_dict(), "sch": sch.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "best": best, "bad": bad}, out / "last_state.pt")
        summ = {"epoch": ep + 1, "train_loss": round(run_loss / max(1, len(loader)), 4), "val": m,
                "best_wer": best, "improved": improved, "minutes": round((time.time() - t0) / 60, 1)}
        append_jsonl(out / "history.jsonl", summ)
        log(f"EPOCH {ep+1}: loss {summ['train_loss']} | val WER {m['WER']*100:.2f}% E-WER {m['E_WER']*100:.2f}% "
            f"| best {best*100:.2f}%")
        if bad >= a.patience:
            log("Early stop")
            break
    (out / "stage_complete.json").write_text(json.dumps({"best_wer": best}), encoding="utf-8")


def cmd_merge(a):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor
    base = load_whisper(a.model, torch.float16)
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(a.adapter if (Path(a.adapter) / "preprocessor_config.json").is_file() else a.model).save_pretrained(out)
    torch.save({k: v.half().contiguous() for k, v in model.state_dict().items()}, out / "asr_merged_fp16.pth")
    log("Merged ASR ->", out, "| params", f"{sum(p.numel() for p in model.parameters()):,}")


# =============================================================================
# Candidate ladder decoding (round0 N-best, round1/2 re-decode with positional penalties)
# =============================================================================
def make_penalty_processor(penalties):
    import torch
    from transformers import LogitsProcessor

    class PositionalPenalty(LogitsProcessor):
        """Subtract delta from token logits at the decoding step where the rejected hypothesis was
        unconfident -> the decoder is pushed away from the suspected error, elsewhere untouched."""

        def __init__(self, pen):
            self.pen, self.start = pen, None

        def __call__(self, input_ids, scores):
            if self.start is None:
                self.start = input_ids.shape[1]
            d = self.pen.get(input_ids.shape[1] - self.start)
            if d:
                idx = torch.tensor(list(d.keys()), device=scores.device)
                val = torch.tensor(list(d.values()), device=scores.device, dtype=scores.dtype)
                scores[:, idx] = scores[:, idx] - val
            return scores

    return PositionalPenalty(penalties)


def build_penalties(cand, eot, delta, conf=0.6, window=1):
    lp = np.asarray(cand["lp"])
    toks = list(cand["ids"]) + [eot]
    low = [t for t in range(len(lp)) if lp[t] < math.log(conf)]
    if not low:
        low = [int(np.argmin(lp))]
    pen = defaultdict(dict)
    for t in low:
        for s in range(max(0, t - window), t + window + 1):
            pen[s][toks[t]] = max(pen[s].get(toks[t], 0.0), delta)
    return dict(pen)


class ASRDecoder:
    def __init__(self, path, dtype="auto", max_new_tokens=192):
        import torch
        from transformers import AutoProcessor
        self.torch = torch
        self.dtype = select_dtype(dtype)
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = load_whisper(path, self.dtype).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.device = next(self.model.parameters()).device
        tok = self.processor.tokenizer
        self.tok = tok
        self.eot = tok.convert_tokens_to_ids("<|endoftext|>")
        self.prefix = {l: [tok.convert_tokens_to_ids("<|startoftranscript|>"), tok.convert_tokens_to_ids(f"<|{l}|>"),
                           tok.convert_tokens_to_ids("<|transcribe|>"), tok.convert_tokens_to_ids("<|notimestamps|>")]
                       for l in ("vi", "en")}
        self.max_new_tokens = max_new_tokens

    def features(self, wave):
        f = self.processor.feature_extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                             return_attention_mask=True)
        return f.input_features.to(self.device, self.dtype), f.attention_mask.to(self.device)

    def _gen(self, feats, mask, language, beams, k, sample=False, temperature=0.0, penalties=None, no_repeat=0):
        from transformers import LogitsProcessorList
        kw = dict(input_features=feats, attention_mask=mask, task="transcribe", language=whisper_lang(language),
                  max_new_tokens=self.max_new_tokens, return_timestamps=False)
        if no_repeat:
            kw["no_repeat_ngram_size"] = no_repeat
        if penalties:
            kw["logits_processor"] = LogitsProcessorList([make_penalty_processor(penalties)])
        if sample:
            kw.update(do_sample=True, temperature=temperature, num_beams=1)
        else:
            kw.update(do_sample=False, num_beams=beams)
        seqs = []
        try:
            out = self.model.generate(**kw, num_return_sequences=k)
            out = out.sequences if hasattr(out, "sequences") else out
            seqs = out.tolist()
        except Exception:                                     # older/newer Whisper generate quirks
            for _ in range(k if sample else 1):
                if penalties:
                    kw["logits_processor"] = LogitsProcessorList([make_penalty_processor(penalties)])
                out = self.model.generate(**kw)
                out = out.sequences if hasattr(out, "sequences") else out
                seqs += out.tolist()
        return [[t for t in s if t < self.eot] for s in seqs]

    def rescore(self, feats, cands_ids, language):
        """Teacher-forced token log-probs and entropies: L_ASR = -1/N sum log P(y_t | x, y_<t)."""
        torch = self.torch
        pre = self.prefix[language]
        rows = [pre + ids + [self.eot] for ids in cands_ids]
        L = max(map(len, rows))
        dec = torch.full((len(rows), L), self.eot, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            dec[i, :len(r)] = torch.tensor(r, device=self.device)
        with torch.inference_mode():
            enc = self.model.model.encoder(feats).last_hidden_state
            logits = self.model(encoder_outputs=(enc.expand(len(rows), -1, -1),), decoder_input_ids=dec[:, :-1]).logits.float()
            logp = torch.log_softmax(logits, -1)
            ent = -(logp.exp() * logp).sum(-1)
            tgt = logp.gather(-1, dec[:, 1:, None]).squeeze(-1)
        res = []
        for i, r in enumerate(rows):
            s, e = len(pre) - 1, len(r) - 1                      # predict content tokens + EOT
            res.append((tgt[i, s:e].cpu().numpy(), ent[i, s:e].cpu().numpy()))
        return res

    def round(self, feats, mask, language, seen, specs, rnd):
        torch = self.torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        new = []
        for spec in specs:
            for ids in self._gen(feats, mask, language, **spec["gen"]):
                key = tuple(ids)
                if key in seen:
                    continue
                seen.add(key)
                new.append((ids, spec["method"]))
        cands = []
        if new:
            scored = self.rescore(feats, [n[0] for n in new], language)
            for rank, ((ids, method), (lp, ent)) in enumerate(zip(new, scored)):
                cands.append({"ids": ids, "text": norm(self.tok.decode(ids, skip_special_tokens=True)),
                              "lp": [round(float(x), 4) for x in lp], "ent": [round(float(x), 3) for x in ent],
                              "round": rnd, "rank": rank, "method": method, "lang": language})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return cands, time.perf_counter() - t0


def ladder_specs(rnd, beams, best_cand, eot, delta):
    if rnd == 0:
        return [{"method": "beam", "gen": dict(beams=beams, k=beams)}]
    if rnd == 1:
        return [{"method": "penalty", "gen": dict(beams=beams, k=min(3, beams), no_repeat=4,
                                                    penalties=build_penalties(best_cand, eot, delta))}]
    return [{"method": "penalty2", "gen": dict(beams=beams, k=2, no_repeat=3,
                                                penalties=build_penalties(best_cand, eot, 2 * delta, conf=0.8))},
            {"method": "sample", "gen": dict(beams=1, k=3, sample=True, temperature=0.5, no_repeat=3)}]


def logloss_of(c):
    return -float(np.mean(c["lp"])) if c["lp"] else 99.0


def decode_ladder(dec, wave, language, rounds=3, beams=5, delta=4.0):
    feats, mask = dec.features(wave)
    seen, out, pool = set(), [], []
    for r in range(rounds):
        best = min(pool, key=logloss_of) if pool else None
        cands, t = dec.round(feats, mask, language, seen, ladder_specs(r, beams, best, dec.eot, delta), r)
        pool += cands
        out.append({"time": round(t, 4), "cands": cands})
    return out


def cmd_nbest(a):
    from tqdm.auto import tqdm
    rows = read_manifest(a.manifest)
    if a.max_rows and len(rows) > a.max_rows:
        by = defaultdict(list)
        for r in rows:
            by[r["source"]].append(r)
        rng = random.Random(a.seed)
        rows = []
        for s, lst in by.items():
            rng.shuffle(lst)
            rows += lst[:max(1, round(a.max_rows * len(lst) / sum(map(len, by.values()))))]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {d["record_id"] for d in read_jsonl(out)}
    dec = ASRDecoder(a.asr, a.dtype)
    log(f"N-best ladder: {len(rows)} rows ({len(done)} cached) | rounds={a.rounds} beams={a.beams} asr={a.asr}")
    for r in tqdm(rows, desc=f"ladder {Path(a.out).stem}"):
        if r["record_id"] in done:
            continue
        wave = load_audio(r["audio_path"])
        dur, rms, snr = audio_stats(wave)
        lang = r["language"] if a.language_mode == "label" else "vi"
        lang = "en" if lang == "en" else "vi"
        lad = decode_ladder(dec, wave, lang, rounds=a.rounds, beams=a.beams, delta=a.delta)
        append_jsonl(out, {"record_id": r["record_id"], "source": r["source"], "ref": r["text"],
                           "language": r["language"], "duration": round(dur, 3), "rms_db": round(rms, 2),
                           "snr_db": round(snr, 2), "rounds": lad})


# =============================================================================
# Embedding model (BGE-M3): data, fine-tuning, discrimination eval
# =============================================================================
def light_variant(text):
    """Benign surface variant (casing / punctuation / number verbalisation) -> same information."""
    t = norm(text)
    choice = stable_hash(t, 3)
    if choice == 0:
        return metric_text(t)
    if choice == 1:
        return re.sub(r"[^\w\s]", "", t).lower()
    return verbalize_numbers(t)


def asr_triplets(cache_paths, max_n, seed):
    trip = []
    for p in cache_paths:
        for d in read_jsonl(p):
            ref = d["ref"]
            wrong = []
            for rd in d["rounds"]:
                for c in rd["cands"]:
                    e = utt_errors(ref, c["text"])
                    if e["errors"] > 0 and c["text"]:
                        wrong.append((e["errors"], c["text"]))
            if wrong:
                wrong.sort()
                trip.append({"anchor": ref, "positive": light_variant(ref), "negative": wrong[0][1]})
    random.Random(seed).shuffle(trip)
    return trip[:max_n]


def hf_pairs(repo, config, split, n, cols_try, token=None, trust=False):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True, token=token or None, trust_remote_code=trust)
    out = []
    for ex in ds:
        keys = {k.lower(): k for k in ex.keys()}
        row = {}
        for tgt, cands in cols_try.items():
            for c in cands:
                if c in keys and ex[keys[c]]:
                    v = ex[keys[c]]
                    if isinstance(v, list):
                        v = v[0] if v else None
                    if isinstance(v, dict):
                        v = v.get("text") or next(iter(v.values()), None)
                    row[tgt] = norm(v) if v else None
                    break
        if row.get("anchor") and row.get("positive"):
            out.append({k: v for k, v in row.items() if v})
        if len(out) >= n:
            break
    return out


def cmd_build_emb_data(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    token = a.token or os.environ.get("HF_TOKEN")
    report = {}

    def save(name, rows):
        if rows:
            df = pd.DataFrame(rows)
            cols = ["anchor", "positive"] + (["negative"] if "negative" in df and df["negative"].notna().all() else [])
            df = df.dropna(subset=cols)[cols].drop_duplicates()
            df.to_parquet(out / f"{name}.parquet", index=False)
            report[name] = len(df)
            log(f"  {name}: {len(df):,} rows, columns {cols}")

    tasks = [
        ("mmarco_vi", lambda: hf_pairs("minhnguyent546/mmarco-vietnamese-split", None, "train", a.mmarco,
                                       {"anchor": ["query", "anchor", "question"],
                                        "positive": ["positive", "pos", "passage", "positive_passage"],
                                        "negative": ["negative", "neg", "negative_passage"]}, token)),
        ("wikimatrix_envi", lambda: hf_pairs("sentence-transformers/parallel-sentences-wikimatrix", "en-vi", "train",
                                             a.wikimatrix, {"anchor": ["english", "en", "anchor"],
                                                            "positive": ["non_english", "vi", "positive"]}, token)),
        ("phomt", lambda: hf_pairs("vinai/PhoMT", None, "train", a.phomt,
                                   {"anchor": ["en", "english", "source"], "positive": ["vi", "vietnamese", "target"]},
                                   token)),
        ("viquad2", lambda: viquad_triplets(a.viquad, token)),
    ]
    for name, fn in tasks:
        if (out / f"{name}.parquet").is_file() and not a.overwrite:
            report[name] = "cached"
            continue
        try:
            save(name, fn())
        except Exception as e:
            log(f"  !! {name} skipped: {type(e).__name__}: {str(e)[:200]}")
            report[name] = f"failed: {type(e).__name__}"
    # ASR hard negatives (judge split only) + LifeMate formatting robustness pairs
    trip = asr_triplets(a.judge_cache, a.asr_neg, a.seed)
    save("asr_hardneg", trip)
    if a.lifemate_manifest and Path(a.lifemate_manifest).is_file():
        lm = [{"anchor": r["text"], "positive": light_variant(r["text"])} for r in read_manifest(a.lifemate_manifest)
              if r["source"] == "lifemate"]
        save("lifemate_format", lm)
    ev = asr_triplets(a.val_cache, 3000, a.seed + 1) if a.val_cache else []
    if ev:
        pd.DataFrame(ev).to_parquet(out / "eval_asr_triplets.parquet", index=False)
    save_json(out / "emb_data_report.json", report)


def viquad_triplets(n, token):
    from datasets import load_dataset
    ds = load_dataset("taidng/UIT-ViQuAD2.0", split="train", token=token or None)
    by_title = defaultdict(set)
    for ex in ds:
        by_title[ex.get("title", "")].add(ex["context"])
    rows, rng = [], random.Random(0)
    for ex in ds:
        if ex.get("is_impossible"):
            continue
        others = list(by_title[ex.get("title", "")] - {ex["context"]})
        if not others:
            continue
        rows.append({"anchor": norm(ex["question"]), "positive": norm(ex["context"]), "negative": norm(rng.choice(others))})
        if len(rows) >= n:
            break
    return rows


def discrimination_acc(model, df, bs=64):
    if df is None or len(df) == 0:
        return None
    ea = model.encode(df["anchor"].tolist(), batch_size=bs, normalize_embeddings=True, convert_to_numpy=True)
    ep = model.encode(df["positive"].tolist(), batch_size=bs, normalize_embeddings=True, convert_to_numpy=True)
    en = model.encode(df["negative"].tolist(), batch_size=bs, normalize_embeddings=True, convert_to_numpy=True)
    sp, sn = (ea * ep).sum(1), (ea * en).sum(1)
    return {"acc": float((sp > sn).mean()), "margin": float((sp - sn).mean()), "n": int(len(df))}


def cmd_train_emb(a):
    import pandas as pd
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
    data = Path(a.data)
    model = SentenceTransformer(a.model, device="cuda")
    model.max_seq_length = a.max_len
    evdf = pd.read_parquet(data / "eval_asr_triplets.parquet") if (data / "eval_asr_triplets.parquet").is_file() else None
    before = discrimination_acc(model, evdf)
    log("Discrimination BEFORE fine-tune:", before)
    caps = parse_kv(a.caps, int)
    train = {}
    for p in sorted(data.glob("*.parquet")):
        if p.stem.startswith("eval_"):
            continue
        df = pd.read_parquet(p)
        cap = caps.get(p.stem, a.default_cap)
        if len(df) > cap:
            df = df.sample(cap, random_state=a.seed)
        train[p.stem] = Dataset.from_pandas(df.reset_index(drop=True))
        log(f"  {p.stem}: {len(df):,}")
    assert train, "no embedding training data"
    bf16 = torch.cuda.is_bf16_supported()
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(a.out) / "ckpt"), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        learning_rate=a.lr, warmup_ratio=0.05, bf16=bf16, fp16=not bf16, seed=a.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES, multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        save_strategy="no", logging_steps=50, report_to="none", max_steps=a.max_steps if a.max_steps > 0 else -1,
        gradient_checkpointing=True)
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=a.mini_batch)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=train, loss=loss).train()
    after = discrimination_acc(model, evdf)
    log("Discrimination AFTER fine-tune:", after)
    out = Path(a.out) / "emb_best"
    model.save(str(out))
    torch.save({k: v.half().contiguous() for k, v in model[0].auto_model.state_dict().items()},
               Path(a.out) / "bge_m3_cs_fp16.pth")
    save_json(Path(a.out) / "emb_metrics.json", {"before": before, "after": after, "base": a.model})
    shutil.rmtree(Path(a.out) / "ckpt", ignore_errors=True)


# =============================================================================
# Features (log-loss, confidence, repetition, code-switch, semantic consistency)
# =============================================================================
class Embedder:
    def __init__(self, path, bank_texts=None, bank_cache=None, cap=60000):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.m = SentenceTransformer(path, device="cuda" if torch.cuda.is_available() else "cpu")
        self.m.max_seq_length = 128
        self.bank = None
        if bank_cache and Path(bank_cache).is_file():
            self.bank = torch.load(bank_cache, map_location=self.m.device)
        elif bank_texts:
            texts = list(dict.fromkeys(t for t in bank_texts if t))
            random.Random(0).shuffle(texts)
            self.bank = self.encode(texts[:cap]).half()
            if bank_cache:
                torch.save(self.bank, bank_cache)

    def encode(self, texts):
        return self.m.encode([t if t else "<empty>" for t in texts], batch_size=64, normalize_embeddings=True,
                             convert_to_tensor=True, show_progress_bar=False).float()

    def knn(self, e, k=5):
        if self.bank is None:
            z = self.torch.zeros(len(e), device=e.device)
            return z, z
        s = e.half() @ self.bank.T
        top = s.topk(min(k, s.shape[1]), dim=1).values.float()
        return top[:, 0], top.mean(1)


BASIC_FEATS = ["logloss", "nll_max", "nll_p90", "nll_std", "frac_low", "ent_mean", "ent_max", "eot_nll", "n_tok",
               "tok_per_s", "n_words", "words_per_s", "comp_ratio", "max_rep3", "uniq_ratio", "en_ratio", "n_switch",
               "empty", "round", "rank", "is_sample", "is_penalty", "ll_gap0", "mbr0"]
SEM_FEATS = ["sem_centroid0", "sem_top0", "sem_knn1", "sem_knn5"]
CTX_BASE = ["duration", "rms_db", "snr_db", "ctx_ll_top", "ctx_mbr", "ctx_en", "ctx_wps"]
CTX_SEM = ["ctx_disp"]


def basic_features(c, dur):
    lp = np.asarray(c["lp"], dtype=np.float64)
    nll = -lp if len(lp) else np.array([20.0])
    ent = np.asarray(c["ent"], dtype=np.float64) if c["ent"] else np.array([0.0])
    words = metric_text(c["text"]).split()
    enr, nsw = switch_stats(words)
    b = c["text"].encode("utf-8")
    return {"logloss": float(nll.mean()), "nll_max": float(nll.max()), "nll_p90": float(np.percentile(nll, 90)),
            "nll_std": float(nll.std()), "frac_low": float((lp < math.log(0.5)).mean()) if len(lp) else 1.0,
            "ent_mean": float(ent.mean()), "ent_max": float(ent.max()), "eot_nll": float(nll[-1]),
            "n_tok": len(c["ids"]), "tok_per_s": len(c["ids"]) / max(dur, 0.1), "n_words": len(words),
            "words_per_s": len(words) / max(dur, 0.1), "comp_ratio": len(b) / max(1, len(zlib.compress(b))),
            "max_rep3": max_ngram_repeat(words), "uniq_ratio": len(set(words)) / max(1, len(words)),
            "en_ratio": enr, "n_switch": nsw, "empty": int(not words), "round": c["round"], "rank": c["rank"],
            "is_sample": int(c["method"] == "sample"), "is_penalty": int(c["method"].startswith("penalty"))}


def norm_ed(a, b):
    a, b = metric_text(a).split(), metric_text(b).split()
    return edit_distance(a, b) / max(1, len(a), len(b))


def record_features(d, emb):
    """Feature rows for every candidate of one ladder record (same code path as the online loop)."""
    r0 = d["rounds"][0]["cands"]
    allc = [c for rd in d["rounds"] for c in rd["cands"]]
    if not allc:
        return []
    dur = d["duration"]
    feats = [basic_features(c, dur) for c in allc]
    ll0 = [f["logloss"] for f, c in zip(feats, allc) if c["round"] == 0] or [99.0]
    best0 = r0[int(np.argmin([logloss_of(c) for c in r0]))] if r0 else allc[0]
    ctx = {"duration": dur, "rms_db": d["rms_db"], "snr_db": d["snr_db"], "ctx_ll_top": min(ll0),
           "ctx_mbr": float(np.mean([norm_ed(x["text"], y["text"]) for i, x in enumerate(r0) for y in r0[i + 1:]]))
           if len(r0) > 1 else 0.0}
    bw = metric_text(best0["text"]).split()
    ctx["ctx_en"] = switch_stats(bw)[0]
    ctx["ctx_wps"] = len(bw) / max(dur, 0.1)
    t0 = time.perf_counter()
    if emb is not None:
        E = emb.encode([c["text"] for c in allc])
        n0 = sum(c["round"] == 0 for c in allc)
        E0 = E[:n0] if n0 else E[:1]
        cen = emb.torch.nn.functional.normalize(E0.mean(0, keepdim=True), dim=-1)
        top0 = E[allc.index(best0)][None]
        k1, k5 = emb.knn(E)
        sc, st = (E @ cen.T).squeeze(1), (E @ top0.T).squeeze(1)
        ctx["ctx_disp"] = float(1 - (E0 @ E0.T).mean()) if len(E0) > 1 else 0.0
        sem = [{"sem_centroid0": float(sc[i]), "sem_top0": float(st[i]), "sem_knn1": float(k1[i]),
                "sem_knn5": float(k5[i])} for i in range(len(allc))]
    else:
        ctx["ctx_disp"] = 0.0
        sem = [{k: 0.0 for k in SEM_FEATS} for _ in allc]
    emb_t = (time.perf_counter() - t0) / len(allc)
    rows = []
    for c, f, s in zip(allc, feats, sem):
        f.update(s)
        f.update(ctx)
        f["ll_gap0"] = f["logloss"] - min(ll0)
        f["mbr0"] = float(np.mean([norm_ed(c["text"], x["text"]) for x in r0 if x is not c])) if len(r0) > 1 else 0.0
        f["emb_time"] = emb_t
        rows.append(f)
    return rows


def cmd_features(a):
    import pandas as pd
    from tqdm.auto import tqdm
    emb = None
    if a.emb and a.emb != "none":
        bank = [r["text"] for r in read_manifest(a.bank_manifest)] if a.bank_manifest else None
        emb = Embedder(a.emb, bank, a.bank_cache)
    out = []
    for spec in a.caches:
        split, path = spec.split("=", 1)
        for d in tqdm(read_jsonl(path), desc=f"features {split}"):
            allc = [c for rd in d["rounds"] for c in rd["cands"]]
            times = np.cumsum([rd["time"] for rd in d["rounds"]])
            for c, f in zip(allc, record_features(d, emb)):
                e = utt_errors(d["ref"], c["text"])
                f.update({"split": split, "record_id": d["record_id"], "source": d["source"], "ref": d["ref"],
                          "text": c["text"], "errors": e["errors"], "n_ref": e["n"],
                          "wer": e["errors"] / max(1, e["n"]), "time_cum": float(times[c["round"]])})
                out.append(f)
    df = pd.DataFrame(out)
    df.to_parquet(a.out, index=False)
    log(f"features: {len(df):,} candidates | {df.record_id.nunique():,} records -> {a.out}")


# =============================================================================
# XGBoost judge + dynamic threshold T_dynamic(x)
# =============================================================================
QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]


def xgb_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_clf(seed=42, n=600):
    import xgboost as xgb
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=n, learning_rate=0.05,
                             max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
                             tree_method="hist", device=xgb_device(), random_state=seed)


def make_qreg(seed=42, n=400):
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES), n_estimators=n,
                            learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.9,
                            tree_method="hist", device=xgb_device(), random_state=seed)


def top0_rows(df):
    r0 = df[df["round"] == 0]
    return r0.loc[r0.groupby("record_id")["logloss"].idxmin()]


def oof_fit_predict(df, feats, ctx, seed=42, folds=5):
    """Out-of-fold P(err) for candidates and T_q(x) for records -> unbiased threshold tuning."""
    from sklearn.model_selection import GroupKFold
    y = (df["errors"] > 0).astype(int).values
    groups = df["record_id"].values
    p = np.zeros(len(df))
    for tr, te in GroupKFold(folds).split(df, y, groups):
        m = make_clf(seed)
        m.fit(df.iloc[tr][feats], y[tr])
        p[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
    top = top0_rows(df).set_index("record_id")
    recs = top.index.values
    tq = pd_frame_zeros(recs)
    ok = top["errors"].values == 0
    rec_fold = {r: stable_hash("fold|" + r, folds) for r in recs}
    for f in range(folds):
        trm = np.array([rec_fold[r] != f for r in recs]) & ok
        tem = np.array([rec_fold[r] == f for r in recs])
        if trm.sum() < 20 or tem.sum() == 0:
            continue
        q = make_qreg(seed)
        q.fit(top[ctx].values[trm], top["logloss"].values[trm])
        tq[tem] = np.sort(q.predict(top[ctx].values[tem]), axis=1)
    return p, dict(zip(recs, tq))


def pd_frame_zeros(recs):
    return np.zeros((len(recs), len(QUANTILES)))


def build_tensors(df, score_col, R):
    """Per record and round: the candidate chosen by `score_col` among the pool of rounds <= r."""
    recs = df["record_id"].unique()
    N = len(recs)
    idx = np.full((N, R), -1)
    pos = {r: i for i, r in enumerate(recs)}
    g = df.groupby("record_id", sort=False)
    for rid, sub in g:
        i = pos[rid]
        for r in range(R):
            pool = sub[sub["round"] <= r]
            idx[i, r] = pool[score_col].idxmin() if len(pool) else idx[i, r - 1]
    return recs, idx


def simulate(df, idx, cond, lam=0.0):
    """cond: bool (N,R) accept matrix. First accepted round wins; last round is forced."""
    cond = cond.copy()
    cond[:, -1] = True
    rstar = cond.argmax(1)
    chosen = idx[np.arange(len(idx)), rstar]
    errs = df.loc[chosen, "errors"].values
    nref = df.loc[chosen, "n_ref"].values
    wer = errs.sum() / max(1, nref.sum())
    rounds = (rstar + 1).mean()
    return wer + lam * (rounds - 1), wer, rounds, chosen


def search_policies(df, p_full, p_nosem, tq_full, tq_nosem, R, lam):
    """Vectorised grid search of thresholds on OOF predictions (thousands of configs in seconds)."""
    df = df.copy()
    df["p_full"], df["p_nosem"] = p_full, p_nosem
    res = {}
    recs, idx_ll = build_tensors(df, "logloss", R)
    L = df["logloss"].values
    Lsel = L[idx_ll]
    # (1) static threshold on log-loss (classical confidence gating)
    best = None
    for th in np.quantile(L, np.linspace(0.05, 0.99, 60)):
        j = simulate(df, idx_ll, Lsel <= th, lam)
        if best is None or j[0] < best[0][0]:
            best = (j, {"theta": float(th)})
    res["static_ll"] = best[1]
    # (2) BGE-M3 only: semantic re-ranking score, static threshold, no XGBoost
    best = None
    for w1 in [0.0, 0.5, 1.0, 2.0]:
        for w2 in [0.0, 0.5, 1.0, 2.0]:
            df["s_bge"] = df["logloss"] - w1 * df["sem_centroid0"] - w2 * df["sem_knn1"]
            _, idx_b = build_tensors(df, "s_bge", R)
            S = df["s_bge"].values[idx_b]
            for th in np.quantile(df["s_bge"].values, np.linspace(0.05, 0.99, 30)):
                j = simulate(df, idx_b, S <= th, lam)
                if best is None or j[0] < best[0][0]:
                    best = (j, {"w1": w1, "w2": w2, "theta": float(th)})
    res["bge_only"] = best[1]
    # (3..5) XGBoost policies
    for name, pcol, tq, dynamic in [("xgb_nosem_dyn", "p_nosem", tq_nosem, True),
                                    ("xgb_full_static", "p_full", tq_full, False),
                                    ("xgb_full_dyn", "p_full", tq_full, True)]:
        _, idx_p = build_tensors(df, pcol, R)
        P = df[pcol].values[idx_p]
        Ls = L[idx_p]
        T = np.stack([tq[r] for r in recs])
        best = None
        taus = list(np.linspace(0.05, 0.95, 19)) + [1.0]
        for tau in taus:
            if not dynamic:
                j = simulate(df, idx_p, P <= tau, lam)
                if best is None or j[0] < best[0][0]:
                    best = (j, {"tau": float(tau)})
                continue
            for qi in range(len(QUANTILES)):
                for beta in [-0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4]:
                    cond = (Ls <= T[:, qi:qi + 1] + beta) & (P <= tau)
                    j = simulate(df, idx_p, cond, lam)
                    if best is None or j[0] < best[0][0]:
                        best = (j, {"tau": float(tau), "q": QUANTILES[qi], "qi": qi, "beta": beta})
        res[name] = best[1]
        log(f"  {name}: judge-OOF WER {best[0][1]*100:.2f}% rounds {best[0][2]:.2f} params {best[1]}")
    return res


def cmd_train_judge(a):
    import pandas as pd
    df = pd.read_parquet(a.features)
    jd = df[df["split"] == "judge"].reset_index(drop=True)
    R = int(jd["round"].max()) + 1
    feats_full = BASIC_FEATS + SEM_FEATS + CTX_BASE + CTX_SEM
    feats_nosem = BASIC_FEATS + CTX_BASE
    log(f"judge candidates {len(jd):,} | records {jd.record_id.nunique():,} | rounds {R} | "
        f"P(err) base rate {(jd.errors > 0).mean():.3f}")
    p_full, tq_full = oof_fit_predict(jd, feats_full, CTX_BASE + CTX_SEM, a.seed)
    p_nosem, tq_nosem = oof_fit_predict(jd, feats_nosem, CTX_BASE, a.seed)
    from sklearn.metrics import log_loss, roc_auc_score
    y = (jd.errors > 0).astype(int)
    rep = {"oof_logloss_full": float(log_loss(y, np.clip(p_full, 1e-6, 1 - 1e-6))),
           "oof_auc_full": float(roc_auc_score(y, p_full)),
           "oof_logloss_nosem": float(log_loss(y, np.clip(p_nosem, 1e-6, 1 - 1e-6))),
           "oof_auc_nosem": float(roc_auc_score(y, p_nosem))}
    log("XGBoost judge OOF:", rep)
    t0 = time.time()
    pol = search_policies(jd, p_full, p_nosem, tq_full, tq_nosem, R, a.lam)
    log(f"threshold search done in {time.time()-t0:.1f}s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ycls = (jd.errors > 0).astype(int)
    top = top0_rows(jd)
    okm = top["errors"].values == 0
    for tag, feats, ctx in [("full", feats_full, CTX_BASE + CTX_SEM), ("nosem", feats_nosem, CTX_BASE)]:
        m = make_clf(a.seed)
        m.fit(jd[feats], ycls)
        m.save_model(str(out / f"clf_{tag}.json"))
        q = make_qreg(a.seed)
        q.fit(top[ctx].values[okm], top["logloss"].values[okm])
        q.save_model(str(out / f"qreg_{tag}.json"))
        imp = sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])[:15]
        rep[f"importance_{tag}"] = [(f, float(v)) for f, v in imp]
    save_json(out / "judge_policy.json", {"policies": pol, "rounds": R, "lam": a.lam, "quantiles": QUANTILES,
                                          "feats_full": feats_full, "feats_nosem": feats_nosem,
                                          "ctx_full": CTX_BASE + CTX_SEM, "ctx_nosem": CTX_BASE, "report": rep})
    log("Saved judge ->", out)


def load_judge(d):
    import xgboost as xgb
    d = Path(d)
    pol = read_json(d / "judge_policy.json")
    models = {}
    for tag in ["full", "nosem"]:
        c = xgb.XGBClassifier()
        c.load_model(str(d / f"clf_{tag}.json"))
        q = xgb.XGBRegressor()
        q.load_model(str(d / f"qreg_{tag}.json"))
        models[tag] = (c, q)
    return pol, models


# =============================================================================
# evaluate : Table 2 / Table 3
# =============================================================================
def apply_policy(df, name, pol, models, R):
    P = pol["policies"]
    df = df.copy()
    recs = df["record_id"].unique()
    if name == "beam_top1":
        chosen = df[(df["round"] == 0)].groupby("record_id")["rank"].idxmin().reindex(recs).values
        return chosen, np.ones(len(recs))
    if name == "oracle":
        chosen = df.groupby("record_id")["errors"].idxmin().reindex(recs).values
        return chosen, np.full(len(recs), R)
    if name in ("static_ll",):
        _, idx = build_tensors(df, "logloss", R)
        cond = df["logloss"].values[idx] <= P[name]["theta"]
    elif name == "bge_only":
        q = P[name]
        df["s_bge"] = df["logloss"] - q["w1"] * df["sem_centroid0"] - q["w2"] * df["sem_knn1"]
        _, idx = build_tensors(df, "s_bge", R)
        cond = df["s_bge"].values[idx] <= q["theta"]
    else:
        base = "xgb_full_dyn" if name in ("xgb_full_dyn", "full_no_redecode") else name
        tag = "nosem" if "nosem" in base else "full"
        c, qr = models[tag]
        df["p"] = c.predict_proba(df[pol["feats_" + tag]])[:, 1]
        _, idx = build_tensors(df, "p", R)
        q = P[base]
        cond = df["p"].values[idx] <= q["tau"]
        if base.endswith("_dyn"):
            top = top0_rows(df).set_index("record_id").reindex(recs)
            T = np.sort(qr.predict(top[pol["ctx_" + tag]].values), axis=1)[:, q["qi"]]
            cond = cond & (df["logloss"].values[idx] <= T[:, None] + q["beta"])
        if name == "full_no_redecode":
            return idx[:, 0], np.ones(len(recs))
    cond = cond.copy()
    cond[:, -1] = True
    rstar = cond.argmax(1)
    return idx[np.arange(len(idx)), rstar], rstar + 1.0


def summarize(df, chosen, rounds, dur_by_rec):
    sel = df.loc[chosen]
    m = corpus_metrics(sel["ref"].tolist(), sel["text"].tolist())
    t = sel["time_cum"].values + sel["emb_time"].values * 8
    m["RTF"] = float(t.sum() / max(1e-9, dur_by_rec.reindex(sel["record_id"]).sum()))
    m["rounds"] = float(np.mean(rounds))
    return m


def fmt(x, pct=True):
    return f"{100*x:.2f}" if pct else f"{x:.3f}"


def cmd_evaluate(a):
    import pandas as pd
    df = pd.read_parquet(a.features)
    pol, models = load_judge(a.judge)
    R = pol["rounds"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    test = df[df["split"] == a.split].reset_index(drop=True)
    subsets = {"All": test}
    if (test.source == "lifemate").any():
        subsets["LifeMate"] = test[test.source == "lifemate"]
    cs_src = {k for k, v in SOURCES.items() if v.get("cs")}
    if test.source.isin(cs_src).any():
        subsets["CS (SEA+ViMedCSS)"] = test[test.source.isin(cs_src)]
    policies = ["beam_top1", "static_ll", "bge_only", "xgb_nosem_dyn", "xgb_full_static", "xgb_full_dyn",
                "full_no_redecode", "oracle"]
    for sname, sub in subsets.items():
        sub = sub.reset_index(drop=True)
        dur = sub.groupby("record_id")["duration"].first()
        results[sname] = {}
        for p in policies:
            ch, rd = apply_policy(sub, p, pol, models, R)
            results[sname][p] = summarize(sub, ch, rd, dur)
        if sname == "All":
            ch, rd = apply_policy(sub, "xgb_full_dyn", pol, models, R)
            sel = sub.loc[ch]
            sel[["record_id", "source", "ref", "text", "errors", "n_ref", "logloss"]].assign(rounds=rd).to_csv(
                out / "final_predictions.csv", index=False)
    # baselines (pretrained models, beam top-1)
    for spec in a.baseline_features or []:
        name, path = spec.split("=", 1)
        bdf = pd.read_parquet(path)
        bdf = bdf[bdf["split"] == a.split].reset_index(drop=True)
        for sname in subsets:
            sub = bdf if sname == "All" else bdf[bdf.source == "lifemate"] if sname == "LifeMate" else bdf[bdf.source.isin(cs_src)]
            sub = sub.reset_index(drop=True)
            if len(sub):
                ch, rd = apply_policy(sub, "beam_top1", pol, models, R)
                results[sname]["baseline:" + name] = summarize(sub, ch, rd, sub.groupby("record_id")["duration"].first())
    save_json(out / "main_results.json", results)
    # ---- Table 2
    rows2 = [(k, k.split(":", 1)[1] + " (pretrained, beam)") for k in results["All"] if k.startswith("baseline:")]
    rows2 += [("beam_top1", "ASR fine-tuned (LoRA), beam top-1"),
              ("static_ll", "+ Static threshold (log-loss) + re-decode"),
              ("xgb_full_dyn", "**+ BGE-M3 + XGBoost T_dynamic (đề xuất)**")]
    hdr = "| Hệ thống | " + " | ".join(f"WER {s} (%)" for s in subsets) + " | MER All (%) | E-WER All (%) | Halluc. (%) | RTF |"
    t2 = [hdr, "|" + "---|" * (len(subsets) + 5)]
    for key, label in rows2:
        cells = [fmt(results[s][key]["WER"]) if key in results[s] else "–" for s in subsets]
        A = results["All"][key]
        t2.append(f"| {label} | " + " | ".join(cells) + f" | {fmt(A['MER'])} | {fmt(A['E_WER'])} | {fmt(A['HALLUC'])} | {A['RTF']:.3f} |")
    # ---- Table 3
    A = results["All"]
    base = A["beam_top1"]["WER"]
    rows3 = [("beam_top1", "(a) ASR fine-tuned, beam top-1"),
             ("static_ll", "(b) + re-decode, static log-loss threshold"),
             ("bge_only", "(c) + BGE-M3 semantic rerank only (no XGBoost)"),
             ("xgb_nosem_dyn", "(d) + XGBoost only (no BGE-M3 features), T_dynamic"),
             ("xgb_full_static", "(e) BGE-M3 + XGBoost, static τ (no T_dynamic)"),
             ("xgb_full_dyn", "(f) **Full: BGE-M3 + XGBoost + T_dynamic**"),
             ("full_no_redecode", "(g) Full without re-decode (rerank round-0 only)"),
             ("oracle", "(h) Oracle (best candidate in pool)")]
    t3 = ["| Cấu hình | WER (%) | ΔWER (điểm) | E-WER (%) | Halluc. (%) | Số vòng TB | RTF |", "|---|---|---|---|---|---|---|"]
    for key, label in rows3:
        m = A[key]
        t3.append(f"| {label} | {fmt(m['WER'])} | {100*(m['WER']-base):+.2f} | {fmt(m['E_WER'])} | {fmt(m['HALLUC'])} | "
                  f"{m['rounds']:.2f} | {m['RTF']:.3f} |")
    (out / "table2_main.md").write_text("\n".join(t2), encoding="utf-8")
    (out / "table3_ablation.md").write_text("\n".join(t3), encoding="utf-8")
    numbers = {"T2_FULL_WER": fmt(A["xgb_full_dyn"]["WER"]), "T2_FT_WER": fmt(A["beam_top1"]["WER"]),
               "T2_STATIC_WER": fmt(A["static_ll"]["WER"]), "T2_FULL_EWER": fmt(A["xgb_full_dyn"]["E_WER"]),
               "T2_FULL_HALLUC": fmt(A["xgb_full_dyn"]["HALLUC"]), "T2_STATIC_HALLUC": fmt(A["static_ll"]["HALLUC"]),
               "T2_FULL_RTF": f"{A['xgb_full_dyn']['RTF']:.3f}", "T2_FULL_ROUNDS": f"{A['xgb_full_dyn']['rounds']:.2f}",
               "T3_BGE_WER": fmt(A["bge_only"]["WER"]), "T3_XGB_WER": fmt(A["xgb_nosem_dyn"]["WER"]),
               "T3_STATICTAU_WER": fmt(A["xgb_full_static"]["WER"]), "T3_ORACLE_WER": fmt(A["oracle"]["WER"]),
               "TABLE2": "\n".join(t2), "TABLE3": "\n".join(t3)}
    for k, v in results["All"].items():
        if k.startswith("baseline:"):
            numbers["T2_BASE_" + re.sub(r"\W+", "_", k.split(":", 1)[1]).upper() + "_WER"] = fmt(v["WER"])
    save_json(out / "paper_numbers.json", numbers)
    log("\nTABLE 2\n" + "\n".join(t2) + "\n\nTABLE 3\n" + "\n".join(t3))
    w = A["xgb_full_dyn"]["WER"]
    log(f"\nFULL SYSTEM WER on held-out '{a.split}': {w*100:.2f}%  -> "
        + ("đạt mục tiêu < 3%" if w < 0.03 else "CHƯA đạt < 3% (báo cáo trung thực, không chỉnh số)"))


# =============================================================================
# Online pipeline: decode -> judge -> re-decode until accepted (or max rounds)
# =============================================================================
class CSASRPipeline:
    def __init__(self, asr_dir, emb_dir, judge_dir, bank_cache=None, rounds=None, beams=5, delta=4.0):
        self.dec = ASRDecoder(asr_dir)
        self.emb = Embedder(emb_dir, bank_cache=bank_cache)
        self.pol, self.models = load_judge(judge_dir)
        self.R = rounds or self.pol["rounds"]
        self.beams, self.delta = beams, delta
        self.q = self.pol["policies"]["xgb_full_dyn"]

    def __call__(self, wave, language="vi"):
        import pandas as pd
        dur, rms, snr = audio_stats(wave)
        feats, mask = self.dec.features(wave)
        seen, rounds, trace, pool = set(), [], [], []
        clf, qreg = self.models["full"]
        T = None
        for r in range(self.R):
            best_ll = min(pool, key=logloss_of) if pool else None
            cands, t = self.dec.round(feats, mask, language, seen,
                                      ladder_specs(r, self.beams, best_ll, self.dec.eot, self.delta), r)
            rounds.append({"time": t, "cands": cands})
            pool += cands
            d = {"duration": dur, "rms_db": rms, "snr_db": snr, "rounds": rounds}
            X = pd.DataFrame(record_features(d, self.emb))
            p = clf.predict_proba(X[self.pol["feats_full"]])[:, 1]
            if T is None:
                ctx = X[self.pol["ctx_full"]].iloc[[0]].values
                T = float(np.sort(qreg.predict(ctx), axis=1)[0, self.q["qi"]]) + self.q["beta"]
            j = int(np.argmin(p))
            ok = bool(p[j] <= self.q["tau"] and X["logloss"].iloc[j] <= T)
            trace.append({"round": r, "best": pool[j]["text"], "logloss": round(float(X["logloss"].iloc[j]), 4),
                          "T_dynamic": round(T, 4), "p_err": round(float(p[j]), 4), "accepted": ok,
                          "n_candidates": len(pool), "sec": round(t, 3)})
            if ok:
                break
        return pool[j]["text"], trace


def split_long_audio(wave, max_s=28.0):
    if len(wave) <= max_s * SAMPLE_RATE:
        return [wave]
    out, start, hop = [], 0, int(0.02 * SAMPLE_RATE)
    while start < len(wave):
        end = min(len(wave), start + int(max_s * SAMPLE_RATE))
        if end < len(wave):
            lo = start + int(0.6 * (end - start))
            seg = wave[lo:end]
            n = max(1, len(seg) // hop)
            e = np.array([np.mean(seg[i * hop:(i + 1) * hop] ** 2) for i in range(n)])
            end = lo + int(np.argmin(e)) * hop
        out.append(wave[start:end])
        start = end
    return out


def cmd_transcribe(a):
    pipe = CSASRPipeline(a.asr, a.emb, a.judge, a.bank_cache)
    wave = load_audio(a.audio)
    texts = []
    for k, seg in enumerate(split_long_audio(wave)):
        txt, trace = pipe(seg, a.language)
        texts.append(txt)
        for t in trace:
            log(f"[seg {k}] round {t['round']}: L_ASR={t['logloss']:.3f} T_dyn={t['T_dynamic']:.3f} "
                f"P(err)={t['p_err']:.3f} accepted={t['accepted']} | {t['best']}")
    log("\nFINAL:", " ".join(texts))


# =============================================================================
# budget / export
# =============================================================================
def cmd_budget(a):
    from transformers import AutoConfig, WhisperForConditionalGeneration
    from sentence_transformers import SentenceTransformer
    import torch
    asr = WhisperForConditionalGeneration.from_pretrained(a.asr, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    n_asr = sum(p.numel() for p in asr.parameters())
    del asr
    emb = SentenceTransformer(a.emb, device="cpu")
    n_emb = sum(p.numel() for p in emb.parameters())
    n_xgb = 0
    for f in Path(a.judge).glob("*.json"):
        if f.name.startswith(("clf_", "qreg_")):
            n_xgb += len(re.findall(r'"split_conditions"|\d+\.\d+', f.read_text()[:0])) or 0
            try:
                j = read_json(f)
                for t in j["learner"]["gradient_booster"]["model"]["trees"]:
                    n_xgb += 2 * len(t["split_conditions"])
            except Exception:
                pass
    tot = n_asr + n_emb + n_xgb
    rep = {"asr": n_asr, "embedding": n_emb, "xgboost_nodes_params": n_xgb, "total": tot,
           "limit": MAX_TOTAL_PARAMS, "remaining": MAX_TOTAL_PARAMS - tot, "ok": tot <= MAX_TOTAL_PARAMS}
    save_json(a.output, rep)
    log(json.dumps(rep, indent=1))
    assert rep["ok"], "TOTAL PIPELINE EXCEEDS 1.5B"


def cmd_export(a):
    import zipfile
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    asr, emb, judge = Path(a.asr), Path(a.emb_root), Path(a.judge)
    copies = [(asr / "asr_merged_fp16.pth", out / "asr_merged_fp16.pth"),
              (emb / "bge_m3_cs_fp16.pth", out / "bge_m3_cs_fp16.pth")]
    for s, d in copies:
        if s.is_file():
            shutil.copy2(s, d)
    bundle = out / "cs_asr_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in asr.glob("*.json"):
            z.write(f, f"asr_config/{f.name}")
        for f in list(asr.glob("*.txt")) + list(asr.glob("*.model")) + list(asr.glob("*.tiktoken")):
            z.write(f, f"asr_config/{f.name}")
        st = emb / "emb_best"
        for f in st.rglob("*"):
            if f.is_file() and not f.name.endswith((".safetensors", ".bin", ".pt", ".pth")):
                z.write(f, f"emb_config/{f.relative_to(st)}")
        for f in judge.glob("*.json"):
            z.write(f, f"judge/{f.name}")
        for extra in a.extra or []:
            p = Path(extra)
            if p.is_file():
                z.write(p, f"extra/{p.name}")
        z.write(__file__, "cs_asr.py")
    for f in out.iterdir():
        log(f"{f.name}: {f.stat().st_size/2**20:.1f} MB")


def load_from_pth(asr_pth, asr_config_dir, emb_pth=None, emb_base=DEFAULT_EMB):
    """Rebuild models from exported .pth files (weights_only load)."""
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    m = WhisperForConditionalGeneration(WhisperConfig.from_pretrained(asr_config_dir))
    m.load_state_dict(torch.load(asr_pth, map_location="cpu", weights_only=True), strict=False)
    e = None
    if emb_pth:
        from sentence_transformers import SentenceTransformer
        e = SentenceTransformer(emb_base, device="cpu")
        e[0].auto_model.load_state_dict(torch.load(emb_pth, map_location="cpu", weights_only=True))
    return m, e


# =============================================================================
# CLI
# =============================================================================
def build_cli():
    P = argparse.ArgumentParser(description="LifeMate CS-ASR ULTRA")
    S = P.add_subparsers(dest="cmd", required=True)

    s = S.add_parser("collect-lifemate"); s.set_defaults(fn=cmd_collect_lifemate)
    s.add_argument("--train-dir", required=True); s.add_argument("--out", required=True)
    s.add_argument("--clearvoice-manifest", default="")

    s = S.add_parser("collect-hf"); s.set_defaults(fn=cmd_collect_hf)
    s.add_argument("--out-dir", required=True); s.add_argument("--sources", required=True)
    s.add_argument("--hours", default=""); s.add_argument("--test-hours", default="3")
    s.add_argument("--giga2-shards", default="2"); s.add_argument("--token", default="")
    s.add_argument("--seed", type=int, default=42); s.add_argument("--overwrite", action="store_true")

    s = S.add_parser("prepare"); s.set_defaults(fn=cmd_prepare)
    s.add_argument("--manifests", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--lifemate-test-ids", default=""); s.add_argument("--val-frac", type=float, default=0.07)
    s.add_argument("--judge-frac", type=float, default=0.08); s.add_argument("--cs-test-frac", type=float, default=0.05)
    s.add_argument("--cs-val-frac", type=float, default=0.02); s.add_argument("--max-val", type=int, default=1500)
    s.add_argument("--max-judge", type=int, default=6000); s.add_argument("--max-test", type=int, default=4000)
    s.add_argument("--seed", type=int, default=42)

    s = S.add_parser("train-asr"); s.set_defaults(fn=cmd_train_asr)
    s.add_argument("--model", default=DEFAULT_ASR); s.add_argument("--data", required=True)
    s.add_argument("--out", required=True); s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--samples-per-epoch", type=int, default=0); s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--batch", type=int, default=8); s.add_argument("--accum", type=int, default=2)
    s.add_argument("--rank", type=int, default=64); s.add_argument("--dropout", type=float, default=0.05)
    s.add_argument("--targets", default="q_proj,k_proj,v_proj,out_proj,fc1,fc2")
    s.add_argument("--augment", type=float, default=1.0); s.add_argument("--alpha", type=float, default=0.6)
    s.add_argument("--boost", default="lifemate=4,sea_cs_envi=2,vimedcss=2")
    s.add_argument("--warmup", type=float, default=0.04); s.add_argument("--patience", type=int, default=2)
    s.add_argument("--eval-max", type=int, default=800); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--dtype", default="auto"); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--max-params", type=int, default=1_000_000_000); s.add_argument("--init-adapter", default="")
    s.add_argument("--only-sources", default=""); s.add_argument("--resume", action="store_true")
    s.add_argument("--baseline", action="store_true"); s.add_argument("--log-every", type=int, default=200)

    s = S.add_parser("merge"); s.set_defaults(fn=cmd_merge)
    s.add_argument("--model", default=DEFAULT_ASR); s.add_argument("--adapter", required=True)
    s.add_argument("--out", required=True)

    s = S.add_parser("nbest"); s.set_defaults(fn=cmd_nbest)
    s.add_argument("--asr", required=True); s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True); s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--beams", type=int, default=5); s.add_argument("--delta", type=float, default=4.0)
    s.add_argument("--max-rows", type=int, default=0); s.add_argument("--language-mode", default="label")
    s.add_argument("--dtype", default="auto"); s.add_argument("--seed", type=int, default=42)

    s = S.add_parser("build-emb-data"); s.set_defaults(fn=cmd_build_emb_data)
    s.add_argument("--out", required=True); s.add_argument("--judge-cache", nargs="+", default=[])
    s.add_argument("--val-cache", nargs="*", default=[]); s.add_argument("--lifemate-manifest", default="")
    s.add_argument("--mmarco", type=int, default=150000); s.add_argument("--wikimatrix", type=int, default=80000)
    s.add_argument("--phomt", type=int, default=80000); s.add_argument("--viquad", type=int, default=30000)
    s.add_argument("--asr-neg", type=int, default=50000); s.add_argument("--token", default="")
    s.add_argument("--seed", type=int, default=42); s.add_argument("--overwrite", action="store_true")

    s = S.add_parser("train-emb"); s.set_defaults(fn=cmd_train_emb)
    s.add_argument("--model", default=DEFAULT_EMB); s.add_argument("--data", required=True)
    s.add_argument("--out", required=True); s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--batch", type=int, default=64); s.add_argument("--mini-batch", type=int, default=16)
    s.add_argument("--lr", type=float, default=1e-5); s.add_argument("--max-len", type=int, default=160)
    s.add_argument("--max-steps", type=int, default=0); s.add_argument("--default-cap", type=int, default=60000)
    s.add_argument("--caps", default="asr_hardneg=50000,lifemate_format=8000"); s.add_argument("--seed", type=int, default=42)

    s = S.add_parser("features"); s.set_defaults(fn=cmd_features)
    s.add_argument("--emb", required=True); s.add_argument("--caches", nargs="+", required=True)
    s.add_argument("--bank-manifest", default=""); s.add_argument("--bank-cache", default="")
    s.add_argument("--out", required=True)

    s = S.add_parser("train-judge"); s.set_defaults(fn=cmd_train_judge)
    s.add_argument("--features", required=True); s.add_argument("--out", required=True)
    s.add_argument("--lam", type=float, default=0.002); s.add_argument("--seed", type=int, default=42)

    s = S.add_parser("evaluate"); s.set_defaults(fn=cmd_evaluate)
    s.add_argument("--features", required=True); s.add_argument("--judge", required=True)
    s.add_argument("--out", required=True); s.add_argument("--split", default="test")
    s.add_argument("--baseline-features", nargs="*", default=[])

    s = S.add_parser("transcribe"); s.set_defaults(fn=cmd_transcribe)
    s.add_argument("--asr", required=True); s.add_argument("--emb", required=True)
    s.add_argument("--judge", required=True); s.add_argument("--bank-cache", default="")
    s.add_argument("--audio", required=True); s.add_argument("--language", default="vi")

    s = S.add_parser("budget"); s.set_defaults(fn=cmd_budget)
    s.add_argument("--asr", required=True); s.add_argument("--emb", required=True)
    s.add_argument("--judge", required=True); s.add_argument("--output", required=True)

    s = S.add_parser("export"); s.set_defaults(fn=cmd_export)
    s.add_argument("--asr", required=True); s.add_argument("--emb-root", required=True)
    s.add_argument("--judge", required=True); s.add_argument("--out", required=True)
    s.add_argument("--extra", nargs="*", default=[])
    return P


if __name__ == "__main__":
    args = build_cli().parse_args()
    args.fn(args)
