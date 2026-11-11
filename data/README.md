# Dataset Preparation & Format Specification

This directory describes the data organization for training, validating, and testing the **LifeMate CS-ASR** closed-loop pipeline in compliance with **LifeMate CS-ASR** standards and **Decision No. 2557/QĐ-BKHCN**.

---

## 1. Directory Structure

When downloading or preparing local datasets, organize audio and transcript files as follows:

```
data/
├── splits/
│   ├── train.csv         # 3,544 conversational utterances (~4.8 hrs)
│   ├── val.csv           # 706 utterances (~1.0 hr)
│   └── test.csv          # 786 evaluation utterances (~1.07 hrs)
├── wavs/                 # 16kHz mono audio files (.wav)
│   ├── utt_00001.wav
│   └── ...
└── vocabulary/
    └── loanword_lexicon.txt # 2,400 English-Vietnamese code-switching boundary tokens
```

---

## 2. Transcript CSV Format

Each split (`train.csv`, `val.csv`, `test.csv`) must follow the standardized header format:

```csv
audio_path,transcript,duration,domain,loanword_count
wavs/utt_00001.wav,"mai schedule meeting với team lúc tám giờ",3.42,enterprise,2
wavs/utt_00002.wav,"nhớ check-in phòng họp và review pr trước khi deploy",4.15,devops,4
```

### Column Definitions:
* `audio_path`: Relative or absolute path to the 16 kHz mono `.wav` audio file.
* `transcript`: Normalized ground-truth reference text preserving English loanword spellings.
* `duration`: Audio segment duration in seconds.
* `domain`: Utterance domain (`enterprise`, `devops`, `medical`, `smarthome`, `general`).
* `loanword_count`: Number of English lexical code-switching tokens in the utterance.

---

## 3. Supported Benchmark Corpora

1. **LifeMate CS Benchmark (ZAIC 2026):**
   * Primary conversational benchmark: 5,036 utterances total (786 test utterances).
   * Spontaneous Vietnamese speech with frequent English IT, office, and smart-home loanwords.
2. **SEA Code-Switching Speech:**
   * 45.3 hours of bilingual conversational speech across Southeast Asian domains.
3. **ViMedCSS (Medical Code-Switching):**
   * 32.64 hours of clinical dialogue with medical English terminology and pharmaceutical names.
4. **BGE-M3 Fine-tuning Corpora:**
   * Triplet pairs from UIT-ViQuAD 2.0, PhoMT, and WikiMatrix EN-VI for contrastive semantic alignment.

---

## 4. Acoustic Audio Preprocessing

To ensure 100% reproducibility and prevent acoustic domain mismatch:
* **Sampling Rate:** All audio is resampled to **16,000 Hz** mono PCM.
* **Normalization:** Amplitude normalized to $-20\text{ dBFS}$ to eliminate recording volume disparities.
* **Max Audio Length:** Clipped or padded to 30.0 seconds according to Whisper specifications.
