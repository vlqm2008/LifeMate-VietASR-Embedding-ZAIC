with open(r'D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\build_fjcai_ieee_2col.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Line 273
text = text.replace(
    'Processing (VLSP) consortium [22] have catalyzed remarkable technological strides for pure Vietnamese speech [23], [24],',
    'Processing (VLSP) consortium [3] have catalyzed remarkable technological strides for pure Vietnamese speech [4], [5],'
)

# 2. Line 288 & 289
text = text.replace(
    'cascade unhindered across subsequent tokens [5]. While recent investigations have explored Large Language Models (LLMs >= 7B) "\n        "for post-ASR grammatical correction, their prohibitive parameter scale (> 14 GB VRAM) and execution latency (RTF > 0.40) preclude deployment on edge devices and real-time interactive dialog systems [6].',
    'cascade unhindered across subsequent tokens [9]. While recent investigations have explored Large Language Models (LLMs >= 7B) "\n        "for post-ASR grammatical correction, their prohibitive parameter scale (> 14 GB VRAM) and execution latency (RTF > 0.40) preclude deployment on edge devices and real-time interactive dialog systems [10].'
)

# 3. Line 311, 312, 313
text = text.replace(
    'acoustic and linguistic models [7]. However, error accumulation from front-end LID components significantly degraded overall transcription quality. "\n        "Modern Sequence-to-Sequence architectures such as Conformer [8] and Whisper [4] have demonstrated competitive multi-lingual modeling by unifying acoustic processing. "\n        "While dedicated Vietnamese speech foundation models such as PhoWhisper [23] and VIVOS-trained architectures [24] exhibit outstanding recognition accuracy on clean native speech',
    'acoustic and linguistic models [11]. However, error accumulation from front-end LID components significantly degraded overall transcription quality. "\n        "Modern Sequence-to-Sequence architectures such as Conformer [12] and Whisper [7] have demonstrated competitive multi-lingual modeling by unifying acoustic processing. "\n        "While dedicated Vietnamese speech foundation models such as PhoWhisper [4] and VIVOS-trained architectures [5] exhibit outstanding recognition accuracy on clean native speech'
)

# 4. Line 327
text = text.replace(
    'Gradient Boosted Decision Trees (XGBoost) [21] have proven exceptionally effective on low-dimensional tabular representations',
    'Gradient Boosted Decision Trees (XGBoost) [18] have proven exceptionally effective on low-dimensional tabular representations'
)

# 5. Line 358
text = text.replace(
    'we apply Low-Rank Adaptation (LoRA) [9] to the query and value projection matrices',
    'we apply Low-Rank Adaptation (LoRA) [13] to the query and value projection matrices'
)

# 6. Line 505 & 506
text = text.replace(
    'PhoWhisper-Medium [23], VIVOS baseline [24])',
    'PhoWhisper-Medium [4], VIVOS baseline [5])'
)
text = text.replace(
    'consortium [22]',
    'consortium [3]'
)

with open(r'D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\build_fjcai_ieee_2col.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("[OK] Replacements complete!")
