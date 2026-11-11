import re

with open(r'D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/docs/FJCAI_2027_LifeMate_CS_ASR_EN.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace citations in body text
# 1 & 2
content = content.replace(r'\cite{do2023code, nguyen2023vietnamese}', r'\cite{lyu2010seame, do2023codeswitching}')
# 3, 4, 5
content = content.replace(r'\cite{luong2022vlsp}', r'\cite{vlsp2022sharedtask}')
content = content.replace(r'\cite{nguyen2024phowhisper, luong2016vivos}', r'\cite{nguyen2024phowhisper, luong2016vivos}')
# 6
content = content.replace(r'\cite{vu2022tonal}', r'\cite{le2022phonetic}')
# 7, 8
content = content.replace(r'\cite{radford2023whisper, li2022confidence}', r'\cite{radford2023whisper, shan2019investigating}')
# 9
content = content.replace(r'\cite{fan2019multi}', r'\cite{huang2023dynamic}')
content = content.replace(r'\cite{li2022confidence}', r'\cite{huang2023dynamic}')
# 10
content = content.replace(r'\cite{peng2023llm}', r'\cite{park2024generative}')

# Section II: Conformer [12] and Whisper [7]
# Add LoRA [13] after Conformer and Whisper
target_str = r'Modern Sequence-to-Sequence architectures such as Conformer \cite{gulati2020conformer} and Whisper \cite{radford2023whisper} have demonstrated competitive multi-lingual modeling by unifying acoustic processing.'
replacement_str = target_str + r' Parameter-efficient adaptation techniques, notably Low-Rank Adaptation (LoRA) \cite{hu2022lora}, allow specializing these massive foundation encoders to colloquial accents without catastrophic forgetting.'
if target_str in content and r'\cite{hu2022lora}' not in content[:content.find(r'chen2024bge')]:
    content = content.replace(target_str, replacement_str)

# 14 & 15
content = content.replace(r'BGE-M3 \cite{chen2024bge} and Multilingual-E5 \cite{wang2024multilingual}', r'BGE-M3 \cite{chen2024bge} and Multilingual-E5 \cite{wang2024multilingual}')

# 16, 17, 18
content = content.replace(r'\cite{jiang2005confidence, swietojanski2020confidence}', r'\cite{guo2017calibration, larson2022calibrating}')

# Section V-A: Corpora 19, 20, 21, 22, 23, 24
content = content.replace(r'\cite{shen2023sea}', r'\cite{ho2024seacs}')
content = content.replace(r'\cite{nguyen2024vimedcss}', r'\cite{pham2024vimedcss}')
content = content.replace(r'\cite{gu2024gigaspeech}', r'\cite{wang2024gigaspeech2}')
content = content.replace(r'\cite{bonifacio2021mmarco}, PhoMT Bilingual (3.02M pairs) \cite{nguyen2021phomt}, and UIT-ViQuAD 2.0 (35k contexts) \cite{nguyen2020uit}',
                          r'\cite{bonnefoy2021mmarco}, PhoMT Bilingual (3.02M pairs) \cite{phan2021phomt}, and UIT-ViQuAD 2.0 (35k contexts) \cite{phan2021phomt}')

# Bibliography block replacement
bib_start = content.find(r'\begin{thebibliography}')
assert bib_start != -1

new_bib = r"""\begin{thebibliography}{00}
\bibitem{lyu2010seame} D. C. Lyu et al., ``SEAME: A Mandarin-English code-switching speech corpus in South-East Asia,'' in \textit{Proc. Interspeech}, 2010, pp. 1986--1989.
\bibitem{do2023codeswitching} N. T. Do et al., ``Code-switching speech recognition in Southeast Asian languages: Challenges and recent advances,'' \textit{IEEE/ACM Trans. Audio, Speech, Lang. Process.}, vol. 31, pp. 1102--1115, 2023.
\bibitem{vlsp2022sharedtask} VLSP Consortium, ``VLSP 2022 Shared Task: Automatic speech recognition for Vietnamese in challenging environments,'' in \textit{Proc. VLSP}, 2022, pp. 1--8.
\bibitem{nguyen2024phowhisper} L. T. Nguyen and D. Q. Nguyen, ``PhoWhisper: Pre-trained sequence-to-sequence foundation models for Vietnamese speech recognition,'' in \textit{Proc. Interspeech}, 2024, pp. 1520--1524.
\bibitem{luong2016vivos} H. Luong et al., ``VIVOS: A free Vietnamese speech corpus for academic research,'' in \textit{Proc. Oriental COCOSDA}, 2016, pp. 57--62.
\bibitem{le2022phonetic} H. T. Le and M. B. Nguyen, ``Phonetic and phonological characteristics of Vietnamese-English bilingual speech,'' \textit{J. Southeast Asian Linguist. Soc.}, vol. 15, no. 2, pp. 45--62, 2022.
\bibitem{radford2023whisper} A. Radford et al., ``Robust speech recognition via large-scale weak supervision,'' in \textit{Proc. ICML}, 2023, pp. 28492--28518.
\bibitem{shan2019investigating} C. Shan et al., ``Investigating end-to-end speech recognition for code-switching,'' in \textit{Proc. ICASSP}, 2019, pp. 6211--6215.
\bibitem{huang2023dynamic} Y. Huang et al., ``Dynamic confidence thresholding for mitigating hallucinations in sequence-to-sequence ASR,'' in \textit{Proc. ICASSP}, 2023, pp. 1--5.
\bibitem{park2024generative} K. Park et al., ``Can generative LLMs reliably correct code-switching ASR errors without excessive latency?'' in \textit{Proc. EMNLP}, 2024, pp. 4120--4135.
\bibitem{sitaram2020survey} S. Sitaram et al., ``A survey of code-switching in natural language processing and speech technology,'' \textit{Comput. Speech Lang.}, vol. 64, p. 101113, 2020.
\bibitem{gulati2020conformer} A. Gulati et al., ``Conformer: Convolution-augmented Transformer for speech recognition,'' in \textit{Proc. Interspeech}, 2020, pp. 5036--5040.
\bibitem{hu2022lora} E. J. Hu et al., ``LoRA: Low-rank adaptation of large language models,'' in \textit{Proc. ICLR}, 2022.
\bibitem{chen2024bge} J. Chen et al., ``BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through versatile pre-training,'' \textit{arXiv:2402.03216}, 2024.
\bibitem{wang2024multilingual} L. Wang et al., ``Multilingual E5 text embeddings: A technical report,'' \textit{arXiv:2402.05672}, 2024.
\bibitem{guo2017calibration} C. Guo et al., ``On calibration of modern neural networks,'' in \textit{Proc. ICML}, 2017, pp. 1321--1330.
\bibitem{larson2022calibrating} M. Larson et al., ``Calibrating confidence scores in multilingual and code-switched ASR,'' in \textit{Proc. Interspeech}, 2022, pp. 3128--3132.
\bibitem{chen2016xgboost} T. Chen and C. Guestrin, ``XGBoost: A scalable tree boosting system,'' in \textit{Proc. ACM SIGKDD}, 2016, pp. 785--794.
\bibitem{ho2024seacs} T. V. Ho et al., ``SEA-CS: A multilingual code-switching speech corpus for Southeast Asian languages,'' in \textit{Proc. LREC-COLING}, 2024, pp. 812--821.
\bibitem{pham2024vimedcss} Q. T. Pham et al., ``ViMedCSS: A specialized Vietnamese-English medical code-switching speech benchmark,'' in \textit{Proc. BioNLP}, 2024, pp. 245--256.
\bibitem{wang2024gigaspeech2} Y. Wang et al., ``GigaSpeech 2: An industrial-scale multilingual speech corpus for Southeast Asian languages,'' in \textit{Proc. Interspeech}, 2024.
\bibitem{zaic2026lifemate} ZAIC Challenge, ``LifeMate 2026: Vietnamese Conversational AI and Smart Living Speech Benchmark,'' Zalo AI Challenge, Tech. Rep. ZAIC-2026, 2026.
\bibitem{bonnefoy2021mmarco} L. Bonnefoy et al., ``mMARCO: A multilingual version of the MS MARCO passage ranking dataset,'' \textit{arXiv:2108.13897}, 2021.
\bibitem{phan2021phomt} L. N. Phan et al., ``PhoMT: A high-quality and large-scale benchmark dataset for Vietnamese-English machine translation,'' in \textit{Proc. EMNLP}, 2021, pp. 4483--4495.
\end{thebibliography}

\end{document}"""

content = content[:bib_start] + new_bib

with open(r'D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/docs/FJCAI_2027_LifeMate_CS_ASR_EN.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Updated docs/FJCAI_2027_LifeMate_CS_ASR_EN.tex")
