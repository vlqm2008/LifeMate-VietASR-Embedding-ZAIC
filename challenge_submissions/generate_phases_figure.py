#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path("D:/Dataset/CHALLENGE_TRACK1_SUBMISSIONS/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Figure 7: Training Phases & Performance Gain
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# 1. Challenge Track: Score Progression across 5 Phases
phases = [
    "Phase 1\nASR Acoustic\nOptimization",
    "Phase 2\nLLM Trajectory\nSFT (Qwen3.5)",
    "Phase 3\nE1P Prompt &\nCalendar Grounding",
    "Phase 4\nRecovery Injection\n& Retry Cap 2",
    "Phase 5\nStep Ensemble &\nSelf-Correction"
]
scores_val = [44.0, 58.7, 70.91, 72.71, 89.15]
scores_pub = [38.8, 48.18, 58.02, 61.26, 70.80]

x = np.arange(len(phases))
width = 0.35

r1 = ax1.bar(x - width/2, scores_val, width, label="Validation Score", color="#1976D2", edgecolor="black")
r2 = ax1.bar(x + width/2, scores_pub, width, label="Public Score (Actual/Est)", color="#388E3C", edgecolor="black")

ax1.set_ylabel("Điểm số Thách thức (0 - 100)", fontsize=11, fontweight="bold")
ax1.set_title("Lộ Trình Training Challenge: Điểm Số Qua 5 Phase", fontsize=12, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(phases, fontsize=9)
ax1.set_ylim(25, 100)
ax1.legend(frameon=True, fontsize=10)

for bar in r1:
    h = bar.get_height()
    ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for bar in r2:
    h = bar.get_height()
    ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# 2. Research Paper: WER & Error Reduction across 4 Phases
paper_phases = [
    "Phase 1\nZero-shot\nWhisper-Turbo",
    "Phase 2\nLoRA Acoustic\nSFT (r=16)",
    "Phase 3\nBGE-M3 Semantic\nReranking",
    "Phase 4\nXGBoost T_dyn\nClosed Loop"
]
wer_vals = [13.45, 6.82, 4.38, 2.85]
ewer_vals = [22.30, 11.40, 7.15, 4.15]

x2 = np.arange(len(paper_phases))
r3 = ax2.bar(x2 - width/2, wer_vals, width, label="Overall WER (%)", color="#D32F2F", edgecolor="black")
r4 = ax2.bar(x2 + width/2, ewer_vals, width, label="Loanword E-WER (%)", color="#F57C00", edgecolor="black")

ax2.set_ylabel("Tỷ Lệ Lỗi (%) - Thấp hơn là tốt hơn", fontsize=11, fontweight="bold")
ax2.set_title("Lộ Trình Training Bài Báo: Giảm Lỗi Qua 4 Phase", fontsize=12, fontweight="bold", pad=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(paper_phases, fontsize=9)
ax2.set_ylim(0, 26)
ax2.legend(frameon=True, fontsize=10)

for bar in r3:
    h = bar.get_height()
    ax2.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for bar in r4:
    h = bar.get_height()
    ax2.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.tight_layout()
fig_path = OUT_DIR / "figure7_training_phases_gain.png"
fig.savefig(fig_path)
plt.close(fig)
print(f"[OK] Generated {fig_path.name}")

# Copy to artifacts visuals
import shutil
artifact_vis = Path("C:/Users/Admin/.gemini/antigravity/brain/27bfc71a-186f-49fe-a91a-e74fff86b671/visuals")
shutil.copy(fig_path, artifact_vis / fig_path.name)

# Copy to Git repo
git_vis = Path("D:/Dataset/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/challenge_submissions/visualizations")
if git_vis.exists():
    shutil.copy(fig_path, git_vis / fig_path.name)
