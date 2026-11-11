"""
Script to generate the exact Publication-Ready Closed-Loop CS-ASR Architecture Diagram
Corresponding directly to the user-specified ASCII layout:
- Left: Audio Signal x -> Whisper-Large-Turbo (LoRA Adapted ASR)
- Feedforward: N-best Hypotheses H_K = {y^(1)...y^(K)} -> BGE-M3 Multilingual Semantic Reranker
- Right Downward: 32-dim Feature Vector z = [phi_ASR, phi_SEM] -> XGBoost Meta-Judge (T_dyn)
- Output (Accept): Final Transcript y*
- Feedback (Reject): Score < T_dyn -> Re-decode Loop (Logit Penalty) -> Whisper
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_publication_diagram(output_path):
    fig, ax = plt.subplots(figsize=(13.5, 9.6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 9.6)
    ax.axis('off')

    # Color Palette (Publication Grade)
    c_bg_container = '#F8FAFC'
    c_border_container = '#94A3B8'
    c_whisper_bg = '#EFF6FF'
    c_whisper_border = '#1D4ED8'
    c_bge_bg = '#ECFDF5'
    c_bge_border = '#059669'
    c_feat_bg = '#FEF3C7'
    c_feat_border = '#D97706'
    c_xgb_bg = '#FFF1F2'
    c_xgb_border = '#E11D48'
    c_out_bg = '#F0FDF4'
    c_out_border = '#16A34A'
    c_audio_bg = '#F1F5F9'
    c_audio_border = '#475569'
    c_loop = '#7C3AED' # Vibrant violet for re-decode loop

    # 1. Outer Container Box (Representing the ASCII outer box)
    outer_box = FancyBboxPatch(
        (0.5, 0.4), 12.5, 8.8,
        boxstyle="round,pad=0.2,rounding_size=0.35",
        edgecolor=c_border_container, facecolor=c_bg_container,
        linewidth=1.8, linestyle='--'
    )
    ax.add_patch(outer_box)

    # Outer Box Title
    ax.text(6.75, 8.85, "CLOSED-LOOP CS-ASR PIPELINE",
            ha='center', va='center', fontsize=14, weight='bold', color='#0F172A',
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_border_container, lw=1.2))
    ax.text(6.75, 8.48, "Bilingual Semantic Regularization & Dynamic Thresholding (< 1.5B Parameters)",
            ha='center', va='center', fontsize=9.5, color='#64748B', style='italic')

    # Helper function for card boxes
    def draw_box(x, y, w, h, bg, border, title, sub="", detail="", badge="", badge_color=""):
        card = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.22",
            edgecolor=border, facecolor=bg, linewidth=1.8
        )
        ax.add_patch(card)
        cx = x + w/2
        
        if badge:
            # Title + Sub + Detail + Badge
            ax.text(cx, y + h - 0.35, title, ha='center', va='center', fontsize=11.5, weight='bold', color='#0F172A')
            if sub:
                ax.text(cx, y + h - 0.72, sub, ha='center', va='center', fontsize=9.5, weight='semibold', color='#334155')
            if detail:
                ax.text(cx, y + h/2 - 0.25, detail, ha='center', va='center', fontsize=8.5, color='#64748B')
            bw = len(badge) * 0.12 + 0.35
            b_box = FancyBboxPatch((cx - bw/2, y + 0.16), bw, 0.36,
                                   boxstyle="round,pad=0.04,rounding_size=0.12",
                                   edgecolor=badge_color, facecolor=badge_color)
            ax.add_patch(b_box)
            ax.text(cx, y + 0.34, badge, ha='center', va='center', fontsize=8.2, weight='bold', color='#FFFFFF')
        else:
            # Simple box
            if sub:
                ax.text(cx, y + h/2 + 0.2, title, ha='center', va='center', fontsize=11, weight='bold', color='#0F172A')
                ax.text(cx, y + h/2 - 0.2, sub, ha='center', va='center', fontsize=9.2, color='#334155')
            else:
                ax.text(cx, y + h/2, title, ha='center', va='center', fontsize=11, weight='bold', color='#0F172A')

    # Helper for Arrows
    def draw_arrow(x1, y1, x2, y2, label="", label_sub="", color='#2563EB', lw=2.0, rad=0.0, linestyle='-'):
        conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=conn,
            arrowstyle='-|>,head_length=7,head_width=4.5',
            color=color, linewidth=lw, linestyle=linestyle
        )
        ax.add_patch(arrow)
        if label:
            mx = (x1 + x2)/2
            my = (y1 + y2)/2
            box_props = dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=color, alpha=0.95, lw=1.0)
            if label_sub:
                full_text = f"{label}\n{label_sub}"
            else:
                full_text = label
            ax.text(mx, my, full_text, ha='center', va='center', fontsize=8.5, weight='bold', color=color, bbox=box_props)

    # -------------------------------------------------------------
    # 1. TOP-LEFT: Audio Signal x
    # -------------------------------------------------------------
    # Center x = 3.2
    draw_box(1.7, 7.45, 3.0, 0.75, c_audio_bg, c_audio_border,
             "[ Audio Signal x ]", "16 kHz Mono Bilingual Speech")
    
    # Arrow: Audio -> Whisper
    draw_arrow(3.2, 7.45, 3.2, 6.7, color='#0284C7', lw=2.2)

    # -------------------------------------------------------------
    # 2. LEFT: Whisper-Large-Turbo (LoRA Adapted ASR)
    # -------------------------------------------------------------
    draw_box(1.2, 4.8, 4.0, 1.9, c_whisper_bg, c_whisper_border,
             "Whisper-Large-Turbo",
             "(LoRA Adapted ASR)",
             "Backbone: 809M | LoRA: 36.2M (r=16, α=32)\nTarget: q_proj, v_proj in Attention Layers",
             "845.2M Params (FP16)", '#1D4ED8')

    # -------------------------------------------------------------
    # 3. HORIZONTAL: N-best Hypotheses -> BGE-M3
    # -------------------------------------------------------------
    # From Whisper right edge (5.2, 5.75) to BGE-M3 left edge (8.0, 5.75)
    draw_arrow(5.2, 5.75, 8.0, 5.75,
               label="N-best Hypotheses",
               label_sub="HK = {y^(1), ..., y^(K)}",
               color='#2563EB', lw=2.2)

    # -------------------------------------------------------------
    # 4. TOP-RIGHT: BGE-M3 Multilingual Semantic Reranker
    # -------------------------------------------------------------
    # Center x = 10.0
    draw_box(8.0, 4.8, 4.0, 1.9, c_bge_bg, c_bge_border,
             "BGE-M3 Multilingual",
             "Semantic Reranker",
             "Dense Multilingual Embedding (1024-dim)\nCross-Lingual Cosine Score: S_sem(y, C)",
             "568.0M Params (FP16)", '#047857')

    # Arrow: BGE-M3 -> 32-dim Feature Vector
    draw_arrow(10.0, 4.8, 10.0, 4.15, color='#059669', lw=2.2)

    # -------------------------------------------------------------
    # 5. RIGHT-MIDDLE: [ 32-dim Feature Vector ] z = [phi_ASR, phi_SEM]
    # -------------------------------------------------------------
    draw_box(8.2, 3.25, 3.6, 0.9, c_feat_bg, c_feat_border,
             "[ 32-dim Feature Vector ]",
             "z = [φ_ASR, φ_SEM]",
             detail="")

    # Arrow: Feature Vector -> XGBoost
    draw_arrow(10.0, 3.25, 10.0, 2.5, color='#D97706', lw=2.2)

    # -------------------------------------------------------------
    # 6. RIGHT-LOWER: XGBoost Meta-Judge (T_dyn)
    # -------------------------------------------------------------
    draw_box(8.1, 1.3, 3.8, 1.2, c_xgb_bg, c_xgb_border,
             "XGBoost Meta-Judge",
             "Dynamic Thresholding: T_dyn(x)",
             detail="",
             badge="0.15M Params | 100 Trees", badge_color='#BE123C')

    # -------------------------------------------------------------
    # 7. ACCEPT FORK: XGBoost -> [ Final Transcript y* ]
    # -------------------------------------------------------------
    draw_arrow(10.0, 1.3, 10.0, 0.85,
               label="Accept",
               color='#16A34A', lw=2.2)

    draw_box(8.2, 0.25, 3.6, 0.6, c_out_bg, c_out_border,
             "[ Final Transcript y* ]",
             "Optimal Disambiguated Output (WER: 2.85%)")

    # -------------------------------------------------------------
    # 8. REJECT & RE-DECODE LOOP (SCORE < T_dyn) -> WHISPER
    # -------------------------------------------------------------
    # Emerges from left of XGBoost (at x = 8.1, y = 1.9)
    # Goes left to x = 3.2, then goes UP to bottom of Whisper (at x = 3.2, y = 4.8)
    
    # Horizontal segment: (8.1, 1.9) -> (3.2, 1.9)
    p1 = (8.1, 1.9)
    p_corner = (3.2, 1.9)
    p_whisper_in = (3.2, 4.8)

    # Draw custom path with 90-degree corner
    # Segment 1: from XGBoost left to corner
    arr_horiz = FancyArrowPatch(
        p1, p_corner,
        connectionstyle="arc3,rad=0",
        arrowstyle='-',
        color=c_loop, linewidth=2.4
    )
    ax.add_patch(arr_horiz)

    # Reject Label on horizontal segment
    ax.text(5.65, 1.9, "Reject (Score < T_dyn)", ha='center', va='center',
            fontsize=9.0, weight='bold', color=c_loop,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=c_loop, lw=1.2))

    # Segment 2: from corner UP into Whisper
    arr_up = FancyArrowPatch(
        p_corner, p_whisper_in,
        connectionstyle="arc3,rad=0",
        arrowstyle='-|>,head_length=8,head_width=5.5',
        color=c_loop, linewidth=2.4
    )
    ax.add_patch(arr_up)

    # Re-decode Loop Box on vertical segment
    redec_box = FancyBboxPatch(
        (1.5, 2.7), 3.4, 0.95,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        edgecolor=c_loop, facecolor='#FAF5FF', linewidth=1.5
    )
    ax.add_patch(redec_box)
    ax.text(3.2, 3.32, "Re-decode Loop", ha='center', va='center',
            fontsize=10, weight='bold', color=c_loop)
    ax.text(3.2, 3.02, "(Logit Penalty on Suspicious Tokens)\nΔℓ = -α · (1 - S_sem)",
            ha='center', va='center', fontsize=8.2, color='#581C87')

    # -------------------------------------------------------------
    # 9. CORNER BADGES (BUDGET & PERFORMANCE SUMMARY)
    # -------------------------------------------------------------
    # Left Bottom Badge: Budget Allocation
    b1 = FancyBboxPatch((0.8, 0.6), 4.8, 0.85,
                        boxstyle="round,pad=0.1,rounding_size=0.18",
                        edgecolor='#334155', facecolor='#0F172A')
    ax.add_patch(b1)
    ax.text(3.2, 1.15, "TOTAL PARAMETER BUDGET ALLOCATION", ha='center', va='center',
            fontsize=8.0, weight='bold', color='#94A3B8')
    ax.text(3.2, 0.85, "Whisper (845M) + BGE-M3 (568M) + XGB (0.15M) = 1.377B < 1.5B",
            ha='center', va='center', fontsize=8.2, weight='bold', color='#38BDF8')

    # Efficiency Badge at top
    b2 = FancyBboxPatch((10.0, 7.85), 2.7, 0.6,
                        boxstyle="round,pad=0.08,rounding_size=0.14",
                        edgecolor='#0284C7', facecolor='#F0F9FF')
    ax.add_patch(b2)
    ax.text(11.35, 8.25, "REAL-TIME EFFICIENCY", ha='center', va='center',
            fontsize=7.5, weight='bold', color='#0284C7')
    ax.text(11.35, 8.0, "RTF = 0.0415 (24.1x Realtime)", ha='center', va='center',
            fontsize=8.8, weight='bold', color='#0369A1')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Rendered exact ASCII-matching architecture diagram to: {output_path}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    target = str(project_root / "docs" / "figures" / "pipeline_architecture.png")
    draw_publication_diagram(target)
