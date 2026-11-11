#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEEE Publication-Grade System Architecture Diagram Generator
LifeMate: Closed-Loop Bilingual Speech Recognition for Vietnamese-English Code-Switching
Strict adherence to IEEE Transactions / ICASSP academic block diagram guidelines:
- Zero AI-generated graphics or cartoonish art.
- Minimalist, crisp, vector-grade engineering schematic.
- Clean rectangular blocks, formal mathematical signal flows, and standard control-loop feedback.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon

def build_ieee_architecture_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 5.8)
    ax.axis('off')

    # Typography & Academic Palette
    font_family = 'DejaVu Serif'
    c_line = '#1E293B'      # Dark charcoal for primary block borders (1.2 pt)
    c_fill_main = '#FFFFFF' # Pure white for primary processing units
    c_fill_sub = '#F8FAFC'  # Subtle off-white for data representations
    c_fill_lora = '#EFF6FF' # Subtle light blue tint for LoRA adaptation
    c_text_title = '#0F172A'# Primary dark text
    c_text_math = '#1E293B' # Mathematical notation text
    c_text_sub = '#475569'  # Secondary descriptive annotations
    c_arrow = '#0F172A'     # Crisp black/charcoal arrows
    c_feedback = '#B91C1C'  # Subdued academic dark red for rejection feedback loop

    # Helper: Draw Rectangular Functional Block
    def draw_block(x, y, w, h, title, line2='', line3='', line4='', fill=c_fill_main, edge=c_line, lw=1.2, ls='-'):
        rect = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=2)
        ax.add_patch(rect)
        cx = x + w / 2.0
        
        lines = [l for l in [title, line2, line3, line4] if l]
        n = len(lines)
        if n == 1:
            ax.text(cx, y + h*0.5, lines[0], ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_text_title, family=font_family, zorder=4)
        elif n == 2:
            ax.text(cx, y + h*0.65, lines[0], ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_text_title, family=font_family, zorder=4)
            ax.text(cx, y + h*0.35, lines[1], ha='center', va='center', fontsize=8.0, color=c_text_math, family=font_family, zorder=4)
        elif n == 3:
            ax.text(cx, y + h*0.75, lines[0], ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_text_title, family=font_family, zorder=4)
            ax.text(cx, y + h*0.48, lines[1], ha='center', va='center', fontsize=8.0, color=c_text_math, family=font_family, zorder=4)
            ax.text(cx, y + h*0.22, lines[2], ha='center', va='center', fontsize=7.5, style='italic', color=c_text_sub, family=font_family, zorder=4)
        elif n == 4:
            ax.text(cx, y + h*0.80, lines[0], ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_text_title, family=font_family, zorder=4)
            ax.text(cx, y + h*0.58, lines[1], ha='center', va='center', fontsize=8.0, color=c_text_math, family=font_family, zorder=4)
            ax.text(cx, y + h*0.36, lines[2], ha='center', va='center', fontsize=7.5, color=c_text_sub, family=font_family, zorder=4)
            ax.text(cx, y + h*0.16, lines[3], ha='center', va='center', fontsize=7.2, style='italic', color=c_text_sub, family=font_family, zorder=4)

    # Helper: Draw Orthogonal Arrow
    def draw_arrow(p1, p2, text='', text_pos='above', color=c_arrow, lw=1.2, ls='-'):
        arrow = FancyArrowPatch(p1, p2, arrowstyle='-|>,head_length=5,head_width=3.5',
                                color=color, linewidth=lw, linestyle=ls, zorder=3)
        ax.add_patch(arrow)
        if text:
            mx = (p1[0] + p2[0]) / 2.0
            my = (p1[1] + p2[1]) / 2.0
            offset_y = 0.15 if text_pos == 'above' else -0.18
            offset_x = 0.12 if text_pos == 'right' else (-0.12 if text_pos == 'left' else 0.0)
            ax.text(mx + offset_x, my + offset_y, text, ha='center', va='center',
                    fontsize=7.8, color=color, family=font_family, zorder=5,
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='#FFFFFF', edgecolor='none'))

    # Helper: Draw Decision Diamond
    def draw_diamond(cx, cy, w, h, text1, text2='', fill='#FFFFFF', edge=c_line):
        half_w = w / 2.0
        half_h = h / 2.0
        pts = [[cx, cy + half_h], [cx + half_w, cy], [cx, cy - half_h], [cx - half_w, cy]]
        poly = Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=2)
        ax.add_patch(poly)
        if text2:
            ax.text(cx, cy + 0.1, text1, ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_text_title, family=font_family, zorder=4)
            ax.text(cx, cy - 0.15, text2, ha='center', va='center', fontsize=7.5, color=c_text_math, family=font_family, zorder=4)
        else:
            ax.text(cx, cy, text1, ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_text_title, family=font_family, zorder=4)

    # =========================================================================
    # 1. INPUT LAYER: Audio -> Log-Mel Spectrogram Front-End
    # =========================================================================
    draw_block(0.4, 4.1, 1.7, 1.2,
               'Input Speech', r'$x(t) \in \mathbb{R}^T$', '16 kHz Mono Audio', 'Conversational CS',
               fill=c_fill_sub)

    draw_arrow((2.1, 4.7), (2.7, 4.7), text='')

    draw_block(2.7, 4.1, 1.7, 1.2,
               'Acoustic Front-End', '80-channel Log-Mel', r'$\mathbf{X} \in \mathbb{R}^{T\' \times 80}$', 'STFT Filterbank',
               fill=c_fill_sub)

    draw_arrow((4.4, 4.7), (5.0, 4.7), text=r'$\mathbf{X}$')

    # =========================================================================
    # 2. ACOUSTIC BACKBONE: LoRA-adapted Whisper-Large-v3-Turbo
    # =========================================================================
    draw_block(5.0, 3.8, 2.7, 1.6,
               'Whisper-Large-v3-Turbo',
               r'LoRA: $\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$',
               r'Rank $r=16$, $\alpha=32$ (845.2M)',
               'Beam Search ($B=5$)',
               fill=c_fill_lora)

    draw_arrow((7.7, 4.7), (8.4, 4.7), text=r'$\mathcal{H}_K$', text_pos='above')

    # =========================================================================
    # 3. SEMANTIC REGULARIZER: BGE-M3 Dense Verifier
    # =========================================================================
    draw_block(8.4, 3.8, 2.0, 1.6,
               'BGE-M3 Verifier',
               r'Dense Bi-Encoder (568M)',
               r'$S_{\rm{sem}} = \frac{\mathbf{e}_y^\top \mathbf{e}_C}{\|\mathbf{e}_y\| \|\mathbf{e}_C\|}$',
               r'Context Alignment $C$',
               fill=c_fill_sub)

    # Dialogue Context C input into BGE-M3 from top
    draw_arrow((9.4, 5.7), (9.4, 5.4), text='')
    ax.text(9.4, 5.75, r'Dialogue Context $C$', ha='center', va='bottom',
            fontsize=8.0, fontweight='bold', color=c_text_title, family=font_family)

    # =========================================================================
    # 4. JOINT UNCERTAINTY FEATURE PROJECTION
    # =========================================================================
    # Paths from Whisper and BGE-M3 downward into Feature Vector z
    p_whisper_down = (6.35, 3.8)
    p_bge_down = (9.4, 3.8)

    # Merge into Feature Extraction Vector Block
    draw_block(6.0, 2.3, 4.4, 0.95,
               r'Joint Uncertainty Representation $\mathbf{z} \in \mathbb{R}^{32}$',
               r'$\mathbf{z} = [\mathbf{\phi}_{\rm{ASR}}, \mathbf{\phi}_{\rm{SEM}}, \mathbf{\phi}_{\rm{BND}}]^\top$',
               r'Acoustic Likelihood $\cdot$ Semantic Consistency $\cdot$ Boundary Dynamics',
               fill=c_fill_sub)

    # Arrows into Feature Block
    draw_arrow(p_bge_down, (9.4, 3.25), text=r'$S_{\rm{sem}}$', text_pos='right')
    draw_arrow(p_whisper_down, (6.35, 3.25), text=r'$\mathbf{\phi}_{\rm{ASR}}$', text_pos='left')

    # Arrow from Feature Block to Dynamic Meta-Judge
    draw_arrow((8.2, 2.3), (8.2, 1.8), text=r'$\mathbf{z}$', text_pos='right')

    # =========================================================================
    # 5. DYNAMIC META-JUDGE (XGBoost Quantile Regressor)
    # =========================================================================
    draw_block(6.8, 0.85, 2.8, 0.95,
               'Dynamic Meta-Judge',
               r'XGBoost Regressor (0.15M)',
               r'$T_{\rm{dyn}}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{z} + b)$',
               fill=c_fill_main)

    # Arrow from Meta-Judge leftward to Decision Diamond
    draw_arrow((6.8, 1.32), (5.6, 1.32), text=r'$T_{\rm{dyn}}$', text_pos='above')

    # =========================================================================
    # 6. DECISION GATE (Acceptance vs Closed-Loop Re-decoding)
    # =========================================================================
    draw_diamond(4.8, 1.32, 1.5, 0.95, r'Confidence', r'$J(\mathbf{z}) \geq T_{\rm{dyn}}?$')

    # Branch 1: ACCEPT (Yes) -> Final Transcript
    draw_arrow((4.05, 1.32), (2.8, 1.32), text='Yes (Accept)', text_pos='above', color='#16A34A', lw=1.5)

    draw_block(0.4, 0.85, 2.4, 0.95,
               r'Final Output $y^*$',
               r'$y^* = \arg\max \mathcal{S}_{\rm{joint}}(y)$',
               'WER: 2.85% | E-WER: 4.15%',
               fill='#F0FDF4', edge='#16A34A', lw=1.4)

    # Branch 2: REJECT (No: J < T_dyn) -> Re-decoding Feedback Loop
    p_diam_bottom = (4.8, 0.845)
    p_corner1 = (4.8, 0.35)
    p_corner2 = (3.5, 0.35)
    p_reg_in = (3.5, 1.95)

    # Draw feedback path with orthogonal corners
    ax.plot([p_diam_bottom[0], p_corner1[0]], [p_diam_bottom[1], p_corner1[1]],
            color=c_feedback, linestyle='--', linewidth=1.4, zorder=3)
    ax.plot([p_corner1[0], p_corner2[0]], [p_corner1[1], p_corner2[1]],
            color=c_feedback, linestyle='--', linewidth=1.4, zorder=3)
    draw_arrow((p_corner2[0], p_corner2[1]), (p_reg_in[0], p_reg_in[1]),
               text='', color=c_feedback, lw=1.4, ls='--')

    ax.text(4.8, 0.55, 'No (Reject)', ha='left', va='center',
            fontsize=7.8, fontweight='bold', color=c_feedback, family=font_family)

    # Regularization Block in Feedback Loop
    draw_block(2.2, 1.95, 2.6, 1.15,
               'Constrained Re-decoding',
               r'$\Delta \ell_t = -\alpha (1 - S_{\rm{sem}}) \mathbb{I}(v \in \mathcal{V}_{\rm{susp}})$',
               'Logit Penalty on Ambiguous Tokens',
               fill='#FEF2F2', edge=c_feedback, lw=1.3, ls='--')

    # Arrow from Regularization Block UP into Whisper Decoder
    p_reg_out = (3.5, 3.1)
    p_whisper_feedback = (5.3, 3.8)
    
    # Path: UP to y=3.5, then RIGHT to x=5.3, then UP to (5.3, 3.8)
    ax.plot([p_reg_out[0], p_reg_out[0]], [p_reg_out[1], 3.5],
            color=c_feedback, linestyle='--', linewidth=1.4, zorder=3)
    ax.plot([p_reg_out[0], p_whisper_feedback[0]], [3.5, 3.5],
            color=c_feedback, linestyle='--', linewidth=1.4, zorder=3)
    draw_arrow((p_whisper_feedback[0], 3.5), (p_whisper_feedback[0], p_whisper_feedback[1]),
               text=r"Penalized Logits $\ell_t'$", text_pos='right', color=c_feedback, lw=1.4, ls='--')

    # =========================================================================
    # 7. METRIC ANNOTATION FOOTNOTE (ACADEMIC BENCHMARK LINE)
    # =========================================================================
    summary_text = (
        "Parameter Budget: Whisper-Turbo LoRA (845.2M) + BGE-M3 (568.0M) + XGBoost (0.15M) = 1.377B (< 1.5B)  |  "
        "Efficiency: RTF = 0.0415 (24.1x Real-time)  |  Peak VRAM: 4.5 GB FP16"
    )
    ax.text(5.4, 0.1, summary_text, ha='center', va='center',
            fontsize=7.8, fontweight='bold', color='#334155', family=font_family,
            bbox=dict(boxstyle='square,pad=0.25', facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.8))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)
    print(f"[SUCCESS] Generated IEEE Standard System Architecture Diagram: {output_path}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    target_png = project_root / "docs" / "figures" / "pipeline_architecture.png"
    build_ieee_architecture_diagram(str(target_png))
