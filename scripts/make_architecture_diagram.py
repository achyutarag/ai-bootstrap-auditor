"""
Draws the paper's architecture / workflow diagram (paper/figures/fig_architecture.pdf, vector,
plus a PNG preview). Sized at its printed width (IEEE two-column \textwidth = 7.16 in) so the
7-pt labels print at 7 pt.

    python scripts/make_architecture_diagram.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")

INK, MUTED, LINE = "#1f2328", "#57606a", "#8c959f"
FILL = {"input": "#f6f8fa", "engine": "#ddf4ff", "batch": "#ddf4ff", "decision": "#fff8c5",
        "audit": "#fbefff", "out": "#f6f8fa"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})

W, H = 7.16, 1.95                       # inches, printed size
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def block(x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=FILL[kind], ec=LINE, lw=0.7))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=7.4,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.33, "\n".join(lines), ha="center", va="top", fontsize=6.6,
            color=MUTED, linespacing=1.35)
    return (x, y, w, h)


def arrow(p, q, label=None, dashed=False, label_dy=0.07):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK,
                                 linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + label_dy, label, ha="center", va="bottom",
                fontsize=6.2, color=MUTED, style="italic")


top_y, top_h = 0.86, 1.0
mid = top_y + top_h / 2
inp = block(0.05, top_y, 1.32, top_h, "Paired per-item logs",
            ["same items, old vs. new", "CSV, arrays or upload", "e.g. lm-eval logs"],
            "input")
eng = block(1.65, top_y, 1.55, top_h, "Statistical engine",
            ["paired bootstrap interval", "exact sign (McNemar) test", "non-inferiority guardrails",
             "sample floor + warnings"], "engine")
bat = block(3.48, top_y, 1.40, top_h, "Batch aggregator",
            ["Bonferroni / Benjamini-", "Hochberg at α/2", "NO-GO never demoted",
             "retest tracker"], "batch")
dec = block(5.16, top_y, 1.95, top_h, "Release decision",
            ["per metric: GO / REVIEW / NO-GO", "guardrail: PASS / INCONCL. / FAIL",
             "overall: GO / HOLD / NO-GO", "→ web app · Python API · CI gate"], "decision")

arrow((inp[0] + inp[2], mid), (eng[0], mid))
arrow((eng[0] + eng[2], mid), (bat[0], mid), "per-metric")
arrow((bat[0] + bat[2], mid), (dec[0], mid), "corrected")

aud = block(1.65, 0.05, 3.23, 0.66, "Self-audit harness",
            ["calibration (A/A tests, coverage) · metamorphic relations",
             "differential checks vs. exact and closed-form results"],
            "audit")
for x in (eng[0] + eng[2] / 2, bat[0] + bat[2] / 2):
    arrow((x, aud[1] + aud[3]), (x, top_y), dashed=True)
ax.text(aud[0] + aud[2] + 0.08, aud[1] + aud[3] / 2,
        "65 CI tests ·\nlive Validation Suite ·\nreproduce scripts", ha="left", va="center",
        fontsize=6.4, color=MUTED, linespacing=1.3)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_architecture.pdf"))
fig.savefig(os.path.join(OUT, "fig_architecture_preview.png"), dpi=220)
print("wrote", os.path.join(OUT, "fig_architecture.pdf"))
