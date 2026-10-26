"""Draw the one-page pynasonde 2.0 overview (old vs new) for the SoftwareX paper.

Run:  python make_overview_figure.py     -> pynasonde_v2_overview.{pdf,svg,png}

Everything in a box was checked against the git tags (v1.0.0 ... v1.3.0, main):
the top row is what shipped before 2.0, the bottom row is what 2.0 adds.
The script fails if any text would run outside its box.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.family": "DejaVu Sans"})

OLD_FILL, OLD_EDGE = "#DCE8F8", "#4A6FA5"
NEW_FILL, NEW_EDGE = "#D5F5D5", "#2E8B3E"

COLUMNS = [
    {
        "title": "VIPIR",
        "old": ["RIQ and NGI file readers", "Echo extraction", "Multi-stage ionogram filter",
                "NGI trace auto-scaling"],
        "new": ["Polarization + scaler", "O/X mode separation", "foF2 · h′F · MUF", "Documented and tested"],
    },
    {
        "title": "Digisonde",
        "old": ["SAO · RSF · SBF · DFT · DVL", "SKY · EDP · MMM · raw IQ", "Directogram, skymap plots",
                "+ multi-record SAO (1.2)", "+ GRM and CADI (1.3)"],
        "new": ["Autoscaler", "RFI filters + Viterbi", "foE · foF2 · hmF2", "for RSF, SBF and MMM"],
    },
    {
        "title": "Remote databases",
        "old": ["DIDBase (GIRO) connector", "SAO files, DataFrames,", "scaled parameters (1.3)"],
        "new": ["DriftBase connector", "Drift and SKY files,", "velocities", "Credentials via .env"],
    },
]

fig, ax = plt.subplots(figsize=(8.0, 4.7))
LEFT, COLW, GAP = 12.0, 28.0, 2.0
Y_TITLE, Y_FOOT = 61.0, 0.0
H_OLD, Y_OLD = 24.0, 30.5
H_NEW, Y_NEW = 19.0, 9.0
FOOT_H = 5.6
ax.set_xlim(0, 101.5); ax.set_ylim(-1.2, 64); ax.axis("off")
checks = []   # (text artist, box right edge, box left edge)


def box(x, y, w, h, fill, edge, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fill, ec=edge, lw=lw))


for i, col in enumerate(COLUMNS):
    x = LEFT + i * (COLW + GAP)
    ax.text(x + COLW / 2, Y_TITLE, col["title"], ha="center", va="center", fontsize=11.5, fontweight="bold")
    box(x, Y_OLD, COLW, H_OLD, OLD_FILL, OLD_EDGE)
    box(x, Y_NEW, COLW, H_NEW, NEW_FILL, NEW_EDGE, lw=1.8)
    for k, line in enumerate(col["old"]):
        t = ax.text(x + 1.4, Y_OLD + H_OLD - 3.4 - k * 4.1, line, ha="left", va="center", fontsize=8.2)
        checks.append((t, x, x + COLW))
    for k, line in enumerate(col["new"]):
        t = ax.text(x + 1.4, Y_NEW + H_NEW - 3.4 - k * 4.1, line, ha="left", va="center", fontsize=8.4,
                    fontweight="bold" if k == 0 else "normal")
        checks.append((t, x, x + COLW))

ax.text(LEFT - 1.8, Y_OLD + H_OLD / 2, "Before\n(v1.0–1.3)", ha="right", va="center", fontsize=10,
        color=OLD_EDGE, fontweight="bold")
ax.text(LEFT - 1.8, Y_NEW + H_NEW / 2, "New in\nv2.0", ha="right", va="center", fontsize=10,
        color=NEW_EDGE, fontweight="bold")

foot_w = 3 * COLW + 2 * GAP
box(LEFT, Y_FOOT, foot_w, FOOT_H, NEW_FILL, NEW_EDGE, lw=1.8)
t = ax.text(LEFT + foot_w / 2, Y_FOOT + FOOT_H / 2,
            'Also new: pip install "pynasonde[vipir]"  or  "[digisonde]"   ·   versioned docs per release',
            ha="center", va="center", fontsize=8.2)
checks.append((t, LEFT, LEFT + foot_w))

# ---- guard: no text may run outside its box ----
fig.canvas.draw()
inv = ax.transData.inverted()
problems = []
for text, left, right in checks:
    bb = text.get_window_extent(fig.canvas.get_renderer())
    (x0, _), (x1, _) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    if x0 < left + 0.4 or x1 > right - 0.4:
        problems.append(f"{text.get_text()!r}: spans {x0:.1f}-{x1:.1f}, box {left:.1f}-{right:.1f}")
if problems:
    raise SystemExit("text outside its box:\n  " + "\n  ".join(problems))

here = Path(__file__).resolve().parent
for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
    fig.savefig(here / f"pynasonde_v2_overview.{ext}", bbox_inches="tight", pad_inches=0.06, **kw)
print("wrote", ", ".join(f"pynasonde_v2_overview.{e}" for e in ("pdf", "svg", "png")), "| text fits in every box")
