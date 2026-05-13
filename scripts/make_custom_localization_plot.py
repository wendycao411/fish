import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("results_figures")
OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "localization_controls_f1.png"

cats = ["Random heatmap", "Shuffled heatmap", "True heatmap", "Oracle ROI"]
vals = [0.42, 0.49, 0.75, 0.89]

# Colors: muted for lower, highlight true and oracle
colors = ["#7f7f7f", "#9e9e9e", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(7,5))
bars = ax.bar(cats, vals, color=colors, edgecolor="none")

# Highlight True heatmap (index 2) with thicker edge
bars[2].set_edgecolor("#000000")
bars[2].set_linewidth(2.0)

# Highlight Oracle ROI (index 3) with hatch and darker edge
bars[3].set_edgecolor("#000000")
bars[3].set_linewidth(2.0)
bars[3].set_hatch("///")

# Dashed visual-only baseline
baseline = 0.39
ax.axhline(baseline, color="black", linestyle="--", linewidth=1.5)
ax.text(0.02, baseline + 0.01, "Visual-only baseline (0.39)", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=12)

# Labels and title
ax.set_ylabel("F1 score")
ax.set_title("Spatially correct audio cues matter")
ax.set_ylim(0, 1.0)

# Value labels above bars
for rect, v in zip(bars, vals):
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, h + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=200, bbox_inches='tight')
plt.close()
print(f"Wrote: {OUT_FILE}")
