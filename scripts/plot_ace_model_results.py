#!/usr/bin/env python3
"""Quick summary plot of model predictions from results/ace_explore/model_eval/per_entry_preds.csv."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = Path("results/ace_explore/model_eval/per_entry_preds.csv")
OUT = Path("results/ace_explore/model_eval/predictions_summary.png")

GT_ANGLE  = 151.5
GT_RADIUS = 1.195

LABELS = {
    "ace_rir.wav":                    "ACE (real)",
    "synthetic_rir_rir_generator.wav": "rir-generator",
    "synthetic_rir_gpurir.wav":        "gpuRIR",
}
COLORS = {"ACE (real)": "#22ddaa", "rir-generator": "#ffaa33", "gpuRIR": "#cc77ff"}

rows = list(csv.DictReader(CSV.open()))

data = {}
for row in rows:
    name = Path(row["rir_path"]).name
    lbl  = LABELS.get(name, name)
    data.setdefault(lbl, {"angles": [], "radii": [], "aerrs": [], "rerrs": []})
    data[lbl]["angles"].append(float(row["pred_angle_deg"]))
    data[lbl]["radii"].append(float(row["pred_radius_m"]))
    data[lbl]["aerrs"].append(float(row["angle_error_deg"]))
    data[lbl]["rerrs"].append(float(row["radius_error_m"]))

order = ["ACE (real)", "rir-generator", "gpuRIR"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#12122a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334466")
    ax.tick_params(colors="#888899", labelsize=9)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.title.set_color("white")

# ── Left: angle predictions scatter ─────────────────────────────────────────
ax = axes[0]
for i, lbl in enumerate(order):
    d = data[lbl]
    x = np.full(len(d["angles"]), i) + np.random.uniform(-0.15, 0.15, len(d["angles"]))
    ax.scatter(x, d["angles"], color=COLORS[lbl], s=40, alpha=0.75, zorder=3)
    ax.plot([i - 0.3, i + 0.3], [np.mean(d["angles"])] * 2,
            color=COLORS[lbl], lw=2.5, zorder=4)
ax.axhline(GT_ANGLE, color="#ff5555", lw=1.8, ls="--", label=f"GT = {GT_ANGLE}°", zorder=2)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, color="white", fontsize=9)
ax.set_ylabel("Predicted angle (°)", color="#aaaacc")
ax.set_title("Angle predictions (10 clips each)", color="white", fontsize=10)
ax.legend(framealpha=0.3, labelcolor="white", fontsize=9, facecolor="#22223366")
ax.set_ylim(0, 185)

# ── Middle: radius predictions scatter ───────────────────────────────────────
ax = axes[1]
for i, lbl in enumerate(order):
    d = data[lbl]
    x = np.full(len(d["radii"]), i) + np.random.uniform(-0.15, 0.15, len(d["radii"]))
    ax.scatter(x, d["radii"], color=COLORS[lbl], s=40, alpha=0.75, zorder=3)
    ax.plot([i - 0.3, i + 0.3], [np.mean(d["radii"])] * 2,
            color=COLORS[lbl], lw=2.5, zorder=4)
ax.axhline(GT_RADIUS, color="#ff5555", lw=1.8, ls="--", label=f"GT = {GT_RADIUS} m", zorder=2)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, color="white", fontsize=9)
ax.set_ylabel("Predicted radius (m)", color="#aaaacc")
ax.set_title("Radius predictions (10 clips each)", color="white", fontsize=10)
ax.legend(framealpha=0.3, labelcolor="white", fontsize=9, facecolor="#22223366")

# ── Right: MAE bars ──────────────────────────────────────────────────────────
ax = axes[2]
ax2 = ax.twinx()
w = 0.3
x = np.arange(len(order))
bar_a = ax.bar(x - w/2, [np.mean(data[l]["aerrs"]) for l in order], width=w,
               color=[COLORS[l] for l in order], alpha=0.85, label="Angle MAE (°)")
bar_r = ax2.bar(x + w/2, [np.mean(data[l]["rerrs"]) for l in order], width=w,
                color=[COLORS[l] for l in order], alpha=0.45, label="Radius MAE (m)", hatch="//")
ax.set_xticks(x)
ax.set_xticklabels(order, color="white", fontsize=9)
ax.set_ylabel("Angle MAE (°)", color="#aaaacc")
ax2.set_ylabel("Radius MAE (m)", color="#aaaacc")
ax2.tick_params(colors="#888899", labelsize=9)
ax.set_title("Mean absolute error per RIR type", color="white", fontsize=10)
for bar, lbl in zip(bar_a, order):
    v = np.mean(data[lbl]["aerrs"])
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}°",
            color="white", ha="center", va="bottom", fontsize=8)
for bar, lbl in zip(bar_r, order):
    v = np.mean(data[lbl]["rerrs"])
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}m",
             color="#ccccee", ha="center", va="bottom", fontsize=8)

fig.suptitle(
    "Model inference on ACE Office 2 / pos 1  —  GT angle=151.5°, radius=1.195m\n"
    "checkpoint: fullroom-edge-y0-film-ctf-backbone/best.tar",
    color="white", fontsize=11, y=1.01
)
plt.tight_layout()
fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"Saved {OUT}")

# Print summary table
print(f"\n{'RIR':<20} {'Angle MAE':>10} {'Mean pred °':>12} {'Radius MAE':>11} {'Mean pred m':>12}")
print("-" * 70)
for lbl in order:
    d = data[lbl]
    print(f"{lbl:<20} {np.mean(d['aerrs']):>10.1f}° {np.mean(d['angles']):>11.1f}°  "
          f"{np.mean(d['rerrs']):>10.2f}m {np.mean(d['radii']):>11.2f}m")
print(f"\nGround truth:          angle={GT_ANGLE}°,  radius={GT_RADIUS:.3f}m")
