#!/usr/bin/env python3
"""
Draw one sample from rir-uniform-angle-fullroom-edge-y0-band,
marking the coordinate origin and axes so the convention is explicit.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyArrow
import numpy as np

NPZ  = Path("/storage/reie/data/rec-rir/rir-uniform-angle-fullroom-edge-y0-band/rir_cfg.npz")
OUT  = Path("results/ace_explore/training_sample_origin.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Load a representative sample ─────────────────────────────────────────────
data = np.load(NPZ, allow_pickle=True)
pars = data["rir_pars"]

# Find a sample whose angle is between 120-160° for a clear non-trivial geometry
chosen = None
for p in pars[:2000]:
    room = p["room_sz"]
    mic  = p["pos_rcv"][0]   # (x, y, z)
    src  = p["pos_src"][0]   # (x, y, z)
    dx   = src[0] - mic[0]
    dy   = src[1] - mic[1]
    ang  = math.degrees(math.atan2(dy, dx)) % 360
    if ang > 180: ang = 360 - ang
    if 130 <= ang <= 160 and 0.4 <= mic[1] <= 0.8 and room[0] < 8 and room[1] < 8:
        chosen = (p, ang)
        break

if chosen is None:
    chosen = (pars[0], 0)   # fallback

p, gt_angle = chosen
room_sz = p["room_sz"]  # [Lx, Ly, Lz]
mic     = p["pos_rcv"][0]
src     = p["pos_src"][0]
Lx, Ly  = room_sz[0], room_sz[1]
mx, my  = mic[0], mic[1]
sx, sy  = src[0], src[1]
radius  = math.hypot(sx - mx, sy - my)

print(f"Sample:   index={p['index']}")
print(f"room_sz:  ({Lx:.2f}, {Ly:.2f}, {room_sz[2]:.2f}) m")
print(f"pos_rcv:  ({mx:.2f}, {my:.2f}, {mic[2]:.2f}) m")
print(f"pos_src:  ({sx:.2f}, {sy:.2f}, {src[2]:.2f}) m")
print(f"angle:    {gt_angle:.1f}°")
print(f"radius:   {radius:.2f} m")
print(f"wall_dist:{my:.2f} m (y=0)")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#12122a")
for sp in ax.spines.values():
    sp.set_edgecolor("#334466")
ax.tick_params(colors="#aaaacc", labelsize=9)
ax.xaxis.label.set_color("#aaaacc")
ax.yaxis.label.set_color("#aaaacc")

margin = max(Lx, Ly) * 0.12
ax.set_xlim(-margin, Lx + margin)
ax.set_ylim(-margin, Ly + margin)
ax.set_aspect("equal")
ax.set_xlabel("x  (m) — along the y=0 wall →", fontsize=11)
ax.set_ylabel("y  (m) — inward from y=0 wall →", fontsize=11)

# ── Room walls ───────────────────────────────────────────────────────────────
ax.add_patch(Rectangle((0, 0), Lx, Ly,
    fill=False, edgecolor="#6677aa", linewidth=2.5, zorder=3))

# ── y=0 "near wall" highlight ─────────────────────────────────────────────────
ax.add_patch(Rectangle((0, 0), Lx, 0.04,
    color="#ff5555", zorder=4, clip_on=False))
ax.text(Lx / 2, -margin * 0.45,
    'y = 0  wall  (the "near wall")',
    color="#ff5555", fontsize=10, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#ff5555", alpha=0.8))

# ── Training band shading on y-axis ──────────────────────────────────────────
ax.add_patch(Rectangle((0, 0.4), Lx, 0.4,
    color="#3355aa", alpha=0.20, zorder=1))
ax.axhline(0.4, color="#4466cc", lw=1.2, ls="--", alpha=0.7)
ax.axhline(0.8, color="#4466cc", lw=1.2, ls="--", alpha=0.7)
ax.text(Lx + margin * 0.08, 0.60,
    "mic-to-wall\ntraining band\n[0.4, 0.8] m",
    color="#7788bb", fontsize=8, va="center", ha="left")

# ── Origin marker ─────────────────────────────────────────────────────────────
ARROW_LEN = max(Lx, Ly) * 0.13
ax.annotate("", xy=(ARROW_LEN, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="-|>", color="#ff9922",
                    lw=2.2, mutation_scale=18), zorder=10)
ax.annotate("", xy=(0, ARROW_LEN), xytext=(0, 0),
    arrowprops=dict(arrowstyle="-|>", color="#22cc88",
                    lw=2.2, mutation_scale=18), zorder=10)
ax.plot(0, 0, "o", color="white", ms=8, zorder=11)
ax.text(ARROW_LEN * 0.55, -margin * 0.30,
    "x", color="#ff9922", fontsize=14, fontweight="bold", ha="center")
ax.text(-margin * 0.25, ARROW_LEN * 0.55,
    "y", color="#22cc88", fontsize=14, fontweight="bold", va="center")
ax.text(-margin * 0.20, -margin * 0.25,
    "origin\n(0, 0)",
    color="white", fontsize=8.5, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.25", fc="#1a1a2e", ec="white", alpha=0.85))

# ── Mic-to-wall distance line ─────────────────────────────────────────────────
ax.annotate("", xy=(mx, 0), xytext=(mx, my),
    arrowprops=dict(arrowstyle="<->", color="#22ddaa", lw=1.6))
ax.text(mx + Lx * 0.016, my / 2,
    f"mic_y = {my:.2f} m\n(dist to y=0)",
    color="#22ddaa", fontsize=8.5, va="center")

# ── Microphone ────────────────────────────────────────────────────────────────
ax.plot(mx, my, "D", color="#22ddaa", ms=11, zorder=7,
    markeredgecolor="#aaffdd", markeredgewidth=0.8)
ax.text(mx + Lx * 0.02, my + Ly * 0.025,
    f"mic\n({mx:.2f}, {my:.2f})",
    color="#22ddaa", fontsize=8.5)

# ── Source ────────────────────────────────────────────────────────────────────
ax.plot(sx, sy, "*", color="#ff9900", ms=18, zorder=7,
    markeredgecolor="#ffcc66", markeredgewidth=0.8)
ax.text(sx + Lx * 0.02, sy + Ly * 0.02,
    f"src\n({sx:.2f}, {sy:.2f})",
    color="#ff9900", fontsize=8.5)

# ── Arrow mic → src ───────────────────────────────────────────────────────────
ax.annotate("", xy=(sx, sy), xytext=(mx, my),
    arrowprops=dict(arrowstyle="-|>", color="#ffcc33",
                    lw=2.0, mutation_scale=20), zorder=6)

# ── Angle arc ─────────────────────────────────────────────────────────────────
arc_r = radius * 0.38
# Arc from +x direction (0°) to angle
theta_rad = math.atan2(sy - my, sx - mx)
thetas = np.linspace(0.0, theta_rad, 80)
ax.plot(mx + arc_r * np.cos(thetas),
        my + arc_r * np.sin(thetas),
        color="#ffcc33", lw=1.8, alpha=0.9, zorder=5)
# 0° reference arrow (along +x)
ax.annotate("", xy=(mx + arc_r * 1.15, my), xytext=(mx, my),
    arrowprops=dict(arrowstyle="-", color="#ffcc3366",
                    lw=1.2, linestyle="dashed"), zorder=5)
ax.text(mx + arc_r * 1.25, my - Ly * 0.025,
    "0°", color="#ffcc3399", fontsize=8, ha="center")
# Angle label
mid_t = theta_rad / 2
ax.text(mx + arc_r * 1.30 * math.cos(mid_t),
        my + arc_r * 1.30 * math.sin(mid_t),
        f"{gt_angle:.1f}°", color="#ffe066", fontsize=10,
        fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="#12122aaa", ec="none"))

# Radius label
mid_x = (mx + sx) / 2
mid_y = (my + sy) / 2
perp_dx = -(sy - my) / radius * 0.12
perp_dy =  (sx - mx) / radius * 0.12
ax.text(mid_x + perp_dx, mid_y + perp_dy,
    f"r = {radius:.2f} m",
    color="#ffdd88", fontsize=8.5, ha="center",
    bbox=dict(boxstyle="round,pad=0.2", fc="#22223355", ec="none"))

# ── Room dimension labels ─────────────────────────────────────────────────────
ax.annotate("", xy=(Lx, -margin * 0.55), xytext=(0, -margin * 0.55),
    arrowprops=dict(arrowstyle="<->", color="#aaaacc", lw=1.2))
ax.text(Lx / 2, -margin * 0.70, f"Lx = {Lx:.2f} m",
    color="#aaaacc", fontsize=9, ha="center")
ax.annotate("", xy=(Lx + margin * 0.55, Ly), xytext=(Lx + margin * 0.55, 0),
    arrowprops=dict(arrowstyle="<->", color="#aaaacc", lw=1.2))
ax.text(Lx + margin * 0.72, Ly / 2, f"Ly = {Ly:.2f} m",
    color="#aaaacc", fontsize=9, ha="left", va="center")

# ── Convention box ────────────────────────────────────────────────────────────
box_txt = (
    "Training coordinate convention\n"
    "────────────────────────────────\n"
    "  origin = (0, 0, 0) at room corner\n"
    "  x-axis = along the y=0 wall\n"
    "  y-axis = inward from y=0 wall\n"
    "  z-axis = up from floor\n"
    "  mic placed near y=0 (band 0.4–0.8 m)\n"
    "  angle  = atan2(Δy, Δx)\n"
    "           0° = +x, 90° = +y (inward)"
)
ax.text(0.98, 0.99, box_txt,
    transform=ax.transAxes, color="#ccccee", fontsize=8,
    va="top", ha="right", family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", fc="#1e1e3a", ec="#334466", alpha=0.92))

ax.set_title(
    f"Training dataset sample  —  rir-uniform-angle-fullroom-edge-y0-band\n"
    f"room ({Lx:.2f} × {Ly:.2f} × {room_sz[2]:.2f} m)   "
    f"mic_y = {my:.2f} m   angle = {gt_angle:.1f}°   r = {radius:.2f} m",
    color="white", fontsize=11, pad=10)

plt.tight_layout()
fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
