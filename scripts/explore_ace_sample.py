#!/usr/bin/env python3
"""
explore_ace_sample.py

Selects one sample from the ACE corpus (8-channel linear array, Office 2,
source position 1) — the room whose microphone center sits closest to the
0.4–0.8 m wall-band used in the rir-uniform-angle-fullroom-edge-y0-band
training distribution — then produces an annotated figure showing:

  1. Bird's-eye floor plan with the room, mic-array, source, wall-band, and
     the geometry arrow (angle / radius).
  2. A FiLM metadata panel with raw and normalised values for every
     conditioning input the model sees at inference time.

All room coordinates come from Table 1 of:
  J. Eaton et al., "Estimation of room acoustic parameters: The ACE
  Challenge", IEEE/ACM TASLP 24(10), 2016. arXiv: 1606.03365

The coordinate convention used by the ACE paper is (L, W, H) where
  L = room length (x-axis)
  W = room width  (y-axis)
  H = room height (z-axis)
and all positions are given as (x, y, z).

Usage
-----
  cd /storage/reie/REC-RIR-LOCALIZATION
  /storage/reie/.venv_custom_ssh/bin/python scripts/explore_ace_sample.py

Output: results/ace_explore/geometry.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# ACE corpus room geometry (from arXiv 1606.03365)
# lin8ch stores the position of Ch 1 (element 0, closest to x=0 wall) of the
# 8-channel linear array, taken from Table 1 of the ACE paper which has a
# dedicated "Ch 1 of 8-ch. linear" column with precise per-channel positions.
# Note: Table 2 of the same paper lists a different reference element (ch4)
# under the heading "8-ch Linear array".
# ---------------------------------------------------------------------------
# Entry format:
#   room_name : dict with
#     dims     : (L, W, H)  room dimensions in metres
#     src      : (x, y, z)  loudspeaker position
#     lin8ch   : (x, y, z)  Ch 1 (element 0) position of Lin8Ch array (Table 1)
#     t60      : float       mean fullband T60 (s), from the corpus CSV
#     rir_path : str         path to the RIR WAV
#     config   : int         source-position config index (1 or 2)

ACE_ROOMS = {
    "Office_1_c1": dict(
        label="Office 1 (pos 1)", room="Office_1", config=1,
        dims=(3.32, 4.83, 2.95),
        src=(2.06, 1.04, 1.19),
        lin8ch=(1.92, 2.14, 1.19),
        t60=0.352,
        rir_path="/storage/data/ace/Lin8Ch/Office_1/1/Lin8Ch_502_1_RIR.wav",
    ),
    "Office_1_c2": dict(
        label="Office 1 (pos 2)", room="Office_1", config=2,
        dims=(3.32, 4.83, 2.95),
        src=(2.06, 1.04, 1.19),
        lin8ch=(1.79, 3.67, 1.19),
        t60=0.307,
        rir_path="/storage/data/ace/Lin8Ch/Office_1/2/Lin8Ch_502_2_RIR.wav",
    ),
    "Office_2_c1": dict(
        label="Office 2 (pos 1)", room="Office_2", config=1,
        dims=(3.22, 5.10, 2.94),
        src=(1.41, 1.73, 1.19),
        lin8ch=(0.60, 2.78, 1.19),   # Table 1: Ch 1 of 8-ch linear at x=0.60 m
        t60=0.392,
        rir_path="/storage/data/ace/Lin8Ch/Office_2/1/Lin8Ch_803_1_RIR.wav",
    ),
    "Office_2_c2": dict(
        label="Office 2 (pos 2)", room="Office_2", config=2,
        dims=(3.22, 5.10, 2.94),
        src=(1.41, 1.73, 1.19),
        lin8ch=(1.58, 4.16, 1.19),
        t60=0.380,
        rir_path="/storage/data/ace/Lin8Ch/Office_2/2/Lin8Ch_803_2_RIR.wav",
    ),
    "Meeting_Room_1_c1": dict(
        label="Meeting Rm 1 (pos 1)", room="Meeting_Room_1", config=1,
        dims=(6.61, 5.11, 2.95),
        src=(1.39, 1.26, 1.19),
        lin8ch=(2.74, 1.14, 1.19),
        t60=0.465,
        rir_path="/storage/data/ace/Lin8Ch/Meeting_Room_1/1/Lin8Ch_503_1_RIR.wav",
    ),
    "Meeting_Room_1_c2": dict(
        label="Meeting Rm 1 (pos 2)", room="Meeting_Room_1", config=2,
        dims=(6.61, 5.11, 2.95),
        src=(1.39, 1.26, 1.19),
        lin8ch=(3.96, 1.14, 1.19),
        t60=0.474,
        rir_path="/storage/data/ace/Lin8Ch/Meeting_Room_1/2/Lin8Ch_503_2_RIR.wav",
    ),
    "Meeting_Room_2_c1": dict(
        label="Meeting Rm 2 (pos 1)", room="Meeting_Room_2", config=1,
        dims=(10.30, 9.07, 2.63),
        src=(4.65, 4.07, 1.19),
        lin8ch=(3.00, 3.59, 1.19),
        t60=0.375,
        rir_path="/storage/data/ace/Lin8Ch/Meeting_Room_2/1/Lin8Ch_611_1_RIR.wav",
    ),
    "Meeting_Room_2_c2": dict(
        label="Meeting Rm 2 (pos 2)", room="Meeting_Room_2", config=2,
        dims=(10.30, 9.07, 2.63),
        src=(4.65, 4.07, 1.19),
        lin8ch=(2.00, 3.59, 1.19),
        t60=0.373,
        rir_path="/storage/data/ace/Lin8Ch/Meeting_Room_2/2/Lin8Ch_611_2_RIR.wav",
    ),
    "Lecture_Room_1_c1": dict(
        label="Lecture Rm 1 (pos 1)", room="Lecture_Room_1", config=1,
        dims=(6.93, 9.73, 3.00),
        src=(3.65, 3.73, 1.19),
        lin8ch=(2.89, 3.04, 1.19),
        t60=0.645,
        rir_path="/storage/data/ace/Lin8Ch/Lecture_Room_1/1/Lin8Ch_508_1_RIR.wav",
    ),
    "Lecture_Room_1_c2": dict(
        label="Lecture Rm 1 (pos 2)", room="Lecture_Room_1", config=2,
        dims=(6.93, 9.73, 3.00),
        src=(3.65, 3.73, 1.19),
        lin8ch=(1.07, 3.12, 1.19),
        t60=0.628,
        rir_path="/storage/data/ace/Lin8Ch/Lecture_Room_1/2/Lin8Ch_508_2_RIR.wav",
    ),
    "Building_Lobby_c1": dict(
        label="Lobby (pos 1)", room="Building_Lobby", config=1,
        dims=(4.47, 5.13, 3.18),
        src=(1.98, 0.61, 1.19),
        lin8ch=(1.95, 2.03, 1.19),
        t60=0.728,
        rir_path="/storage/data/ace/Lin8Ch/Building_Lobby/1/Lin8Ch_EE_lobby_1_RIR.wav",
    ),
    "Building_Lobby_c2": dict(
        label="Lobby (pos 2)", room="Building_Lobby", config=2,
        dims=(4.47, 5.13, 3.18),
        src=(1.98, 0.61, 1.19),
        lin8ch=(1.86, 3.54, 1.19),
        t60=0.742,
        rir_path="/storage/data/ace/Lin8Ch/Building_Lobby/2/Lin8Ch_EE_lobby_2_RIR.wav",
    ),
}

# Training distribution limits (for the ✓ / ⚠ flags)
_TRAIN = dict(
    room_xy=(3.0, 15.0),
    room_z=(2.5, 6.0),
    rt60=(0.2, 1.5),
    angle=(0.0, 180.0),
    mic_wall=(0.4, 0.8),
)
# FiLM normalisation constants (must match dataset_RecRIR.py)
_RT60_MIN, _RT60_MAX = 0.1, 1.5
_ROOM_SZ_MAX         = 15.0
_LOG_VOL_MIN         = math.log(10.0)
_LOG_VOL_MAX         = math.log(1000.0)
_EPS                 = 1e-8

# ---------------------------------------------------------------------------
# Helper: compute geometry for one room entry
# ---------------------------------------------------------------------------

def _nearest_wall(x: float, y: float, L: float, W: float):
    """Return (wall_label, min_dist) for the nearest horizontal wall."""
    dists = {"x=0": x, f"x={L:.2f}": L - x, "y=0": y, f"y={W:.2f}": W - y}
    nw = min(dists, key=dists.__getitem__)
    return nw, dists[nw]


def compute_geometry(room: dict) -> dict:
    """
    Compute all FiLM-relevant geometry for one ACE room entry.

    Coordinate rotation to training frame
    --------------------------------------
    Training convention: mic near the y=0 wall.
    We identify which wall is nearest and rotate so that the
    inward direction always becomes the training y-axis:

        nearest wall   | Lx_train   | Ly_train   | mx_train | my_train
        ───────────────┼────────────┼────────────┼──────────┼─────────
        x=0  (depth0)  | W          | L          | y_mic    | x_mic
        x=L  (depthMax)| W          | L          | y_mic    | L-x_mic
        y=0  (width0)  | L          | W          | x_mic    | y_mic   [identity]
        y=W  (widthMax)| L          | W          | x_mic    | W-y_mic
    """
    L, W, H = room["dims"]
    sx, sy, sz = room["src"]
    mx, my, mz = room["lin8ch"]

    nw, wall_dist = _nearest_wall(mx, my, L, W)

    # Rotate to training frame
    if nw == "x=0":
        Lx, Ly = W, L
        tx, ty = my, mx          # along-wall, inward
        tsx, tsy = sy, sx
    elif nw.startswith("x="):    # x=L wall
        Lx, Ly = W, L
        tx, ty = my, L - mx
        tsx, tsy = sy, L - sx
    elif nw == "y=0":            # identity
        Lx, Ly = L, W
        tx, ty = mx, my
        tsx, tsy = sx, sy
    else:                        # y=W wall
        Lx, Ly = L, W
        tx, ty = mx, W - my
        tsx, tsy = sx, W - sy

    # Angle and radius in training frame
    dx = tsx - tx
    dy = tsy - ty
    radius = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    if angle_deg > 180.0:
        angle_deg = 360.0 - angle_deg  # fold to [0, 180]

    # FiLM normalisation
    rt60        = room["t60"]
    rt60_norm   = float(np.clip((rt60 - _RT60_MIN) / (_RT60_MAX - _RT60_MIN), 0, 1))
    room_sz     = np.array([Lx, Ly, H])
    room_sz_n   = np.clip(room_sz / _ROOM_SZ_MAX, 0, 1)
    mic_center  = np.array([tx, ty, mz])
    mic_norm    = np.clip(mic_center / (room_sz + _EPS), 0, 1)
    vol         = L * W * H
    vol_norm    = float(np.clip(
        (math.log(max(vol, _EPS)) - _LOG_VOL_MIN) / (_LOG_VOL_MAX - _LOG_VOL_MIN), 0, 1))

    film_vec = np.array([rt60_norm, *room_sz_n, *mic_norm, vol_norm], dtype=np.float32)

    return dict(
        L=L, W=W, H=H,
        mx=mx, my=my, mz=mz,
        sx=sx, sy=sy,
        nearest_wall=nw, wall_dist=wall_dist,
        # training frame
        Lx_t=Lx, Ly_t=Ly,
        tx=tx, ty=ty,
        tsx=tsx, tsy=tsy,
        angle_deg=angle_deg,
        radius=radius,
        rt60=rt60, rt60_norm=rt60_norm,
        room_sz=room_sz, room_sz_n=room_sz_n,
        mic_center=mic_center, mic_norm=mic_norm,
        vol=vol, vol_norm=vol_norm,
        film_vec=film_vec,
    )


# ---------------------------------------------------------------------------
# Sample selection: pick the room with wall_dist closest to [0.4, 0.8] band
# ---------------------------------------------------------------------------

def select_sample() -> tuple[str, dict, dict]:
    """Return (key, room_dict, geom_dict) for the best-matching sample."""
    best_key, best_room, best_geom, best_score = None, None, None, float("inf")
    for key, room in ACE_ROOMS.items():
        g = compute_geometry(room)
        lo, hi = _TRAIN["mic_wall"]
        d = g["wall_dist"]
        score = 0.0 if lo <= d <= hi else min(abs(d - lo), abs(d - hi))
        if score < best_score:
            best_key, best_room, best_geom, best_score = key, room, g, score
    return best_key, best_room, best_geom


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _flag(val, lo, hi):
    return "✓" if lo <= val <= hi else "⚠"


def plot_sample(room: dict, g: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax_floor = fig.add_axes([0.03, 0.10, 0.52, 0.80])
    ax_film  = fig.add_axes([0.58, 0.04, 0.40, 0.92])

    L, W = g["L"], g["W"]
    mx, my = g["mx"], g["my"]
    sx, sy = g["sx"], g["sy"]

    # ── Floor plan ─────────────────────────────────────────────────────────
    ax_floor.set_facecolor("#12122a")
    margin = max(L, W) * 0.08
    ax_floor.set_xlim(-margin, L + margin)
    ax_floor.set_ylim(-margin, W + margin)
    ax_floor.set_aspect("equal", adjustable="box")
    ax_floor.set_xlabel("x  (m)", color="#aaaacc", fontsize=11)
    ax_floor.set_ylabel("y  (m)", color="#aaaacc", fontsize=11)
    ax_floor.tick_params(colors="#888899")
    for spine in ax_floor.spines.values():
        spine.set_edgecolor("#333355")

    # Room boundary
    ax_floor.add_patch(Rectangle((0, 0), L, W,
        fill=False, edgecolor="#6677aa", linewidth=2.5, zorder=2))

    # Training wall-band shading (nearest wall)
    lo_b, hi_b = _TRAIN["mic_wall"]
    nw = g["nearest_wall"]
    if nw == "x=0":
        ax_floor.add_patch(Rectangle((lo_b, 0), hi_b - lo_b, W,
            color="#3355aa", alpha=0.25, zorder=1))
        ax_floor.axvline(lo_b, color="#4466cc", lw=1, ls="--", alpha=0.6)
        ax_floor.axvline(hi_b, color="#4466cc", lw=1, ls="--", alpha=0.6)
    elif nw.startswith("x="):
        ax_floor.add_patch(Rectangle((L - hi_b, 0), hi_b - lo_b, W,
            color="#3355aa", alpha=0.25, zorder=1))
    elif nw == "y=0":
        ax_floor.add_patch(Rectangle((0, lo_b), L, hi_b - lo_b,
            color="#3355aa", alpha=0.25, zorder=1))
        ax_floor.axhline(lo_b, color="#4466cc", lw=1, ls="--", alpha=0.6)
        ax_floor.axhline(hi_b, color="#4466cc", lw=1, ls="--", alpha=0.6)
    else:
        ax_floor.add_patch(Rectangle((0, W - hi_b), L, hi_b - lo_b,
            color="#3355aa", alpha=0.25, zorder=1))

    # Wall-distance annotation
    nw_label = f"wall-band [0.4, 0.8] m"
    ax_floor.text(0.02, 0.02, nw_label,
        transform=ax_floor.transAxes, color="#7788bb",
        fontsize=8.5, va="bottom")

    # Source
    ax_floor.plot(sx, sy, "*", color="#ff9900", ms=18, zorder=5,
        markeredgecolor="#ffcc66", markeredgewidth=0.8, label="Source")

    # 8-element array elements (ch1 at (mx,my), ch8 at (mx+7*0.08, my))
    _spacing = 0.08
    _n_mics  = 8
    elem_xs = mx + np.arange(_n_mics) * _spacing
    ax_floor.scatter(elem_xs, [my] * _n_mics, c="#22ddaa", s=18, zorder=5,
        linewidths=0.5, edgecolors="#aaffdd")
    # Ch 1 highlighted larger
    ax_floor.plot(mx, my, "D", color="#22ddaa", ms=10, zorder=6,
        markeredgecolor="#aaffdd", markeredgewidth=0.8, label="Ch 1 of Lin8Ch (reference)")

    # Arrow  source→ mic  (direction the model uses for angle)
    ax_floor.annotate("", xy=(sx, sy), xytext=(mx, my),
        arrowprops=dict(arrowstyle="-|>", color="#ffcc33",
                        lw=2.0, mutation_scale=20), zorder=6)

    # Angle arc
    arc_r = min(g["radius"] * 0.45, min(L, W) * 0.18)
    theta_rad = math.atan2(sy - my, sx - mx)
    thetas = np.linspace(0.0, theta_rad, 60)
    ax_floor.plot(mx + arc_r * np.cos(thetas), my + arc_r * np.sin(thetas),
        color="#ffcc33", lw=1.5, alpha=0.8)
    mid_t = theta_rad / 2
    ax_floor.text(
        mx + arc_r * 1.25 * math.cos(mid_t),
        my + arc_r * 1.25 * math.sin(mid_t),
        f"{g['angle_deg']:.1f}°", color="#ffe066", fontsize=9, ha="center")

    # Radius label along the arrow
    mid_x = (mx + sx) / 2 + 0.06
    mid_y = (my + sy) / 2 + 0.06
    ax_floor.text(mid_x, mid_y, f"r = {g['radius']:.2f} m",
        color="#ffdd88", fontsize=9, ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="#22223355", ec="none"))

    # Mic wall-distance line
    if nw == "x=0":
        ax_floor.annotate("", xy=(0, my), xytext=(mx, my),
            arrowprops=dict(arrowstyle="<->", color="#22ddaa", lw=1.5))
        ax_floor.text(mx / 2, my + W * 0.015,
            f"{g['wall_dist']:.2f} m", color="#22ddaa", fontsize=8.5, ha="center")
    elif nw == "y=0":
        ax_floor.annotate("", xy=(mx, 0), xytext=(mx, my),
            arrowprops=dict(arrowstyle="<->", color="#22ddaa", lw=1.5))
        ax_floor.text(mx + L * 0.015, my / 2,
            f"{g['wall_dist']:.2f} m", color="#22ddaa", fontsize=8.5)

    ax_floor.legend(loc="upper right", framealpha=0.3,
        labelcolor="white", fontsize=9, facecolor="#22223366")

    room_label = room["label"]
    ax_floor.set_title(
        f"ACE Corpus — {room_label}\n"
        f"Selected: closest mic-to-wall dist ({g['wall_dist']:.2f} m) to training band [0.4, 0.8] m",
        color="white", fontsize=11, pad=10)

    # ── FiLM metadata panel ────────────────────────────────────────────────
    ax_film.set_facecolor("#12122a")
    ax_film.set_xlim(0, 1)
    ax_film.set_ylim(0, 1)
    ax_film.axis("off")

    # Title
    ax_film.text(0.5, 0.97, "FiLM conditioning inputs", color="white",
        ha="center", va="top", fontsize=13, fontweight="bold")
    ax_film.text(0.5, 0.925, "( 8-element room_params vector )", color="#888899",
        ha="center", va="top", fontsize=9)

    # Table rows: (description, raw_value, norm_value, in_dist_flag)
    lo_xy, hi_xy = _TRAIN["room_xy"]
    lo_z,  hi_z  = _TRAIN["room_z"]
    lo_rt, hi_rt = _TRAIN["rt60"]
    lo_ang, hi_ang = _TRAIN["angle"]
    lo_mw, hi_mw = _TRAIN["mic_wall"]
    Lx_t, Ly_t = g["Lx_t"], g["Ly_t"]

    rows = [
        # (label, raw str, norm str, flag)
        ("RT60",
         f"{g['rt60']:.3f} s",
         f"{g['rt60_norm']:.3f}",
         _flag(g["rt60"], lo_rt, hi_rt)),
        ("room_sz x  (along wall)",
         f"{Lx_t:.2f} m",
         f"{g['room_sz_n'][0]:.3f}",
         _flag(Lx_t, lo_xy, hi_xy)),
        ("room_sz y  (inward depth)",
         f"{Ly_t:.2f} m",
         f"{g['room_sz_n'][1]:.3f}",
         _flag(Ly_t, lo_xy, hi_xy)),
        ("room_sz z  (height)",
         f"{g['H']:.2f} m",
         f"{g['room_sz_n'][2]:.3f}",
         _flag(g["H"], lo_z, hi_z)),
        ("mic_ch1 x  (along wall)",
         f"{g['tx']:.2f} m",
         f"{g['mic_norm'][0]:.3f}",
         "—"),
        ("mic_ch1 y  (dist to wall)",
         f"{g['ty']:.2f} m",
         f"{g['mic_norm'][1]:.3f}",
         _flag(g["ty"], lo_mw, hi_mw)),
        ("mic_ch1 z  (height)",
         f"{g['mz']:.2f} m",
         f"{g['mic_norm'][2]:.3f}",
         "—"),
        ("volume",
         f"{g['vol']:.1f} m³",
         f"{g['vol_norm']:.3f}",
         "—"),
    ]

    header_color   = "#4466aa"
    row_colors     = ["#1e1e3a", "#22223a"]
    ok_color       = "#22cc77"
    warn_color     = "#ffaa33"
    text_color     = "#ccccee"
    label_color    = "#aaaacc"

    # Column x positions
    cx = [0.01, 0.55, 0.76, 0.95]
    row_h = 0.068
    top_y = 0.88

    # Header
    for txt, x in zip(["Parameter", "Raw value", "Normalised", "✓/⚠"], cx):
        ax_film.text(x, top_y + 0.008, txt, color=header_color,
            fontsize=9, fontweight="bold", va="center")
    ax_film.axhline(top_y - 0.012, color="#334466", lw=1)

    film_indices = list(range(8))
    for i, (lbl, raw, norm, flag) in enumerate(rows):
        y = top_y - (i + 1) * row_h
        # Row background
        ax_film.add_patch(Rectangle((0, y - row_h * 0.42), 1, row_h * 0.84,
            color=row_colors[i % 2], zorder=0, clip_on=False))
        # Index badge
        ax_film.text(cx[0], y, f"[{film_indices[i]}]  {lbl}",
            color=label_color, fontsize=9, va="center")
        ax_film.text(cx[1], y, raw,   color=text_color, fontsize=9.5, va="center")
        ax_film.text(cx[2], y, norm,  color=text_color, fontsize=9.5, va="center")
        fc = ok_color if flag == "✓" else (warn_color if flag == "⚠" else "#777799")
        ax_film.text(cx[3], y, flag,  color=fc, fontsize=11, va="center", ha="center")

    # Separator
    ax_film.axhline(top_y - len(rows) * row_h - 0.01, color="#334466", lw=1)

    # Extra: angle + radius (not in room_params but passed as labels)
    extra_y = top_y - (len(rows) + 1) * row_h
    ax_film.text(0.5, extra_y + 0.025,
        "Target labels  (passed to model as labels, not FiLM)",
        color="#888899", fontsize=8.5, ha="center", va="center")
    for j, (lbl, raw, flag) in enumerate([
        ("angle (folded)", f"{g['angle_deg']:.1f}°",
         _flag(g["angle_deg"], lo_ang, hi_ang)),
        ("radius", f"{g['radius']:.3f} m", "—"),
    ]):
        y2 = extra_y - (j + 0.5) * row_h
        ax_film.add_patch(Rectangle((0, y2 - row_h * 0.42), 1, row_h * 0.84,
            color=row_colors[j % 2], zorder=0, clip_on=False))
        ax_film.text(cx[0], y2, lbl,   color=label_color, fontsize=9, va="center")
        ax_film.text(cx[1], y2, raw,   color=text_color, fontsize=9.5, va="center")
        ax_film.text(cx[2], y2, "—",   color="#777799", fontsize=9.5, va="center")
        fc = ok_color if flag == "✓" else (warn_color if flag == "⚠" else "#777799")
        ax_film.text(cx[3], y2, flag,  color=fc, fontsize=11, va="center", ha="center")

    # Nearest wall annotation
    warn_y = extra_y - (len([1, 2]) + 1) * row_h
    ax_film.text(0.5, warn_y,
        f"Nearest wall: {g['nearest_wall']}  |  dist = {g['wall_dist']:.2f} m  "
        f"(training band: 0.40–0.80 m)",
        color=warn_color if not (_TRAIN["mic_wall"][0] <= g["wall_dist"] <= _TRAIN["mic_wall"][1]) else ok_color,
        fontsize=8.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3",
                  fc="#2a2040", ec=warn_color if g["wall_dist"] > 0.80 else ok_color,
                  alpha=0.8))

    # FiLM vector summary
    vec_y = warn_y - 0.10
    vec_str = " ".join(f"{v:.3f}" for v in g["film_vec"])
    ax_film.text(0.5, vec_y + 0.03,
        "Full FiLM vector  room_params[0:8]:", color="#888899",
        fontsize=8.5, ha="center")
    ax_film.text(0.5, vec_y,
        f"[ {vec_str} ]",
        color="#99bbff", fontsize=8, ha="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="#161630", ec="#334466"))

    # RIR info
    rir_path = room["rir_path"]
    try:
        info = sf.info(rir_path)
        rir_label = (f"RIR:  {Path(rir_path).name}   "
                     f"({info.channels} ch, {info.samplerate} Hz, {info.frames/info.samplerate:.3f} s)")
    except Exception:
        rir_label = f"RIR: {Path(rir_path).name}"
    ax_film.text(0.5, 0.02, rir_label, color="#666677", fontsize=7.5, ha="center")

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
        bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table: all rooms
# ---------------------------------------------------------------------------

def print_summary_table():
    print(f"\n{'Room':<25} {'Config':>6} {'Wall':>8} {'Dist(m)':>8} "
          f"{'T60(s)':>7} {'Angle°':>7} {'Radius':>7} {'InDist':>8}")
    print("-" * 85)
    for key, room in ACE_ROOMS.items():
        g = compute_geometry(room)
        lo, hi = _TRAIN["mic_wall"]
        flag = "✓" if lo <= g["wall_dist"] <= hi else "⚠"
        print(f"{room['label']:<25} {room['config']:>6} "
              f"{g['nearest_wall']:>8} {g['wall_dist']:>8.3f} "
              f"{g['rt60']:>7.3f} {g['angle_deg']:>7.1f} {g['radius']:>7.3f} "
              f"{flag:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/ace_explore",
                        help="Output directory")
    parser.add_argument("--key", default=None,
                        help="Force a specific room key (e.g. Office_2_c1)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_summary_table()

    if args.key:
        key = args.key
        room = ACE_ROOMS[key]
        g    = compute_geometry(room)
    else:
        key, room, g = select_sample()

    print(f"\nSelected sample:  {key}  ({room['label']})")
    print(f"  nearest wall : {g['nearest_wall']} at {g['wall_dist']:.3f} m")
    print(f"  angle (fold) : {g['angle_deg']:.1f}°")
    print(f"  radius       : {g['radius']:.3f} m")
    print(f"  RT60         : {g['rt60']:.3f} s")
    print(f"  FiLM vec     : {g['film_vec']}")

    plot_sample(room, g, out_dir / "geometry.png")
