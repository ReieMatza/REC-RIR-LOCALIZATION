#!/usr/bin/env python3
"""
explore_locata_sample.py

Selects one sample from the LOCATA dev dataset, prints all of its
parameters in a readable table, and produces two annotated figures:
  1. geometry.png  — bird's-eye floor-plan
  2. audio_panel.png — waveform + spectrogram of the reference mic

No LOCATA SDK required.  Position files are plain TSV; audio is float32
WAV at 48 kHz (resampled to 16 kHz for display).

Usage
-----
  cd /storage/reie/REC-RIR-LOCALIZATION
  python scripts/explore_locata_sample.py

Options
-------
  --locata_root   Path to the LOCATA dev root  (default: /storage/data/locata/dev)
  --task          e.g. task1                   (default: task1)
  --recording     e.g. recording1              (default: recording1)
  --array         e.g. dicit                   (default: dicit)
  --source        e.g. loudspeaker1            (default: loudspeaker1)
  --ref_mic       0-based channel index        (default: 6  = mic7 = array centre)
  --out_dir       Output directory             (default: results/locata_explore)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

# ---------------------------------------------------------------------------
# Constants from the LOCATA paper
# ---------------------------------------------------------------------------
ROOM_SZ_M = np.array([7.10, 9.80, 3.00])   # m  (from paper, Section IV-A)
RT60_S    = 0.55                             # s  (from paper, "about 0.55 s")

# Training distribution limits (for the ✓/⚠ check)
_TRAIN = dict(
    room_xy=(3.0, 15.0),
    room_z=(2.5, 6.0),
    rt60=(0.2, 1.5),
    radius=(0.0, 13.0),
    angle=(0.0, 180.0),
    mic_wall=(0.4, 0.8),     # y0-wall convention: mic y in [0.4, 0.8] m
)


# ---------------------------------------------------------------------------
# Position-file parser
# ---------------------------------------------------------------------------

def parse_position_file(path: Path) -> dict:
    """Parse a LOCATA position TSV and average all rows (static for Task 1).

    Returns a flat dict of column_name -> mean_value (floats).
    """
    rows = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(line.split("\t"))

    if not rows:
        raise ValueError(f"No data rows in {path}")

    values: dict[str, list[float]] = {h: [] for h in header}
    for row in rows:
        for h, v in zip(header, row):
            try:
                values[h].append(float(v))
            except ValueError:
                pass  # skip non-numeric fields (year/month/day etc.)

    return {k: float(np.mean(v)) for k, v in values.items() if v}


def extract_source_xyz(pos: dict) -> np.ndarray:
    return np.array([pos["x"], pos["y"], pos["z"]])


def extract_array_center_xyz(pos: dict) -> np.ndarray:
    return np.array([pos["x"], pos["y"], pos["z"]])


def extract_mic_positions(pos: dict, n_mics: int) -> np.ndarray:
    """Return shape (n_mics, 3) array of mic XYZ positions."""
    mics = []
    for i in range(1, n_mics + 1):
        mics.append([pos[f"mic{i}_x"], pos[f"mic{i}_y"], pos[f"mic{i}_z"]])
    return np.array(mics)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def compute_geometry(source_xyz: np.ndarray, array_xyz: np.ndarray) -> dict:
    delta_xy   = source_xyz[:2] - array_xyz[:2]
    radius_m   = float(np.linalg.norm(delta_xy))
    angle_deg  = float(np.degrees(np.arctan2(delta_xy[1], delta_xy[0])) % 360)
    folded_deg = float(min(angle_deg, 360.0 - angle_deg))   # → [0, 180]
    return dict(delta_xy=delta_xy, radius_m=radius_m,
                angle_global_deg=angle_deg, folded_angle_deg=folded_deg)


def compute_wall_distances(array_xyz: np.ndarray) -> dict:
    """Approximate distances to each wall assuming the coordinate origin
    is at the centre of the LOCATA room (7.1 × 9.8 m floor plan).
    """
    half = ROOM_SZ_M[:2] / 2.0
    x, y = float(array_xyz[0]), float(array_xyz[1])
    return {
        "x-": x + half[0],
        "x+": half[0] - x,
        "y-": y + half[1],
        "y+": half[1] - y,
    }


# ---------------------------------------------------------------------------
# Audio and VAD
# ---------------------------------------------------------------------------

def load_audio_channel(wav_path: Path, channel: int, target_sr: int = 16_000):
    """Load one channel of a multi-channel WAV and resample to target_sr.

    Returns (waveform_np, original_sr).
    """
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    ch = data[:, channel]                          # shape (N,)
    if sr != target_sr:
        t = torch.from_numpy(ch).unsqueeze(0)
        t = AF.resample(t, sr, target_sr)
        ch = t.squeeze(0).numpy()
    return ch, sr


def parse_vad(vad_path: Path) -> np.ndarray:
    """Return VAD as a boolean numpy array (one entry per audio sample)."""
    with open(vad_path) as fh:
        lines = fh.readlines()
    # Skip header line ("VAD")
    return np.array([int(ln.strip()) for ln in lines[1:] if ln.strip()], dtype=bool)


def find_first_active_window(vad: np.ndarray, sr_orig: int, min_dur_s: float = 1.0) -> tuple:
    """Find the first contiguous active window >= min_dur_s seconds.

    Returns (start_s, end_s) or None.
    """
    min_samples = int(min_dur_s * sr_orig)
    i = 0
    while i < len(vad):
        if vad[i]:
            j = i
            while j < len(vad) and vad[j]:
                j += 1
            if (j - i) >= min_samples:
                return i / sr_orig, j / sr_orig
            i = j
        else:
            i += 1
    return None


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def check(val: float, lo: float, hi: float) -> str:
    return "✓" if lo <= val <= hi else "✗"


def print_summary(
    *,
    task: str, recording: str, array: str, source_id: str,
    ref_mic_idx: int,
    source_xyz: np.ndarray, array_xyz: np.ndarray, ref_mic_xyz: np.ndarray,
    mic_positions: np.ndarray,
    geom: dict,
    wall_dists: dict,
    audio_path: Path,
    n_channels: int,
    sr_orig: int,
    duration_s: float,
    rms: float,
    vad_active_frac: float,
    active_window: tuple | None,
) -> None:
    W = 65
    sep = "=" * W

    print(f"\n{sep}")
    print(f"LOCATA Sample — {task} / {recording} / {array}")
    print(sep)

    # --- Positions ---
    print("\n--- Raw positions (global LOCATA frame, metres) ---")
    print(f"  Source ({source_id}):   "
          f"x = {source_xyz[0]:+.4f}  y = {source_xyz[1]:+.4f}  z = {source_xyz[2]:+.4f}")
    print(f"  Array centre ({array}):  "
          f"x = {array_xyz[0]:+.4f}  y = {array_xyz[1]:+.4f}  z = {array_xyz[2]:+.4f}")
    print(f"  Ref mic (mic{ref_mic_idx+1} / ch {ref_mic_idx}):  "
          f"x = {ref_mic_xyz[0]:+.4f}  y = {ref_mic_xyz[1]:+.4f}  z = {ref_mic_xyz[2]:+.4f}")

    print(f"\n--- {array.upper()} mic positions (x, y, z) ---")
    for i, m in enumerate(mic_positions):
        tag = "  [array centre]" if np.allclose(m, array_xyz, atol=1e-3) else ""
        print(f"  mic{i+1:2d}:  ({m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f}){tag}")

    # --- Geometry ---
    dx, dy = geom["delta_xy"]
    print("\n--- Geometry ---")
    print(f"  Horizontal delta (source − array):  Δx = {dx:+.3f} m,  Δy = {dy:+.3f} m")
    print(f"  Radius (2-D):                        {geom['radius_m']:.3f} m")
    print(f"  Angle (global frame, CCW from +x):  {geom['angle_global_deg']:.1f}°")
    print(f"  Folded angle [0, 180]:              {geom['folded_angle_deg']:.1f}°")

    # --- Room ---
    vol = float(np.prod(ROOM_SZ_M))
    print("\n--- Room (from LOCATA paper) ---")
    print(f"  Size:    {ROOM_SZ_M[0]:.2f} × {ROOM_SZ_M[1]:.2f} × {ROOM_SZ_M[2]:.2f} m")
    print(f"  RT60:    ~{RT60_S:.2f} s")
    print(f"  Volume:  {vol:.1f} m³")

    # --- Nearest wall ---
    nw, nd = min(wall_dists.items(), key=lambda kv: kv[1])
    print("\n--- Approximate nearest wall (centred-origin assumption) ---")
    for wall, d in sorted(wall_dists.items()):
        flag = " ← nearest" if wall == nw else ""
        print(f"  {wall}:  {d:.2f} m{flag}")
    print(f"\n  ⚠  Nearest wall = {nw}, dist ≈ {nd:.2f} m")
    print(f"     Training expects mic 0.4–0.8 m from wall;")
    print(f"     LOCATA arrays are positioned in the room interior.")

    # --- Audio ---
    print("\n--- Audio ---")
    print(f"  File:           {audio_path.name}")
    print(f"  Channels:       {n_channels}  (using channel {ref_mic_idx}, 0-indexed)")
    print(f"  SR (raw):       {sr_orig} Hz  →  16000 Hz after resample")
    print(f"  Duration:       {duration_s:.3f} s")
    print(f"  Ref-ch RMS:     {rms:.5f}")

    # --- VAD ---
    print("\n--- VAD (array-level) ---")
    print(f"  Active fraction: {vad_active_frac*100:.1f} %")
    if active_window:
        print(f"  First active window ≥ 1 s:  [{active_window[0]:.3f} s – {active_window[1]:.3f} s]")
    else:
        print("  No continuous active window ≥ 1 s found.")

    # --- Training distribution check ---
    rx, ry, rz = _TRAIN["room_xy"], _TRAIN["room_xy"], _TRAIN["room_z"]
    print("\n--- Training distribution check ---")
    print(f"  Room Lx  {ROOM_SZ_M[0]:.2f} m  ∈ {_TRAIN['room_xy']}  {check(ROOM_SZ_M[0], *_TRAIN['room_xy'])}")
    print(f"  Room Ly  {ROOM_SZ_M[1]:.2f} m  ∈ {_TRAIN['room_xy']}  {check(ROOM_SZ_M[1], *_TRAIN['room_xy'])}")
    print(f"  Room Lz  {ROOM_SZ_M[2]:.2f} m  ∈ {_TRAIN['room_z']}   {check(ROOM_SZ_M[2], *_TRAIN['room_z'])}")
    print(f"  RT60     {RT60_S:.2f} s    ∈ {_TRAIN['rt60']}   {check(RT60_S, *_TRAIN['rt60'])}")
    print(f"  Radius   {geom['radius_m']:.2f} m   ∈ {_TRAIN['radius']}    {check(geom['radius_m'], *_TRAIN['radius'])}")
    print(f"  Angle    {geom['folded_angle_deg']:.1f}°    ∈ {_TRAIN['angle']}   {check(geom['folded_angle_deg'], *_TRAIN['angle'])}")
    wall_flag = "⚠  (LOCATA arrays are room-central, not near a wall)"
    print(f"  Mic-wall {nd:.2f} m  ∈ {_TRAIN['mic_wall']}   ✗  {wall_flag}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Figure 1 — geometry
# ---------------------------------------------------------------------------

def plot_geometry(
    source_xyz: np.ndarray,
    array_xyz: np.ndarray,
    mic_positions: np.ndarray,
    geom: dict,
    task: str, recording: str, array: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))

    # Room rectangle (approximate, centred origin)
    half = ROOM_SZ_M[:2] / 2.0
    room_rect = Rectangle(
        (-half[0], -half[1]), ROOM_SZ_M[0], ROOM_SZ_M[1],
        fill=False, edgecolor="black", linewidth=2.0, linestyle="--",
        label=f"Room {ROOM_SZ_M[0]:.1f}×{ROOM_SZ_M[1]:.1f} m (approx.)",
    )
    ax.add_patch(room_rect)
    ax.text(0, half[1] + 0.15, "approx. room boundary\n(centred origin assumed)",
            ha="center", va="bottom", fontsize=7, color="#888")

    # Training wall band — 0.4–0.8 m shaded strip on all 4 sides
    band_lo, band_hi = _TRAIN["mic_wall"]
    alpha = 0.08
    color = "#2166ac"
    for (ox, oy, w, h) in [
        # bottom strip
        (-half[0], -half[1], ROOM_SZ_M[0], band_hi),
        # top strip
        (-half[0], half[1] - band_hi, ROOM_SZ_M[0], band_hi),
        # left strip
        (-half[0], -half[1], band_hi, ROOM_SZ_M[1]),
        # right strip
        (half[0] - band_hi, -half[1], band_hi, ROOM_SZ_M[1]),
    ]:
        ax.add_patch(Rectangle((ox, oy), w, h,
                                facecolor=color, alpha=alpha, edgecolor="none"))

    # Label one strip
    ax.text(-half[0] + 0.05, -half[1] + band_hi + 0.05,
            f"training mic band\n0.4–0.8 m from wall",
            fontsize=6.5, color=color, va="bottom")

    # DICIT mics — linear array drawn as line + dots
    mx = mic_positions[:, 0]
    my = mic_positions[:, 1]
    ax.plot(mx, my, color="#4393c3", linewidth=1.2, zorder=3)
    ax.scatter(mx, my, s=30, c="#4393c3", zorder=4, label="DICIT mics (15)")

    # Array centre
    ax.scatter([array_xyz[0]], [array_xyz[1]], s=150, c="#1a6faf",
               marker="s", zorder=5, label="Array centre")
    ax.annotate(
        f" array\n ({array_xyz[0]:.2f}, {array_xyz[1]:.2f})",
        (array_xyz[0], array_xyz[1]), fontsize=7.5, va="top", color="#1a6faf",
    )

    # Source
    ax.scatter([source_xyz[0]], [source_xyz[1]], s=220, c="#d62728",
               marker="*", zorder=5, label="Source (loudspeaker)")
    ax.annotate(
        f" source\n ({source_xyz[0]:.2f}, {source_xyz[1]:.2f})",
        (source_xyz[0], source_xyz[1]), fontsize=7.5, va="bottom", color="#d62728",
    )

    # Arrow array → source
    dx, dy = geom["delta_xy"]
    ax.annotate(
        "", xy=(source_xyz[0], source_xyz[1]),
        xytext=(array_xyz[0], array_xyz[1]),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8),
    )
    mid_x = (array_xyz[0] + source_xyz[0]) / 2 + 0.15
    mid_y = (array_xyz[1] + source_xyz[1]) / 2 + 0.15
    ax.text(mid_x, mid_y,
            f"r = {geom['radius_m']:.2f} m\nangle = {geom['folded_angle_deg']:.1f}°",
            fontsize=8, color="#333",
            bbox=dict(fc="white", ec="#ccc", boxstyle="round,pad=0.2", alpha=0.85))

    # Height annotations
    ax.text(source_xyz[0], source_xyz[1] - 0.25,
            f"z = {source_xyz[2]:.2f} m", fontsize=6.5, ha="center",
            color="#d62728", alpha=0.75)
    ax.text(array_xyz[0], array_xyz[1] - 0.25,
            f"z = {array_xyz[2]:.2f} m", fontsize=6.5, ha="center",
            color="#1a6faf", alpha=0.75)

    margin = 0.5
    ax.set_xlim(-half[0] - margin, half[0] + margin)
    ax.set_ylim(-half[1] - margin, half[1] + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m) — LOCATA global frame", fontsize=10)
    ax.set_ylabel("Y (m) — LOCATA global frame", fontsize=10)
    ax.set_title(
        f"{task} / {recording} / {array}  |  "
        f"RT60 ≈ {RT60_S:.2f} s  |  "
        f"Room {ROOM_SZ_M[0]:.1f}×{ROOM_SZ_M[1]:.1f}×{ROOM_SZ_M[2]:.1f} m\n"
        f"Folded angle = {geom['folded_angle_deg']:.1f}°,  "
        f"radius = {geom['radius_m']:.2f} m",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved  {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — audio panel
# ---------------------------------------------------------------------------

def plot_audio_panel(
    wav_16k: np.ndarray,
    vad_48k: np.ndarray,
    sr_orig: int,
    ref_mic_idx: int,
    task: str, recording: str, array: str,
    out_path: Path,
) -> None:
    from scipy.signal import spectrogram as _spec

    sr = 16_000
    t_wav = np.arange(len(wav_16k)) / sr

    # Downsample VAD to 16 kHz
    ratio = sr / sr_orig
    vad_16k = vad_48k[: int(len(vad_48k) / (sr_orig / sr) + 0.5)]
    # Nearest-neighbour resampling of VAD to match wav_16k length exactly
    idx = np.round(np.arange(len(wav_16k)) * (len(vad_48k) / len(wav_16k))).astype(int)
    idx = np.clip(idx, 0, len(vad_48k) - 1)
    vad_matched = vad_48k[idx]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                             gridspec_kw={"hspace": 0.4, "height_ratios": [1.2, 1.8]})

    # --- Waveform ---
    ax = axes[0]
    ax.plot(t_wav, wav_16k, color="#1f77b4", lw=0.4)
    # VAD overlay
    for i in range(len(vad_matched)):
        if vad_matched[i]:
            ax.axvspan(t_wav[i], t_wav[min(i + 1, len(t_wav) - 1)],
                       color="#2ca02c", alpha=0.12, linewidth=0)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title(
        f"Reference mic  ch {ref_mic_idx}  |  {task}/{recording}/{array}  |  16 kHz",
        fontsize=9,
    )
    ax.set_xlim(0, t_wav[-1])
    ax.grid(True, lw=0.3, alpha=0.4)
    # Legend proxy
    vad_patch = mpatches.Patch(color="#2ca02c", alpha=0.4, label="VAD active")
    ax.legend(handles=[vad_patch], fontsize=7, loc="upper right")

    # --- Spectrogram ---
    ax = axes[1]
    nperseg = 256
    f, ts, Sxx = _spec(wav_16k, fs=sr, nperseg=nperseg,
                       noverlap=nperseg // 2, scaling="spectrum")
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
    im = ax.pcolormesh(ts, f / 1000, Sxx_db,
                       vmin=-90, vmax=-20, cmap="inferno", shading="auto")
    plt.colorbar(im, ax=ax, label="dB")
    ax.set_ylabel("Frequency (kHz)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title("Spectrogram (256-pt STFT, log-power)", fontsize=9)
    ax.set_xlim(0, ts[-1])

    fig.suptitle(
        f"LOCATA audio — {task} / {recording} / {array} — ch {ref_mic_idx}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locata_root", type=Path,
                    default=Path("/storage/data/locata/dev"))
    ap.add_argument("--task",      default="task1")
    ap.add_argument("--recording", default="recording1")
    ap.add_argument("--array",     default="dicit")
    ap.add_argument("--source",    default="loudspeaker1")
    ap.add_argument("--ref_mic",   type=int, default=6,
                    help="0-based mic channel to use as reference (default: 6 = mic7 = DICIT centre)")
    ap.add_argument("--out_dir",   type=Path,
                    default=Path("results/locata_explore"))
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_dir = args.locata_root / args.task / args.recording / args.array
    if not rec_dir.exists():
        raise FileNotFoundError(f"Recording directory not found: {rec_dir}")

    print(f"\nLoading LOCATA sample: {args.task}/{args.recording}/{args.array}")
    print(f"  data dir : {rec_dir}")

    # ── 1. Parse position files ────────────────────────────────────────────
    src_pos_path   = rec_dir / f"position_source_{args.source}.txt"
    array_pos_path = rec_dir / f"position_array_{args.array}.txt"

    print("  parsing position files …")
    src_pos   = parse_position_file(src_pos_path)
    array_pos = parse_position_file(array_pos_path)

    source_xyz     = extract_source_xyz(src_pos)
    array_xyz      = extract_array_center_xyz(array_pos)

    # Detect number of mics from header columns micN_x
    n_mics = sum(1 for k in array_pos if k.startswith("mic") and k.endswith("_x"))
    mic_positions  = extract_mic_positions(array_pos, n_mics)
    ref_mic_xyz    = mic_positions[args.ref_mic]

    # ── 2. Geometry ────────────────────────────────────────────────────────
    geom       = compute_geometry(source_xyz, array_xyz)
    wall_dists = compute_wall_distances(array_xyz)

    # ── 3. Audio and VAD ──────────────────────────────────────────────────
    audio_path = rec_dir / f"audio_array_{args.array}.wav"
    vad_path   = rec_dir / f"VAD_{args.array}_{args.source}.txt"

    print("  loading audio …")
    # Read raw info first (sr, channels, frames)
    with sf.SoundFile(str(audio_path)) as f_info:
        sr_orig   = f_info.samplerate
        n_channels = f_info.channels
        n_frames   = len(f_info)
    duration_s = n_frames / sr_orig

    wav_16k, _ = load_audio_channel(audio_path, args.ref_mic, target_sr=16_000)
    rms        = float(np.sqrt(np.mean(wav_16k ** 2)))

    print("  parsing VAD …")
    vad = parse_vad(vad_path)
    vad_active_frac = float(vad.mean())
    active_window   = find_first_active_window(vad, sr_orig, min_dur_s=1.0)

    # ── 4. Print summary ───────────────────────────────────────────────────
    print_summary(
        task=args.task, recording=args.recording,
        array=args.array, source_id=args.source,
        ref_mic_idx=args.ref_mic,
        source_xyz=source_xyz,
        array_xyz=array_xyz,
        ref_mic_xyz=ref_mic_xyz,
        mic_positions=mic_positions,
        geom=geom,
        wall_dists=wall_dists,
        audio_path=audio_path,
        n_channels=n_channels,
        sr_orig=sr_orig,
        duration_s=duration_s,
        rms=rms,
        vad_active_frac=vad_active_frac,
        active_window=active_window,
    )

    # ── 5. Plot geometry ───────────────────────────────────────────────────
    print("Plotting geometry …")
    plot_geometry(
        source_xyz=source_xyz,
        array_xyz=array_xyz,
        mic_positions=mic_positions,
        geom=geom,
        task=args.task, recording=args.recording, array=args.array,
        out_path=out_dir / "geometry.png",
    )

    # ── 6. Plot audio panel ────────────────────────────────────────────────
    print("Plotting audio panel …")
    plot_audio_panel(
        wav_16k=wav_16k,
        vad_48k=vad,
        sr_orig=sr_orig,
        ref_mic_idx=args.ref_mic,
        task=args.task, recording=args.recording, array=args.array,
        out_path=out_dir / "audio_panel.png",
    )

    print(f"\nDone — results in {out_dir}")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
