#!/usr/bin/env python3
"""
demo_real_vs_synthetic.py

Takes one representative BUT sample (SpkID03/mic18, width0 wall — best match
to the training distribution), then:

  1. Plots the room geometry (floor-plan view)
  2. Generates a synthetic RIR with gpuRIR using the exact same parameters
  3. Convolves a real speech clip with both RIRs
  4. Saves everything to <out_dir>/ so you can listen and compare

Outputs
-------
  out_dir/
    geometry.png              — floor-plan with mic, speaker, angle
    rir_real.wav              — the actual BUT sweep-sine RIR
    rir_synthetic.wav         — gpuRIR simulation of same geometry
    speech_dry.wav            — the dry (anechoic) speech segment
    reverb_real.wav           — speech convolved with real RIR
    reverb_synthetic.wav      — speech convolved with synthetic RIR
    rir_comparison.png        — waveform + EDC + spectrogram overlay
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import soundfile as sf
import torchaudio
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gpuRIR
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

# ---------------------------------------------------------------------------
# Config — which sample to use
# ---------------------------------------------------------------------------

SAMPLE = dict(
    spk_setup = "SpkID03_20190503_S",
    mic_id    = "18",
)

SPEECH_FILE = (
    "/storage/reie/data/rec-rir/speech/VCTK/VCTK-Corpus/"
    "VCTK-Corpus/wav48/p326/p326_118.wav"
)
SEG_SECONDS = 4.0          # speech segment length
TARGET_SR   = 16_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mono(path: str, target_sr: int, seg_s: float | None = None) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)                          # stereo → mono
    if sr != target_sr:
        import torchaudio.functional as AF
        import torch
        t = torch.from_numpy(wav).unsqueeze(0)
        t = AF.resample(t, sr, target_sr)
        wav = t.squeeze(0).numpy()
    if seg_s is not None:
        n = int(seg_s * target_sr)
        if len(wav) >= n:
            wav = wav[:n]
        else:
            reps = math.ceil(n / len(wav))
            wav  = np.tile(wav, reps)[:n]
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak
    return wav


def save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    wav = wav / max(np.abs(wav).max(), 1e-9)        # normalise to ±1
    sf.write(str(path), wav.astype(np.float32), sr)
    print(f"  saved  {path}  ({len(wav)/sr:.2f}s)")


def simulate_rir(room_sz, pos_src, pos_rcv, rt60, sr) -> np.ndarray:
    Tmax = gpuRIR.att2t_SabineEstimator(60.0, rt60)
    nb_img = gpuRIR.t2n(Tmax, room_sz)
    beta   = gpuRIR.beta_SabineEstimation(room_sz, rt60)
    rir    = gpuRIR.simulateRIR(
        room_sz=room_sz, beta=beta,
        pos_src=pos_src, pos_rcv=pos_rcv,
        nb_img=nb_img, Tmax=Tmax, Tdiff=None, fs=sr,
    )
    return rir[0, 0, :]                             # [T]


def simulate_rir_cpu(room_sz, pos_src, pos_rcv, rt60, sr) -> np.ndarray:
    """Generate RIR using rir-generator (Allen & Berkley CPU ISM)."""
    import rir_generator
    # Use the same Sabine beta as gpuRIR for a fair comparison
    beta = gpuRIR.beta_SabineEstimation(room_sz, rt60)
    filter_length = int((rt60 + 0.1) * sr)
    room_sz_arr  = np.ascontiguousarray(room_sz.reshape(1, 3), dtype=np.float64)
    pos_src_arr  = np.ascontiguousarray(pos_src.reshape(1, 3), dtype=np.float64)
    pos_rcv_arr  = np.ascontiguousarray(pos_rcv.reshape(1, 3), dtype=np.float64)
    h = rir_generator.generate(
        c=343.0,
        fs=sr,
        r=pos_rcv_arr,
        s=pos_src_arr[0],
        L=room_sz_arr[0],
        beta=beta.astype(np.float64),
        reverberation_time=rt60,
        nsample=filter_length,
        mtype=rir_generator.mtype.omnidirectional,
    )
    return np.array(h).T[0]                         # [T]


def convolve(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    import torch
    s = torch.from_numpy(speech)
    h = torch.from_numpy(rir.astype(np.float32))
    out = AF.fftconvolve(s, h, mode="full")[: len(speech)]
    if out.shape[0] < len(speech):
        out = torch.nn.functional.pad(out, (0, len(speech) - out.shape[0]))
    return out.numpy()


# ---------------------------------------------------------------------------
# Geometry plot
# ---------------------------------------------------------------------------

def plot_geometry(
    room_d: float, room_w: float, room_h: float,
    mic_d:  float, mic_w:  float, mic_h:  float,
    spk_d:  float, spk_w:  float, spk_h:  float,
    angle_deg: float, radius_m: float,
    nearest_wall: str, rt60: float,
    label: str, out_path: Path,
) -> None:
    Lx, Ly = room_d, room_w     # BUT convention: x=depth, y=width
    mx, my = mic_d, mic_w
    sx, sy = spk_d, spk_w

    fig, ax = plt.subplots(figsize=(8, 6))
    margin = 0.4
    ax.set_xlim(-margin, Lx + margin)
    ax.set_ylim(-margin, Ly + margin)
    ax.set_aspect("equal", adjustable="box")

    # Room boundary
    ax.add_patch(Rectangle((0, 0), Lx, Ly,
                 fill=False, edgecolor="black", linewidth=2.2))

    # Wall labels
    ax.text(Lx / 2, -0.28, f"depth0  (x=0 → x={Lx:.2f} m)",
            ha="center", va="top", fontsize=8, color="#555")
    ax.text(-0.25, Ly / 2, f"width0\n(y=0→{Ly:.2f}m)",
            ha="right", va="center", fontsize=7, color="#555", rotation=90)

    # Training-distribution band (y0 band, matching training mic placement)
    band_o, band_w = 0.4, 0.4
    ax.add_patch(Rectangle((band_o, band_o), Lx - 2*band_o, band_w,
                 facecolor="#4393c3", edgecolor="#2166ac",
                 alpha=0.15, linewidth=1.0, linestyle="--",
                 label=f"training y0 band [{band_o:.1f}, {band_o+band_w:.1f}] m"))

    # Mic
    ax.scatter([mx], [my], s=180, c="#2166ac", marker="s", zorder=5, label="mic")
    ax.annotate(f" mic ({mx:.2f}, {my:.2f})\n h={mic_h:.2f}m",
                (mx, my), fontsize=8, va="center", color="#2166ac")

    # Speaker
    ax.scatter([sx], [sy], s=220, c="#d6604d", marker="*", zorder=5, label="speaker")
    ax.annotate(f" speaker ({sx:.2f}, {sy:.2f})\n h={spk_h:.2f}m",
                (sx, sy), fontsize=8, va="top", color="#d6604d")

    # Arrow mic → speaker
    ax.annotate("", xy=(sx, sy), xytext=(mx, my),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
    mid_x = (mx + sx) / 2 + 0.15
    mid_y = (my + sy) / 2 + 0.15
    ax.text(mid_x, mid_y,
            f"r = {radius_m:.2f} m\nangle = {angle_deg:.1f}°",
            fontsize=8, color="#333",
            bbox=dict(fc="white", ec="#ccc", boxstyle="round,pad=0.2", alpha=0.8))

    # Nearest-wall annotation
    ax.text(Lx + 0.05, my, f"◀ nearest wall: {nearest_wall}",
            fontsize=8, va="center", color="#888")

    ax.set_xlabel("Depth (m) — BUT x-axis", fontsize=10)
    ax.set_ylabel("Width (m) — BUT y-axis", fontsize=10)
    ax.set_title(
        f"{label}\n"
        f"Room {Lx:.2f} × {Ly:.2f} × {room_h:.2f} m  |  RT60 = {rt60:.3f} s  |  nearest_wall = {nearest_wall}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved  {out_path}")


# ---------------------------------------------------------------------------
# RIR comparison plot  (3 columns: Real | gpuRIR ISM | rir-generator ISM)
# ---------------------------------------------------------------------------

def plot_rir_comparison(
    real_rir:  np.ndarray,
    synt_rir:  np.ndarray,
    cpu_rir:   np.ndarray,
    sr: int,
    out_path: Path,
) -> None:
    from scipy.signal import spectrogram as _spec

    def edc_db(h):
        e = h ** 2
        c = np.cumsum(e[::-1])[::-1]
        return 10 * np.log10(np.maximum(c / c[0], 1e-12))

    entries = [
        (real_rir, "Real (BUT swept-sine)",         "#1f77b4"),
        (synt_rir, "Synthetic — gpuRIR (GPU ISM)",  "#d62728"),
        (cpu_rir,  "Synthetic — rir-generator (CPU ISM)", "#2ca02c"),
    ]
    ncols = len(entries)
    fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols, 9),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.3})

    for col, (rir, name, color) in enumerate(entries):
        t = np.arange(len(rir)) / sr

        # Waveform
        axes[0, col].plot(t, rir, color=color, lw=0.5)
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].set_ylabel("Amplitude")
        axes[0, col].set_xlabel("Time (s)")
        axes[0, col].grid(True, lw=0.3, alpha=0.4)

        # EDC
        ed = edc_db(rir)
        axes[1, col].plot(t, ed, color=color, lw=0.9)
        axes[1, col].axhline(-60, color="k", lw=0.8, ls="--", alpha=0.5, label="−60 dB")
        axes[1, col].set_ylim(-80, 5)
        axes[1, col].set_ylabel("EDC (dB)")
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].legend(fontsize=7)
        axes[1, col].grid(True, lw=0.3, alpha=0.4)

        # Spectrogram
        nperseg = 256
        f, ts, Sxx = _spec(rir, fs=sr, nperseg=nperseg,
                           noverlap=nperseg // 2, scaling="spectrum")
        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
        axes[2, col].pcolormesh(ts, f / 1000, Sxx_db,
                                vmin=-100, vmax=0, cmap="inferno", shading="auto")
        axes[2, col].set_ylabel("kHz")
        axes[2, col].set_xlabel("Time (s)")

    fig.suptitle(
        "Real vs Synthetic RIRs — SpkID03 / mic 18\n"
        "(same geometry & β, two different ISM engines)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--but_csv",     type=Path,
                    default=Path("config/but_matched.csv"))
    ap.add_argument("--speech_file", type=str,
                    default=SPEECH_FILE)
    ap.add_argument("--out_dir",     type=Path,
                    default=Path("results/demo_real_vs_synthetic"))
    ap.add_argument("--seg_seconds", type=float, default=SEG_SECONDS)
    args = ap.parse_args(argv)

    out_dir = args.out_dir.expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Find sample in CSV ────────────────────────────────────────────────
    row: dict | None = None
    for r in csv.DictReader(open(args.but_csv)):
        if r["spk_setup"] == SAMPLE["spk_setup"] and r["mic_id"] == SAMPLE["mic_id"]:
            row = r
            break
    if row is None:
        sys.exit(f"Sample {SAMPLE} not found in {args.but_csv}")

    spk_setup    = row["spk_setup"]
    mic_id       = row["mic_id"]
    room_d       = float(row["room_depth"])
    room_w       = float(row["room_width"])
    room_h       = float(row["room_height"])
    mic_d        = float(row["mic_depth"])
    mic_w        = float(row["mic_width"])
    mic_h        = float(row["mic_height"])
    rt60         = float(row["rt60"])
    nearest_wall = row["nearest_wall"]
    angle_deg    = float(row["angle_deg"])
    radius_m     = float(row["radius_m"])
    rir_wav_path = row["rir_path"]

    # Read speaker position from spk_meta.txt
    spk_meta = Path(rir_wav_path).parent.parent.parent / "spk_meta.txt"
    fields: dict[str, str] = {}
    with open(spk_meta) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("$"):
                k, _, v = line.partition("\t")
                fields[k.lstrip("$")] = v.strip()
    spk_d = float(fields["EnvSpk1Depth"])
    spk_w = float(fields["EnvSpk1Width"])
    spk_h = float(fields["EnvSpk1Height"])

    label = f"{spk_setup} / mic {mic_id}"
    print(f"\n=== Sample: {label} ===")
    print(f"  Room   : {room_d:.3f} × {room_w:.3f} × {room_h:.3f} m")
    print(f"  Mic    : ({mic_d:.4f}, {mic_w:.4f}, {mic_h:.4f})")
    print(f"  Speaker: ({spk_d:.3f}, {spk_w:.3f}, {spk_h:.3f})")
    print(f"  Nearest wall: {nearest_wall}  |  angle={angle_deg:.1f}°  radius={radius_m:.3f} m")
    print(f"  RT60 (metadata): {rt60:.3f} s\n")

    # ── 1. Geometry plot ──────────────────────────────────────────────────
    print("1. Plotting geometry …")
    plot_geometry(
        room_d, room_w, room_h,
        mic_d, mic_w, mic_h,
        spk_d, spk_w, spk_h,
        angle_deg, radius_m, nearest_wall, rt60,
        label, out_dir / "geometry.png",
    )

    # ── 2. Load real RIR ──────────────────────────────────────────────────
    print("2. Loading real RIR …")
    real_rir, sr_rir = sf.read(rir_wav_path, dtype="float32", always_2d=True)
    real_rir = real_rir.mean(axis=1)
    # Resample to target SR if needed
    if sr_rir != TARGET_SR:
        import torch
        t = torch.from_numpy(real_rir).unsqueeze(0)
        t = AF.resample(t, sr_rir, TARGET_SR)
        real_rir = t.squeeze(0).numpy()
    real_rir = real_rir / max(np.abs(real_rir).max(), 1e-9)
    save_wav(out_dir / "rir_real.wav", real_rir, TARGET_SR)

    # ── 3. Generate synthetic RIRs ────────────────────────────────────────
    room_sz = np.array([room_d, room_w, room_h], dtype=np.float64)
    pos_src = np.array([[spk_d, spk_w, spk_h]], dtype=np.float64)
    pos_rcv = np.array([[mic_d, mic_w, mic_h]], dtype=np.float64)

    print("3a. Generating synthetic RIR with gpuRIR (GPU ISM) …")
    synt_rir = simulate_rir(room_sz, pos_src, pos_rcv, rt60, TARGET_SR)
    synt_rir = synt_rir / max(np.abs(synt_rir).max(), 1e-9)
    save_wav(out_dir / "rir_synthetic_gpurir.wav", synt_rir.astype(np.float32), TARGET_SR)

    print("3b. Generating synthetic RIR with rir-generator (CPU ISM) …")
    cpu_rir = simulate_rir_cpu(room_sz, pos_src, pos_rcv, rt60, TARGET_SR)
    cpu_rir = cpu_rir / max(np.abs(cpu_rir).max(), 1e-9)
    save_wav(out_dir / "rir_synthetic_cpu.wav", cpu_rir.astype(np.float32), TARGET_SR)

    # Print peak timing for both
    for tag, h in [("gpuRIR", synt_rir), ("rir-gen", cpu_rir)]:
        pk = int(np.argmax(np.abs(h)))
        print(f"  [{tag}] peak at sample {pk} = {h[pk]:.4f}  ({pk/TARGET_SR*1000:.2f} ms)")

    # ── 4. RIR comparison plot (3-way) ────────────────────────────────────
    print("4. Plotting 3-way RIR comparison …")
    plot_rir_comparison(
        real_rir, synt_rir.astype(np.float32), cpu_rir.astype(np.float32),
        TARGET_SR, out_dir / "rir_comparison.png",
    )

    # ── 5. Load speech and convolve ───────────────────────────────────────
    print("5. Loading speech …")
    speech = load_mono(args.speech_file, TARGET_SR, seg_s=args.seg_seconds)
    save_wav(out_dir / "speech_dry.wav", speech, TARGET_SR)

    print("6. Convolving with real RIR …")
    reverb_real = convolve(speech, real_rir)
    save_wav(out_dir / "reverb_real.wav", reverb_real, TARGET_SR)

    print("7. Convolving with gpuRIR synthetic …")
    reverb_gpu = convolve(speech, synt_rir.astype(np.float32))
    save_wav(out_dir / "reverb_synthetic_gpurir.wav", reverb_gpu, TARGET_SR)

    print("8. Convolving with rir-generator synthetic …")
    reverb_cpu = convolve(speech, cpu_rir.astype(np.float32))
    save_wav(out_dir / "reverb_synthetic_cpu.wav", reverb_cpu, TARGET_SR)

    print(f"\nDone!  All files written to: {out_dir}")
    print("\nKey files to compare:")
    print("  reverb_real.wav               — real BUT room acoustics")
    print("  reverb_synthetic_gpurir.wav   — gpuRIR GPU ISM simulation")
    print("  reverb_synthetic_cpu.wav      — rir-generator CPU ISM simulation")
    print("  rir_comparison.png            — 3-way waveform / EDC / spectrogram")


if __name__ == "__main__":
    main()
