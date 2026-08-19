#!/usr/bin/env python3
"""
compare_rirs.py — Plot real (BUT) vs synthetic (gpuRIR) RIRs side by side.

For a selection of entries, shows:
  • Time-domain waveform (real vs synthetic overlaid)
  • Energy Decay Curve (EDC) with estimated RT60 marked
  • Spectrogram (log-magnitude)

Usage (from project root):
    python scripts/compare_rirs.py --out_dir results/rir_comparison
"""

from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram as scipy_spectrogram

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load mono float32; mix to mono if needed."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] > 1:
        x = x.mean(axis=1)
    else:
        x = x[:, 0]
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak
    return x, sr


def estimate_rt60(rir: np.ndarray, sr: int) -> float:
    """Estimate RT60 from the backward-integrated energy decay curve (Schroeder)."""
    energy = rir ** 2
    # Schroeder backward integration
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10 * np.log10(np.maximum(edc / edc[0], 1e-12))
    # Find where EDC crosses -5 dB and -65 dB and interpolate for RT60
    t = np.arange(len(edc_db)) / sr
    try:
        t5  = t[np.where(edc_db <= -5)[0][0]]
        t65 = t[np.where(edc_db <= -65)[0][0]]
        rt60 = (t65 - t5) * 60.0 / 60.0  # from -5 to -65 dB = 60 dB range
    except IndexError:
        t5  = t[np.where(edc_db <= -5)[0][0]]  if (edc_db <= -5).any()  else None
        t25 = t[np.where(edc_db <= -25)[0][0]] if (edc_db <= -25).any() else None
        if t5 is not None and t25 is not None:
            rt60 = (t25 - t5) * 3.0   # T20 * 3 ≈ RT60
        else:
            rt60 = float("nan")
    return rt60


def edc_db(rir: np.ndarray) -> np.ndarray:
    energy = rir ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    return 10 * np.log10(np.maximum(edc / edc[0], 1e-12))


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

ENTRIES = [
    # (spk_setup,   mic_id, label)
    ("SpkID01_20190503_S", "18", "SpkID01/mic18  width0  gt=108°"),
    ("SpkID01_20190503_S", "19", "SpkID01/mic19  depth0  gt=77°"),
    ("SpkID01_20190503_S", "29", "SpkID01/mic29  widthMax  gt=80°"),
    ("SpkID03_20190503_S", "18", "SpkID03/mic18  width0  gt=172°"),
    ("SpkID03_20190503_S", "19", "SpkID03/mic19  depth0  gt=81°"),
    ("SpkID05_20190503_S", "29", "SpkID05/mic29  widthMax  gt=156°"),
]


def plot_entry(
    label: str,
    real_wav: np.ndarray,
    synt_wav: np.ndarray,
    sr: int,
    rt60_meta: float,
    ax_wave, ax_edc, ax_spec_real, ax_spec_synt,
) -> None:
    t_real = np.arange(len(real_wav)) / sr
    t_synt = np.arange(len(synt_wav)) / sr

    # ── waveform ──────────────────────────────────────────────────────────
    ax_wave.plot(t_real, real_wav, color="steelblue", lw=0.5, alpha=0.8, label="Real (BUT)")
    ax_wave.plot(t_synt, synt_wav, color="crimson",   lw=0.5, alpha=0.8, label="Synthetic")
    ax_wave.set_xlim(0, max(t_real[-1], t_synt[-1]))
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(label, fontsize=8)
    ax_wave.legend(fontsize=6, loc="upper right")
    ax_wave.grid(True, lw=0.3, alpha=0.5)

    # ── EDC ──────────────────────────────────────────────────────────────
    edc_r = edc_db(real_wav)
    edc_s = edc_db(synt_wav)
    rt60_r = estimate_rt60(real_wav, sr)
    rt60_s = estimate_rt60(synt_wav, sr)

    ax_edc.plot(t_real, edc_r, color="steelblue", lw=0.8,
                label=f"Real  RT60≈{rt60_r:.2f}s")
    ax_edc.plot(t_synt, edc_s, color="crimson",   lw=0.8,
                label=f"Synth RT60≈{rt60_s:.2f}s  (meta={rt60_meta:.2f}s)")
    ax_edc.axhline(-60, color="k", lw=0.6, ls="--", alpha=0.5)
    ax_edc.set_ylim(-80, 5)
    ax_edc.set_ylabel("EDC (dB)")
    ax_edc.legend(fontsize=6, loc="upper right")
    ax_edc.grid(True, lw=0.3, alpha=0.5)

    # ── spectrograms ─────────────────────────────────────────────────────
    for ax, wav, title in [
        (ax_spec_real, real_wav, "Real spectrogram"),
        (ax_spec_synt, synt_wav, "Synthetic spectrogram"),
    ]:
        nperseg = 256
        f, t_s, Sxx = scipy_spectrogram(wav, fs=sr, nperseg=nperseg,
                                         noverlap=nperseg//2, scaling="spectrum")
        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
        ax.pcolormesh(t_s, f / 1000, Sxx_db, vmin=-100, vmax=0,
                      cmap="inferno", shading="auto", rasterized=True)
        ax.set_ylabel("kHz")
        ax.set_title(title, fontsize=7)
        ax.set_ylim(0, sr / 2000)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--but_csv",   type=Path,
                    default=Path("config/but_matched.csv"))
    ap.add_argument("--synt_csv",  type=Path,
                    default=Path("results/but_synthetic_rirs/but_synthetic.csv"))
    ap.add_argument("--out_dir",   type=Path,
                    default=Path("results/rir_comparison"))
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir.expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup: (spk_setup, mic_id) → row
    real_meta: dict[tuple, dict] = {}
    for r in csv.DictReader(open(args.but_csv)):
        p = r["rir_path"]
        parts = p.split("/")
        idx = parts.index("RIR")
        real_meta[(parts[idx-2], parts[idx-1])] = r

    synt_meta: dict[tuple, dict] = {}
    for r in csv.DictReader(open(args.synt_csv)):
        p = r["rir_path"]
        parts = p.replace("\\", "/").split("/")
        fname = parts[-1]                            # mic18_synthetic.wav
        mic   = fname.replace("mic","").replace("_synthetic.wav","")
        spk   = parts[-2]
        synt_meta[(spk, mic)] = r

    n = len(ENTRIES)
    # 4 rows per entry (wave, edc, spec_real, spec_synth), n columns
    fig, axes = plt.subplots(
        4, n, figsize=(4 * n, 12),
        gridspec_kw={"hspace": 0.6, "wspace": 0.35},
    )
    if n == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["Waveform", "EDC", "Real spectrogram", "Synthetic spectrogram"]
    for ri, rl in enumerate(row_labels):
        axes[ri, 0].set_ylabel(rl, fontsize=8, labelpad=4)

    for col, (spk, mic, lbl) in enumerate(ENTRIES):
        key = (spk, mic)
        if key not in real_meta:
            print(f"[warn] {key} not in real CSV, skipping")
            continue
        if key not in synt_meta:
            print(f"[warn] {key} not in synthetic CSV, skipping")
            continue

        real_path = real_meta[key]["rir_path"]
        synt_path = synt_meta[key]["rir_path"]
        rt60_meta = float(real_meta[key]["rt60"])

        real_wav, sr = load_wav(real_path)
        synt_wav, _  = load_wav(synt_path)

        plot_entry(
            lbl,
            real_wav, synt_wav, sr, rt60_meta,
            axes[0, col], axes[1, col], axes[2, col], axes[3, col],
        )
        # x-label only on bottom row
        axes[3, col].set_xlabel("Time (s)", fontsize=7)

    fig.suptitle("Real (BUT) vs Synthetic (gpuRIR) RIR comparison", fontsize=12, y=1.01)
    out_path = out_dir / "real_vs_synthetic_rirs.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
