#!/usr/bin/env python3
"""
generate_synthetic_rir.py

ACE Office 2 / pos 1 — Ch 1 of the 8-channel linear array.

  1. Extract Ch1 from the real ACE RIR and copy one anechoic speech file
     into results/ace_explore/.
  2. Simulate a matching single-channel 48 kHz RIR with rir-generator
     (CPU / Allen-Berkley ISM).
  3. Simulate the same RIR with gpuRIR (GPU / ISM + diffuse tail).
  4. Generate 16 kHz training-matched RIRs (no diffuse tail, matching
     dataset/gen_rir.py).
  5. Produce a side-by-side comparison plot (time-domain + magnitude spectrum).
  6. Write results/ace_explore/but_matched.csv — feed for eval_but.py.

Room geometry (ACE paper Table 1 — Office 2, pos 1, Lin8Ch):
  dims  : (3.22, 5.10, 2.94) m   (L=depth/x, W=width/y, H=height/z)
  source: (1.41, 1.73, 1.19) m
  Ch 1  : (0.84, 2.78, 1.19) m   ← nearest wall is x=0 (depth0)

Training-frame mapping (nearest_wall = "depth0"):
  Lx = room_width  (5.10),  Ly = room_depth (3.22)
  mx = mic_width   (2.78),  my = mic_depth  (0.84)
  angle = atan2(src_x - mic_x,  src_y - mic_y)   [train_y vs train_x delta]
        = atan2(1.41-0.84, 1.73-2.78) = atan2(0.57, -1.05) ≈ 151.5°

Usage
-----
  cd /storage/reie/REC-RIR-LOCALIZATION
  /storage/reie/.venv_custom_ssh/bin/python scripts/generate_synthetic_rir.py

To evaluate with eval_but.py:
  python scripts/eval_but.py \\
      -c config/<your>.toml \\
      -k /path/to/best.tar \\
      --but_csv results/ace_explore/but_matched.csv \\
      --speech_txt config/reie_train_speech_quick.txt
"""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# ── constants ────────────────────────────────────────────────────────────────
SR          = 48_000
SR_TRAIN    = 16_000
C_SOUND     = 343.0
T60         = 0.392
T_MAX       = 0.60
T_MAX_TRAIN = T60 + 0.1    # 0.492 s → 7872 samples @ 16 kHz

ROOM_SZ = np.array([3.22, 5.10, 2.94])
SRC_POS = np.array([1.41, 1.73, 1.19])
MIC_POS = np.array([0.60, 2.78, 1.19])   # Ch 1 of Lin8Ch (ACE paper Table 1)

ACE_RIR_SRC    = Path("/storage/data/ace/Lin8Ch/Office_2/1/Lin8Ch_803_1_RIR.wav")
ACE_SPEECH_SRC = Path("/storage/data/ace/Speech/F1s1.wav")

OUT_DIR = Path("results/ace_explore")

# ── derived geometry ──────────────────────────────────────────────────────────
_L, _W, _H = ROOM_SZ
# Training-frame convention: ACE x→train y (depth), ACE y→train x (along-wall)
_dx = SRC_POS[1] - MIC_POS[1]   # train_x delta
_dy = SRC_POS[0] - MIC_POS[0]   # train_y delta
ANGLE_DEG = round(math.degrees(math.atan2(_dy, _dx)) % 360.0, 2)
RADIUS_M  = round(math.hypot(_dx, _dy), 3)
MIC_DEPTH, MIC_WIDTH, MIC_HEIGHT = float(MIC_POS[0]), float(MIC_POS[1]), float(MIC_POS[2])


# ── helpers ──────────────────────────────────────────────────────────────────

def save_wav(path: Path, data: np.ndarray, sr: int = SR) -> None:
    """Save a mono float32 array as a 32-bit float WAV."""
    sf.write(str(path), data.astype(np.float32), sr, subtype="FLOAT")
    print(f"  → {path.name}  (mono, {sr} Hz, {len(data)/sr:.3f} s)")


# ── step 1: copy real ACE data ────────────────────────────────────────────────

def copy_ace_files(out_dir: Path) -> np.ndarray:
    """Extract Ch1 from the 8-channel ACE RIR and copy the speech file."""
    print("\n[1] Copying real ACE files …")
    raw, sr = sf.read(str(ACE_RIR_SRC))
    ch1 = raw[:, 0].astype(np.float32)
    out_rir = out_dir / "ace_rir_ch1.wav"
    sf.write(str(out_rir), ch1, sr, subtype="FLOAT")
    print(f"  → ace_rir_ch1.wav  (mono, {sr} Hz, {len(ch1)/sr:.3f} s)")
    shutil.copy2(ACE_SPEECH_SRC, out_dir / "ace_speech.wav")
    print(f"  → ace_speech.wav   (mono, {sr} Hz)")
    return ch1


# ── step 2: rir-generator 48 kHz ─────────────────────────────────────────────

def generate_rir_rg(out_dir: Path) -> np.ndarray:
    """Allen-Berkley ISM via rir-generator, 48 kHz, single channel."""
    import rir_generator as rir
    print("\n[2] rir-generator (CPU ISM, 48 kHz) …")
    h = rir.generate(
        c=C_SOUND, fs=SR,
        r=MIC_POS[np.newaxis, :],
        s=SRC_POS, L=ROOM_SZ,
        reverberation_time=T60,
        nsample=int(T_MAX * SR),
        mtype=rir.mtype.omnidirectional,
        order=-1, dim=3, hp_filter=True,
    )
    h = np.array(h, dtype=np.float32).flatten()
    save_wav(out_dir / "synthetic_rir_rir_generator.wav", h)
    return h


# ── step 3: gpuRIR 48 kHz ────────────────────────────────────────────────────

def generate_rir_gpu(out_dir: Path) -> np.ndarray:
    """GPU ISM + diffuse tail via gpuRIR, 48 kHz, single channel."""
    import gpuRIR
    print("\n[3] gpuRIR (GPU ISM, 48 kHz) …")
    beta   = gpuRIR.beta_SabineEstimation(ROOM_SZ, T60)
    nb_img = gpuRIR.t2n(T60, ROOM_SZ)
    h_all  = gpuRIR.simulateRIR(
        room_sz=ROOM_SZ, beta=beta,
        pos_src=SRC_POS[np.newaxis, :],
        pos_rcv=MIC_POS[np.newaxis, :],
        nb_img=nb_img, Tmax=T_MAX, fs=SR,
        Tdiff=T60, c=C_SOUND,
    )
    h = h_all[0, 0, :].astype(np.float32)
    save_wav(out_dir / "synthetic_rir_gpurir.wav", h)
    return h


# ── step 4: training-matched 16 kHz RIRs ─────────────────────────────────────

def generate_rir_rg_16k(out_dir: Path) -> np.ndarray:
    """rir-generator at 16 kHz — matches dataset/gen_rir.py parameters."""
    import rir_generator as rir
    print("\n[4a] rir-generator (16 kHz, training-matched) …")
    h = rir.generate(
        c=C_SOUND, fs=SR_TRAIN,
        r=MIC_POS[np.newaxis, :],
        s=SRC_POS, L=ROOM_SZ,
        reverberation_time=T60,
        nsample=int(T_MAX_TRAIN * SR_TRAIN),
        mtype=rir.mtype.omnidirectional,
        order=-1, dim=3, hp_filter=True,
    )
    h = np.array(h, dtype=np.float32).flatten()
    save_wav(out_dir / "synthetic_rir_rir_generator_16k.wav", h, sr=SR_TRAIN)
    return h


def generate_rir_gpu_16k(out_dir: Path) -> np.ndarray:
    """gpuRIR at 16 kHz, no diffuse tail — matches dataset/gen_rir.py parameters."""
    import gpuRIR
    print("\n[4b] gpuRIR (16 kHz, training-matched, no diffuse tail) …")
    beta   = gpuRIR.beta_SabineEstimation(ROOM_SZ, T60)
    Tmax   = gpuRIR.att2t_SabineEstimator(60.0, T60)
    nb_img = gpuRIR.t2n(Tmax, ROOM_SZ)
    h_all  = gpuRIR.simulateRIR(
        room_sz=ROOM_SZ, beta=beta,
        pos_src=SRC_POS[np.newaxis, :],
        pos_rcv=MIC_POS[np.newaxis, :],
        nb_img=nb_img, Tmax=Tmax,
        Tdiff=None, fs=SR_TRAIN, c=C_SOUND,
    )
    h = h_all[0, 0, :].astype(np.float32)
    save_wav(out_dir / "synthetic_rir_gpurir_16k.wav", h, sr=SR_TRAIN)
    return h


# ── step 5: comparison plot ───────────────────────────────────────────────────

def plot_comparison(ace: np.ndarray, rg: np.ndarray, gpu: np.ndarray,
                    out_dir: Path) -> None:
    print("\n[5] Plotting comparison …")
    n = min(len(ace), len(rg), len(gpu))
    ace, rg, gpu = ace[:n].astype(np.float64), rg[:n].astype(np.float64), gpu[:n].astype(np.float64)
    t = np.arange(n) / SR * 1000
    freqs = np.fft.rfftfreq(n, 1.0 / SR)

    def fft_db(x):
        X = np.abs(np.fft.rfft(x, n=n))
        return 20 * np.log10(X / (X.max() + 1e-12))

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#12122a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334466")
        ax.tick_params(colors="#888899", labelsize=8)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("white")

    panels = [
        ("ACE (real)",    ace, "#22ddaa"),
        ("rir-generator", rg,  "#ffaa33"),
        ("gpuRIR",        gpu, "#cc77ff"),
    ]
    for col, (lbl, h, c) in enumerate(panels):
        axes[0, col].plot(t, h, color=c, lw=0.6, alpha=0.9)
        axes[0, col].set_title(f"{lbl}  (Ch1)", fontsize=10)
        axes[0, col].set_xlabel("time (ms)")
        axes[0, col].set_ylabel("amplitude" if col == 0 else "")
        axes[0, col].axhline(0, color="#334466", lw=0.5)

        db = fft_db(h)
        axes[1, col].semilogx(freqs[1:], db[1:], color=c, lw=0.8, alpha=0.9)
        axes[1, col].set_xlim(100, SR / 2)
        axes[1, col].set_ylim(-80, 5)
        axes[1, col].set_xlabel("frequency (Hz)")
        axes[1, col].set_ylabel("magnitude (dB)" if col == 0 else "")
        axes[1, col].axhline(-60, color="#444466", lw=0.5, ls="--")

    fig.suptitle(
        f"RIR comparison — Office 2 / pos 1, Lin8Ch Ch1\n"
        f"room {ROOM_SZ[0]}×{ROOM_SZ[1]}×{ROOM_SZ[2]} m,  "
        f"src {tuple(SRC_POS)},  mic {tuple(MIC_POS)},  T60={T60} s",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = out_dir / "compare_rirs.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


# ── step 6: CSV ───────────────────────────────────────────────────────────────

def write_csv(out_dir: Path, rir_paths: list[Path]) -> None:
    csv_path = out_dir / "but_matched.csv"
    fieldnames = [
        "rir_path", "angle_deg", "radius_m",
        "room", "spk_setup", "mic_id",
        "room_depth", "room_width", "room_height",
        "rt60", "mic_depth", "mic_width", "mic_height",
        "nearest_wall", "mic_wall_dist",
    ]
    rows = [{
        "rir_path":      str(p.resolve()),
        "angle_deg":     ANGLE_DEG,
        "radius_m":      RADIUS_M,
        "room":          "Office_2",
        "spk_setup":     "pos_1",
        "mic_id":        "lin8ch_ch1_16k" if "16k" in p.name else "lin8ch_ch1",
        "room_depth":    round(_L, 2),
        "room_width":    round(_W, 2),
        "room_height":   round(_H, 2),
        "rt60":          T60,
        "mic_depth":     MIC_DEPTH,
        "mic_width":     MIC_WIDTH,
        "mic_height":    MIC_HEIGHT,
        "nearest_wall":  "depth0",
        "mic_wall_dist": MIC_DEPTH,
    } for p in rir_paths]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[CSV] {csv_path}  ({len(rows)} entries, angle={ANGLE_DEG}°)")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACE — Office 2 / pos 1, Lin8Ch Ch1")
    print("=" * 60)
    print(f"Room:    {ROOM_SZ} m  (L×W×H)")
    print(f"Source:  {SRC_POS} m")
    print(f"Mic Ch1: {MIC_POS} m  (dist to x=0 wall: {MIC_DEPTH} m)")
    print(f"Angle:   {ANGLE_DEG}°,  radius: {RADIUS_M} m")
    print(f"T60:     {T60} s,  SR: {SR} Hz")

    ace_ch1 = copy_ace_files(OUT_DIR)
    rg_rir  = generate_rir_rg(OUT_DIR)
    gpu_rir = generate_rir_gpu(OUT_DIR)
    generate_rir_rg_16k(OUT_DIR)
    generate_rir_gpu_16k(OUT_DIR)

    write_csv(OUT_DIR, [
        OUT_DIR / "ace_rir_ch1.wav",
        OUT_DIR / "synthetic_rir_rir_generator.wav",
        OUT_DIR / "synthetic_rir_gpurir.wav",
        OUT_DIR / "synthetic_rir_rir_generator_16k.wav",
        OUT_DIR / "synthetic_rir_gpurir_16k.wav",
    ])

    plot_comparison(ace_ch1, rg_rir, gpu_rir, OUT_DIR)

    print("\nDone. Files in", OUT_DIR)
    for p in sorted(OUT_DIR.glob("*")):
        print(f"  {p.name}")
