#!/usr/bin/env python3
"""
gen_but_synthetic_rirs.py — Generate gpuRIR synthetic RIRs that match the
BUT ReverbDB geometry exactly.

For each row in but_matched.csv the script:
  1. Reads speaker position from BUT spk_meta.txt
  2. Generates a synthetic RIR with gpuRIR using the real room size,
     mic position, speaker position, and RT60 — identical to the training
     pipeline (same beta_SabineEstimation, no Tdiff, fs=16 kHz)
  3. Saves the WAV to <out_dir>/wavs/<room>/<spk_setup>/mic<mic_id>.wav
  4. Writes <out_dir>/but_synthetic.csv with the same columns as
     but_matched.csv but pointing to the synthetic WAV files

Usage (from project root):
    python scripts/gen_but_synthetic_rirs.py \\
        --but_csv  config/but_matched.csv \\
        --but_root /storage/data/but \\
        --out_dir  results/but_synthetic_rirs
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project root to sys.path so dataset/ imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import gpuRIR
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(False)
except ImportError as e:
    sys.exit(f"gpuRIR not available: {e}")

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# gpuRIR helper
# ---------------------------------------------------------------------------

def _estimate_minimal_rt60(room_sz: np.ndarray) -> float:
    V = float(np.prod(room_sz))
    S = 2.0 * (
        room_sz[0] * room_sz[1]
        + room_sz[0] * room_sz[2]
        + room_sz[1] * room_sz[2]
    )
    return 0.161 * V / S


def generate_rir(
    room_sz: np.ndarray,   # [Lx, Ly, Lz] in metres
    pos_src: np.ndarray,   # [1, 3]
    pos_rcv: np.ndarray,   # [1, 3]
    rt60: float,
    fs: int = 16_000,
) -> np.ndarray:
    """Simulate a single RIR with gpuRIR, matching the training pipeline."""
    rt60_min = _estimate_minimal_rt60(room_sz)
    if rt60 <= rt60_min:
        warnings.warn(
            f"RT60={rt60:.3f}s is at or below minimal achievable "
            f"{rt60_min:.3f}s for room {room_sz}.  Clamping to min+0.05."
        )
        rt60 = rt60_min + 0.05

    Tmax = gpuRIR.att2t_SabineEstimator(60.0, rt60)
    nb_img = gpuRIR.t2n(Tmax, room_sz)
    beta = gpuRIR.beta_SabineEstimation(room_sz, rt60)

    rir = gpuRIR.simulateRIR(
        room_sz=room_sz,
        beta=beta,
        pos_src=pos_src,
        pos_rcv=pos_rcv,
        nb_img=nb_img,
        Tmax=Tmax,
        Tdiff=None,
        fs=fs,
    )
    # shape: [n_src=1, n_mic=1, T]  →  [T]
    return rir[0, 0, :]


# ---------------------------------------------------------------------------
# BUT metadata helpers
# ---------------------------------------------------------------------------

def parse_spk_meta(spk_meta_path: Path) -> np.ndarray:
    """Return speaker position [depth, width, height] from spk_meta.txt."""
    fields: dict[str, str] = {}
    with open(spk_meta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("$"):
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    fields[parts[0].lstrip("$")] = parts[1].strip()
    d = float(fields["EnvSpk1Depth"])
    w = float(fields["EnvSpk1Width"])
    h = float(fields["EnvSpk1Height"])
    return np.array([d, w, h], dtype=np.float64)


def _spk_meta_path(rir_path: str, but_root: Path) -> Path:
    """Derive spk_meta.txt path from a rir_path string.

    rir_path looks like:
        /storage/data/but/<room>/<MicSetup>/<spk_setup>/<mic_id>/RIR/...wav
    spk_meta.txt is at:
        <but_root>/<room>/<MicSetup>/<spk_setup>/spk_meta.txt
    """
    p = Path(rir_path)
    # Walk up: .../RIR/<wav> → mic_id dir → spk_setup dir → MicSetup → room
    spk_setup_dir = p.parent.parent.parent      # <room>/<MicSetup>/<spk_setup>
    meta = spk_setup_dir / "spk_meta.txt"
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--but_csv",  type=Path,
                    default=Path("config/but_matched.csv"),
                    help="Input BUT matched CSV (default: config/but_matched.csv)")
    ap.add_argument("--out_dir",  type=Path,
                    default=Path("results/but_synthetic_rirs"),
                    help="Output directory (default: results/but_synthetic_rirs)")
    ap.add_argument("--fs",       type=int,   default=16_000,
                    help="Sample rate in Hz (default: 16000)")
    args = ap.parse_args(argv)

    but_csv:  Path = args.but_csv.expanduser().absolute()
    out_dir:  Path = args.out_dir.expanduser().absolute()
    fs:       int  = args.fs

    if not but_csv.exists():
        sys.exit(f"CSV not found: {but_csv}")

    wav_root = out_dir / "wavs"
    wav_root.mkdir(parents=True, exist_ok=True)

    with open(but_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows)} entries from {but_csv}")

    out_rows: list[dict] = []

    for i, row in enumerate(rows):
        rir_path_str = row["rir_path"]
        room      = row["room"]
        spk_setup = row["spk_setup"]
        mic_id    = row["mic_id"]
        rt60      = float(row["rt60"])

        room_sz = np.array([
            float(row["room_depth"]),
            float(row["room_width"]),
            float(row["room_height"]),
        ], dtype=np.float64)

        pos_rcv = np.array([[
            float(row["mic_depth"]),
            float(row["mic_width"]),
            float(row["mic_height"]),
        ]], dtype=np.float64)

        # Speaker position from BUT metadata
        spk_meta = _spk_meta_path(rir_path_str, but_root=Path("/storage/data/but"))
        if not spk_meta.exists():
            sys.exit(f"spk_meta.txt not found: {spk_meta}")
        pos_spk = parse_spk_meta(spk_meta)
        pos_src = pos_spk[np.newaxis, :]          # [1, 3]

        print(
            f"[{i+1:2d}/{len(rows)}] {spk_setup}/mic{mic_id}  "
            f"room={room_sz}  src={pos_spk}  rcv={pos_rcv[0]}  RT60={rt60:.3f}s",
            end="  ",
        )

        rir = generate_rir(room_sz, pos_src, pos_rcv, rt60, fs=fs)
        print(f"len={len(rir)/fs:.2f}s")

        # Normalise peak to ±1
        peak = np.abs(rir).max()
        if peak > 0:
            rir = rir / peak

        # Save WAV  (room/spk_setup/mic<id>.wav)
        wav_dir = wav_root / room / spk_setup
        wav_dir.mkdir(parents=True, exist_ok=True)
        wav_path = wav_dir / f"mic{mic_id}_synthetic.wav"
        sf.write(str(wav_path), rir.astype(np.float32), fs, subtype="PCM_16")

        out_row = dict(row)
        out_row["rir_path"] = str(wav_path)
        out_rows.append(out_row)

    # Write new CSV
    out_csv = out_dir / "but_synthetic.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} synthetic RIRs.")
    print(f"WAVs:    {wav_root}")
    print(f"CSV:     {out_csv}")


if __name__ == "__main__":
    main()
