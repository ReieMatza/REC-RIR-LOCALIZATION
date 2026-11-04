#!/usr/bin/env python3
"""Prepare McKenzie/Arni 6DoF SRIR dataset as a BUT-style eval tree.

Paper: arXiv:2111.11882  |  Zenodo: 10.5281/zenodo.5720724

Each absorption SOFA packs M=21 measurements (7 receivers × 3 sources) with
16 Ambisonic channels (Zylia 3rd-order). First iteration extracts W (ch 0),
resamples to 16 kHz, rotates nearest wall → y=0, and writes:

  <out>/<room>/<mic>/<src>/rotated_metadata.json
  <out>/<room>/<mic>/<src>/RIR/rir.wav

Default keeps near-wall receivers R1–R5 only (≈1.3 m from south wall).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import soundfile as sf

ROOM_LX = 8.9
ROOM_LY = 6.3
ROOM_LZ = 3.6

# Nominal IDs from the paper; SOFA xyz overrides when present.
# Paper Table I lists R5 at x=6.0; the released SOFA uses x=6.5.
RECEIVER_XY = {
    "R1": (1.5, 1.3),
    "R2": (2.75, 1.3),
    "R3": (4.0, 1.3),
    "R4": (5.25, 1.3),
    "R5": (6.5, 1.3),  # matches released SOFA
    "R6": (2.75, 4.745),
    "R7": (5.25, 4.745),
}
SOURCE_XY = {
    "S1": (1.5, 2.8),
    "S2": (5.0, 2.8),
    "S3": (3.5, 4.245),
}

RT60_BY_ABSORBERS = {
    "100": 0.32,
    "75": 0.40,
    "50": 0.54,
    "25": 0.76,
    "0": 1.22,
}

TARGET_SR = 16000


def nearest_wall(x: float, y: float, Lx: float, Ly: float) -> Tuple[str, float]:
    d = {"x0": x, "xL": Lx - x, "y0": y, "yL": Ly - y}
    wall = min(d, key=d.get)
    return wall, d[wall]


def rotate_to_y0(mx, my, mz, sx, sy, sz, wall, Lx, Ly):
    if wall == "y0":
        return mx, my, mz, sx, sy, sz, Lx, Ly, "nearest_wall_y0", 0
    if wall == "yL":
        return Lx - mx, Ly - my, mz, Lx - sx, Ly - sy, sz, Lx, Ly, "nearest_wall_yL", 180
    if wall == "x0":
        return Ly - my, mx, mz, Ly - sy, sx, sz, Ly, Lx, "nearest_wall_x0", 90
    if wall == "xL":
        return my, Lx - mx, mz, sy, Lx - sx, sz, Ly, Lx, "nearest_wall_xL", 270
    raise ValueError(wall)


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float64).ravel()
    if int(sr_in) == int(sr_out):
        return wav
    try:
        import librosa

        return librosa.resample(wav, orig_sr=int(sr_in), target_sr=int(sr_out))
    except Exception:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        y = AF.resample(t, int(sr_in), int(sr_out))
        return y.squeeze(0).numpy().astype(np.float64)


def _peak_norm(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m <= 0:
        return x.astype(np.float32)
    return (x / m).astype(np.float32)


def _parse_absorbers(name: str) -> Optional[str]:
    m = re.search(r"(\d+)\s*percent", name.lower())
    return m.group(1) if m else None


def _match_id(xy: np.ndarray, catalog: Dict[str, Tuple[float, float]], tol: float = 0.15) -> Optional[str]:
    best_n, best_d = None, 1e9
    for n, (x, y) in catalog.items():
        d = math.hypot(float(xy[0]) - x, float(xy[1]) - y)
        if d < best_d:
            best_d, best_n = d, n
    return best_n if best_d <= tol else None


def prepare(
    zip_path: Path,
    out_dir: Path,
    receivers: Sequence[str],
    absorbers: Sequence[str],
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_rcv = {r.upper() for r in receivers}
    keep_abs = {a for a in absorbers}
    stage = out_dir / "_sofa_stage"
    stage.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    skip_reasons: Dict[str, int] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".sofa")]
        if not members:
            raise SystemExit(f"no .sofa files in {zip_path}")

        for name in sorted(members):
            abs_key = _parse_absorbers(name)
            if abs_key is None or abs_key not in keep_abs:
                n_skip += 1
                skip_reasons["absorbers"] = skip_reasons.get("absorbers", 0) + 1
                continue

            dest = stage / Path(name).name
            if not dest.is_file():
                with zf.open(name) as src_f, open(dest, "wb") as out_f:
                    out_f.write(src_f.read())

            with h5py.File(dest, "r") as f:
                ir = np.asarray(f["Data.IR"], dtype=np.float64)  # (M, R, N)
                fs = int(np.asarray(f["Data.SamplingRate"]).reshape(-1)[0])
                lp = np.asarray(f["ListenerPosition"], dtype=np.float64)
                sp = np.asarray(f["SourcePosition"], dtype=np.float64)

            if ir.ndim != 3:
                raise SystemExit(f"unexpected IR shape {ir.shape} in {name}")

            for mi in range(ir.shape[0]):
                rcv = _match_id(lp[mi, :2], RECEIVER_XY)
                src = _match_id(sp[mi, :2], SOURCE_XY)
                if rcv is None or src is None:
                    n_skip += 1
                    skip_reasons["id_match"] = skip_reasons.get("id_match", 0) + 1
                    continue
                if rcv not in keep_rcv:
                    n_skip += 1
                    skip_reasons["receiver"] = skip_reasons.get("receiver", 0) + 1
                    continue

                # Use measured SOFA xyz (more accurate than paper table).
                mx, my, mz = map(float, lp[mi])
                sx, sy, sz = map(float, sp[mi])
                wall, dmic = nearest_wall(mx, my, ROOM_LX, ROOM_LY)
                mx2, my2, mz2, sx2, sy2, sz2, Lx2, Ly2, mode, deg = rotate_to_y0(
                    mx, my, mz, sx, sy, sz, wall, ROOM_LX, ROOM_LY
                )

                w = ir[mi, 0, :]  # Ambisonic W
                wav = _peak_norm(_resample(w, fs, TARGET_SR))

                room_name = f"Arni_abs{abs_key}"
                sample_dir = out_dir / room_name / rcv / src
                rir_dir = sample_dir / "RIR"
                rir_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(rir_dir / "rir.wav"), wav, TARGET_SR)

                rt60 = float(RT60_BY_ABSORBERS[abs_key])
                vol = float(ROOM_LX * ROOM_LY * ROOM_LZ)
                meta = {
                    "dataset": "arni_mckenzie_2021",
                    "environment": {
                        "env_id": f"abs{abs_key}",
                        "env_name": room_name,
                        "env_type": "variable_acoustics_lab",
                    },
                    "rotation": {
                        "mode": mode,
                        "nearest_wall": wall,
                        "degrees": deg,
                        "handedness": "proper_rotation",
                    },
                    "original": {
                        "room": {
                            "width": ROOM_LX,
                            "depth": ROOM_LY,
                            "height": ROOM_LZ,
                            "volume": vol,
                        },
                        "source": {"width": sx, "depth": sy, "height": sz},
                        "mic": {"width": mx, "depth": my, "height": mz},
                    },
                    "rotated": {
                        "RT60": rt60,
                        "RT60_source": f"paper_table2_1000Hz_abs{abs_key}",
                        "room_sz_x": Lx2,
                        "room_sz_y": Ly2,
                        "room_sz_z": ROOM_LZ,
                        "volume": vol,
                        "source_x": sx2,
                        "source_y": sy2,
                        "source_z": sz2,
                        "mic_x": mx2,
                        "mic_y": my2,
                        "mic_z": mz2,
                    },
                    "scenario": {
                        "speaker": src,
                        "microphone": rcv,
                        "absorbers_percent": int(abs_key),
                        "array": "zylia_ZM1_SH",
                        "channel": "W",
                        "measurement_index": int(mi),
                        "src_type": "loudspeaker",
                    },
                    "acoustics_from_mic_meta": {
                        "RT20": rt60,
                        "rir_distance_m": float(math.hypot(sx - mx, sy - my)),
                        "d_wall_m": float(dmic),
                    },
                    "source_file": name,
                }
                (sample_dir / "rotated_metadata.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )
                n_ok += 1

    summary = {
        "n_ok": n_ok,
        "n_skip": n_skip,
        "skip_reasons": skip_reasons,
        "out_dir": str(out_dir),
        "receivers": sorted(keep_rcv),
        "absorbers": sorted(keep_abs, key=lambda x: int(x)),
        "room_sz": [ROOM_LX, ROOM_LY, ROOM_LZ],
        "channel": "W",
        "zip": str(zip_path),
        "note": "R5 x=6.5 from SOFA (paper table lists 6.0).",
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, default=Path("/tmp/arni-srir/6dof_SRIRs_zylia_SH.zip"))
    p.add_argument("--out-dir", type=Path, default=Path("/storage/reie/data/arni-mckenzie-nearwall"))
    p.add_argument("--receivers", default="R1,R2,R3,R4,R5")
    p.add_argument("--absorbers", default="0,25,50,75,100")
    args = p.parse_args()
    prepare(
        zip_path=args.zip,
        out_dir=args.out_dir,
        receivers=[x.strip() for x in args.receivers.split(",") if x.strip()],
        absorbers=[x.strip() for x in args.absorbers.split(",") if x.strip()],
    )


if __name__ == "__main__":
    main()
