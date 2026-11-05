#!/usr/bin/env python3
"""Generate dense gpuRIR twins for Arni near-wall geometry.

1) Exact twins of measured samples under --measured-root (optional filter by
   held-in speakers).
2) Dense random sources in front of R1–R5 wall mics in the Arni room, with RT60
   drawn from the measured absorber table. Written as labeled npz ready for mix
   training.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset.gen_rir import beta_SabineEstimation, generate_rir_gpu  # noqa: E402
from dataset.rir_corrupt import peak_cut_dp  # noqa: E402
from eval_but import compute_angle_radius_rotated  # noqa: E402
from prepare_arni_mckenzie import (  # noqa: E402
    RECEIVER_XY,
    RT60_BY_ABSORBERS,
    ROOM_LX,
    ROOM_LY,
    ROOM_LZ,
    nearest_wall,
    rotate_to_y0,
)

ABS_WEIGHT_LIMS = [(0.5, 1.0)] * 6
T60_ERROR_TOL = 0.05
MAX_ABS_TRIES = 200


def solve_beta(room_sz: np.ndarray, rt60: float, rng: np.random.Generator):
    best = None
    for _ in range(MAX_ABS_TRIES):
        weights = [float(rng.uniform(lo, hi)) for lo, hi in ABS_WEIGHT_LIMS]
        beta, err = beta_SabineEstimation(room_sz, rt60, abs_weights=weights)
        if best is None or err < best[1]:
            best = (beta, err)
        if err <= T60_ERROR_TOL:
            return beta, err
    return best


def simulate(room_sz, pos_src, pos_rcv, rt60, fs, rng):
    beta, err = solve_beta(room_sz, rt60, rng)
    rir = generate_rir_gpu(
        room_sz=room_sz,
        pos_src=pos_src[None, :],
        pos_rcv=pos_rcv[None, :],
        RT60=rt60,
        fs=fs,
        att_diff=None,
        beta=beta,
    )
    wav = np.asarray(rir[0, 0, :], dtype=np.float64)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = wav / peak
    return wav.astype(np.float32), float(err)


def _pack_npz(path: Path, wav: np.ndarray, rot: dict, fs: int, angle: float, radius: float):
    import torch

    t = torch.from_numpy(wav)
    dp = peak_cut_dp(t, fs)
    room_sz = np.array(
        [float(rot["room_sz_x"]), float(rot["room_sz_y"]), float(rot["room_sz_z"])],
        dtype=np.float64,
    )
    pos_rcv = np.array(
        [[float(rot["mic_x"]), float(rot["mic_y"]), float(rot["mic_z"])]],
        dtype=np.float32,
    )
    pos_src = np.array(
        [[float(rot["source_x"]), float(rot["source_y"]), float(rot["source_z"])]],
        dtype=np.float32,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fs=np.int64(fs),
        RT60=np.float64(rot["RT60"]),
        room_sz=room_sz,
        pos_src=pos_src,
        pos_rcv=pos_rcv,
        pos_noise=np.zeros((0, 3), dtype=np.float32),
        rir=wav.astype(np.float32)[None, None, :],
        rir_dp=dp.detach().cpu().numpy().astype(np.float32)[None, None, :],
        angle_deg=np.float64(angle),
        radius_m=np.float64(radius),
    )


def gen_exact_twins(
    measured_root: Path,
    out_wav_root: Path,
    out_npz_dir: Path,
    speakers: set[str],
    fs: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    metas = sorted(measured_root.rglob("rotated_metadata.json"))
    for meta_path in metas:
        sample_dir = meta_path.parent
        spk = sample_dir.name
        if speakers and spk not in speakers:
            continue
        meta = json.loads(meta_path.read_text())
        rot = meta["rotated"]
        room_sz = np.array(
            [rot["room_sz_x"], rot["room_sz_y"], rot["room_sz_z"]], dtype=np.float64
        )
        pos_rcv = np.array([rot["mic_x"], rot["mic_y"], rot["mic_z"]], dtype=np.float64)
        pos_src = np.array(
            [rot["source_x"], rot["source_y"], rot["source_z"]], dtype=np.float64
        )
        pos_rcv = np.clip(pos_rcv, 0.01, room_sz - 0.01)
        pos_src = np.clip(pos_src, 0.01, room_sz - 0.01)
        wav, err = simulate(room_sz, pos_src, pos_rcv, float(rot["RT60"]), fs, rng)
        rel = sample_dir.relative_to(measured_root)
        wav_dir = out_wav_root / "exact" / rel / "RIR"
        wav_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_dir / "twin.wav"), wav, fs)
        out_meta = dict(meta)
        out_meta["twin"] = {
            "variant": "exact",
            "t60_fit_error": err,
            "source": "gpuRIR",
        }
        (out_wav_root / "exact" / rel / "rotated_metadata.json").write_text(
            json.dumps(out_meta, indent=2) + "\n"
        )
        angle, radius = compute_angle_radius_rotated(
            float(rot["source_x"]),
            float(rot["source_y"]),
            float(rot["mic_x"]),
            float(rot["mic_y"]),
        )
        npz_name = f"arni_twin_exact_{rel.as_posix().replace('/', '_')}.npz"
        npz_path = out_npz_dir / npz_name
        _pack_npz(npz_path, wav, rot, fs, angle, radius)
        rows.append(
            {
                "npz": str(npz_path.resolve()),
                "angle_deg": angle,
                "radius_m": radius,
                "kind": "exact_twin",
            }
        )
    return rows


def gen_dense(
    out_npz_dir: Path,
    n: int,
    receivers: list[str],
    fs: int,
    seed: int,
    min_radius: float,
    max_radius: float,
) -> list[dict]:
    """Sample sources in the half-plane in front of the south wall (y > mic_y)."""
    rng = np.random.default_rng(seed)
    rt_vals = np.array([float(v) for v in RT60_BY_ABSORBERS.values()], dtype=np.float64)
    rows = []
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rcv_name = receivers[int(rng.integers(0, len(receivers)))]
        mx0, my0 = RECEIVER_XY[rcv_name]
        mz0 = 1.5
        # Keep south-wall placement; rotate is identity for y0.
        wall, _ = nearest_wall(mx0, my0, ROOM_LX, ROOM_LY)
        assert wall == "y0", f"expected y0 for {rcv_name}, got {wall}"
        mx, my, mz, _, _, _, Lx, Ly, _, _ = rotate_to_y0(
            mx0, my0, mz0, mx0, my0 + 1.0, mz0, wall, ROOM_LX, ROOM_LY
        )
        # Sample angle in [0,180] folded frame: atan2(dy,dx) then fold — generate
        # sources with dy>0 (into room) and dx spanning left/right.
        ang = float(rng.uniform(5.0, 175.0))  # avoid exactly along wall
        rad = float(rng.uniform(min_radius, max_radius))
        # In training frame after y0 rotation: mic at (mx, my), +y into room.
        # angle_deg = atan2(dy, dx) folded to [0,180]; invert for a source with
        # that folded angle by placing in upper half-plane.
        # Use unsigned angle from +x: place with dy = rad*sin(theta), dx = rad*cos(theta)
        # where theta in (0, pi).
        theta = math.radians(ang)
        sx = mx + rad * math.cos(theta)
        sy = my + rad * math.sin(theta)
        sz = float(rng.uniform(1.2, 1.8))
        # Reject if outside room with margin.
        if not (0.2 < sx < Lx - 0.2 and 0.2 < sy < Ly - 0.2):
            # resample radially inward
            for _ in range(20):
                rad = float(rng.uniform(min_radius, max_radius))
                sx = mx + rad * math.cos(theta)
                sy = my + rad * math.sin(theta)
                if 0.2 < sx < Lx - 0.2 and 0.2 < sy < Ly - 0.2:
                    break
            else:
                continue
        rt60 = float(rng.choice(rt_vals))
        # Slight RT60 jitter
        rt60 = float(np.clip(rt60 + rng.normal(0, 0.03), 0.2, 1.4))
        room_sz = np.array([Lx, Ly, ROOM_LZ], dtype=np.float64)
        pos_rcv = np.array([mx, my, mz], dtype=np.float64)
        pos_src = np.array([sx, sy, sz], dtype=np.float64)
        wav, err = simulate(room_sz, pos_src, pos_rcv, rt60, fs, rng)
        rot = {
            "RT60": rt60,
            "room_sz_x": Lx,
            "room_sz_y": Ly,
            "room_sz_z": ROOM_LZ,
            "mic_x": mx,
            "mic_y": my,
            "mic_z": mz,
            "source_x": sx,
            "source_y": sy,
            "source_z": sz,
        }
        angle, radius = compute_angle_radius_rotated(sx, sy, mx, my)
        npz_path = out_npz_dir / f"arni_dense_{i:05d}_{rcv_name}.npz"
        _pack_npz(npz_path, wav, rot, fs, angle, radius)
        rows.append(
            {
                "npz": str(npz_path.resolve()),
                "angle_deg": angle,
                "radius_m": radius,
                "kind": "dense_twin",
                "t60_fit_error": err,
                "mic": rcv_name,
            }
        )
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  dense {i + 1}/{n}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--measured-root",
        type=Path,
        default=Path("/storage/reie/data/arni-mckenzie-nearwall"),
    )
    p.add_argument("--out-wav-root", type=Path, default=Path("/tmp/arni-twins"))
    p.add_argument("--out-npz-dir", type=Path, default=Path("/tmp/arni-twins-npz"))
    p.add_argument("--train-speakers", default="S1,S2", help="Speakers used for exact twins.")
    p.add_argument("--dense-n", type=int, default=5000)
    p.add_argument("--receivers", default="R1,R2,R3,R4,R5")
    p.add_argument("--min-radius", type=float, default=0.5)
    p.add_argument("--max-radius", type=float, default=5.0)
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--seed", type=int, default=2033)
    args = p.parse_args()

    speakers = {x.strip() for x in args.train_speakers.split(",") if x.strip()}
    receivers = [x.strip() for x in args.receivers.split(",") if x.strip()]
    args.out_npz_dir.mkdir(parents=True, exist_ok=True)
    args.out_wav_root.mkdir(parents=True, exist_ok=True)

    print("=== exact twins ===", flush=True)
    exact = gen_exact_twins(
        args.measured_root,
        args.out_wav_root,
        args.out_npz_dir,
        speakers,
        args.fs,
        args.seed,
    )
    print(f"exact twins: {len(exact)}", flush=True)

    print("=== dense twins ===", flush=True)
    dense = gen_dense(
        args.out_npz_dir,
        args.dense_n,
        receivers,
        args.fs,
        args.seed + 17,
        args.min_radius,
        args.max_radius,
    )
    print(f"dense twins: {len(dense)}", flush=True)

    manifest = {
        "n_exact": len(exact),
        "n_dense": len(dense),
        "train_speakers": sorted(speakers),
        "receivers": receivers,
        "npz_dir": str(args.out_npz_dir),
        "wav_root": str(args.out_wav_root),
        "exact": exact,
        "dense": dense,
    }
    (args.out_npz_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_exact", "n_dense", "npz_dir")}, indent=2))


if __name__ == "__main__":
    main()
