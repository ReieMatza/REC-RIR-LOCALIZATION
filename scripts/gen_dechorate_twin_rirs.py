#!/usr/bin/env python3
"""Generate gpuRIR acoustic twins from a rotated dEchorate/BUT-style tree.

Each input sample must contain ``rotated_metadata.json`` and is recreated with
the same rotated room dimensions, mic/source coordinates, and RT60. The output
mirrors the input tree, replacing the measured RIR wav with ``RIR/twin.wav``.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from dataset.gen_rir import beta_SabineEstimation, generate_rir_gpu

warnings.filterwarnings("ignore")

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


def simulate(
    room_sz: np.ndarray,
    pos_src: np.ndarray,
    pos_rcv: np.ndarray,
    rt60: float,
    fs: int,
    rng: np.random.Generator,
):
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
    return np.asarray(rir[0, 0, :], dtype=np.float64), err


def variant_geometry(rot: Dict, variant: str) -> Dict:
    out = dict(rot)
    if variant == "exact":
        return out
    if variant == "rt60":
        out["RT60"] = 0.8
        out["RT60_source"] = "training_midpoint"
        return out
    if variant == "heights":
        out["mic_z"] = 2.0
        out["source_z"] = 1.5
        return out
    raise ValueError(f"unknown variant {variant!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, help="Rotated metadata tree to twin.")
    p.add_argument("--out-root", required=True, help="Output root; variant subdirs are created below it.")
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("variants", nargs="*", default=["exact"], choices=("exact", "rt60", "heights"))
    args = p.parse_args()

    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    meta_paths = sorted(input_root.rglob("rotated_metadata.json"))
    if not meta_paths:
        raise SystemExit(f"no rotated_metadata.json under {input_root}")
    print(f"{len(meta_paths)} samples under {input_root}", flush=True)

    for variant in args.variants:
        rng = np.random.default_rng(args.seed)
        root = out_root / variant
        print(f"\n===== {variant} -> {root} =====", flush=True)
        worst: List[tuple[float, str]] = []

        for i, meta_path in enumerate(meta_paths):
            sample_dir = meta_path.parent
            rel = sample_dir.relative_to(input_root)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rot = variant_geometry(meta["rotated"], variant)

            room_sz = np.array(
                [rot["room_sz_x"], rot["room_sz_y"], rot["room_sz_z"]],
                dtype=np.float64,
            )
            pos_rcv = np.array([rot["mic_x"], rot["mic_y"], rot["mic_z"]], dtype=np.float64)
            pos_src = np.array(
                [rot["source_x"], rot["source_y"], rot["source_z"]],
                dtype=np.float64,
            )
            rt60 = float(rot["RT60"])

            pos_rcv = np.clip(pos_rcv, 0.01, room_sz - 0.01)
            pos_src = np.clip(pos_src, 0.01, room_sz - 0.01)

            rir, err = simulate(room_sz, pos_src, pos_rcv, rt60, args.fs, rng)
            peak = np.max(np.abs(rir))
            if peak > 0:
                rir = rir / peak
            worst.append((float(err), str(rel)))

            wav_dir = root / rel / "RIR"
            wav_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav_dir / "twin.wav"), rir.astype(np.float32), args.fs)

            out_meta = dict(meta)
            out_meta["rotated"] = rot
            out_meta["twin"] = {
                "variant": variant,
                "source": "gpuRIR image-source, training pipeline",
                "t60_fit_error": float(err),
                "measured_from": str(meta_path),
                "gpuRIR_T60": float(rt60),
            }
            (root / rel / "rotated_metadata.json").write_text(
                json.dumps(out_meta, indent=2),
                encoding="utf-8",
            )

            if (i + 1) % 25 == 0 or i + 1 == len(meta_paths):
                print(f"  {i + 1}/{len(meta_paths)}", flush=True)

        worst.sort(reverse=True)
        bad = [item for item in worst if item[0] > T60_ERROR_TOL]
        errs = [item[0] for item in worst]
        print(
            f"  RT60 fit error: median {np.median(errs):.4f}, "
            f"{len(bad)} of {len(worst)} above {T60_ERROR_TOL}",
            flush=True,
        )
        for err, rel in worst[:5]:
            print(f"    {err:.4f}  {rel}", flush=True)


if __name__ == "__main__":
    main()
