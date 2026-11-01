#!/usr/bin/env python3
"""Pack dEchorate wavs as labeled npz and build a 50/50 mix list.

Hold out two rooms so mixed training is not scored on data it has already
seen. Default holdouts are the RT extremes: 000000 (shortest) and 011111
(longest). The original eval tree is not modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from dataset.rir_corrupt import peak_cut_dp  # noqa: E402
from eval_but import compute_angle_radius_rotated, latest_rir_wav  # noqa: E402

DEFAULT_HOLD = ("000000", "011111")


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _parse_csv_ints(text: str | None) -> set[int] | None:
    if text is None:
        return None
    vals = {int(x.strip()) for x in text.split(",") if x.strip()}
    return vals if vals else None


def _array_id(meta: dict, sample_dir: Path) -> int | None:
    scenario = meta.get("scenario") or {}
    raw = scenario.get("mic_setup_id")
    if raw is not None:
        return int(raw)
    mic = str(scenario.get("microphone") or sample_dir.parent.name)
    if mic.startswith("A") and "_M" in mic:
        return int(mic[1:].split("_M", 1)[0])
    return None


def _resample_mono(wav_path: Path, sr_out: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path.as_posix())
    if wav.numel() == 0:
        raise ValueError(f"empty wav: {wav_path}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != sr_out:
        wav = torchaudio.functional.resample(wav, int(sr), sr_out)
    return wav.squeeze(0).float()


def _pack_npz(wav_path: Path, meta: dict, out_path: Path, sr: int = 16000) -> tuple[float, float]:
    rot = meta["rotated"]
    rir = _resample_mono(wav_path, sr)
    peak = float(rir.abs().max())
    if peak > 0:
        rir = rir / peak
    dp = peak_cut_dp(rir, sr)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        fs=np.int64(sr),
        RT60=np.float64(rot["RT60"]),
        room_sz=room_sz,
        pos_src=pos_src,
        pos_rcv=pos_rcv,
        pos_noise=np.zeros((0, 3), dtype=np.float32),
        rir=rir.detach().cpu().numpy().astype(np.float32)[None, None, :],
        rir_dp=dp.detach().cpu().numpy().astype(np.float32)[None, None, :],
    )
    return compute_angle_radius_rotated(
        float(rot["source_x"]),
        float(rot["source_y"]),
        float(rot["mic_x"]),
        float(rot["mic_y"]),
    )


def _symlink_sample(src: Path, dst: Path) -> None:
    meta = src / "rotated_metadata.json"
    wav = latest_rir_wav(src)
    if not meta.is_file() or wav is None:
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    meta_link = dst / "rotated_metadata.json"
    if meta_link.exists() or meta_link.is_symlink():
        meta_link.unlink()
    meta_link.symlink_to(meta.resolve())
    rir_dir = dst / "RIR"
    rir_dir.mkdir(parents=True, exist_ok=True)
    wav_link = rir_dir / wav.name
    if wav_link.exists() or wav_link.is_symlink():
        wav_link.unlink()
    wav_link.symlink_to(wav.resolve())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/storage/reie/data/dechorate-arr01")
    p.add_argument("--npz-dir", default="/storage/reie/data/dechorate-arr01-npz")
    p.add_argument(
        "--sim-list",
        default=(
            "/storage/reie/REC-RIR-LOCALIZATION/config/"
            "rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w09-zfree-s2030/reie-train-rir.txt"
        ),
    )
    p.add_argument(
        "--sim-loc-csv",
        default=(
            "/storage/reie/data/rec-rir/"
            "rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w09-zfree-s2030/rir_localization.csv"
        ),
    )
    p.add_argument(
        "--out-dir",
        default="/storage/reie/REC-RIR-LOCALIZATION/config/rir-zfree-wideband-mix-dechorate",
    )
    p.add_argument(
        "--holdout-root",
        default="/storage/reie/data/dechorate-arr01-holdout",
    )
    p.add_argument("--hold-rooms", default=",".join(DEFAULT_HOLD))
    p.add_argument(
        "--train-arrays",
        default="",
        help="Comma-separated array IDs to include as real training RIRs. Empty means all non-held rows.",
    )
    p.add_argument(
        "--hold-arrays",
        default="",
        help="Comma-separated array IDs to symlink into the holdout/eval tree and exclude from train.",
    )
    p.add_argument("--mix-frac", type=float, default=0.5)
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild npz even when the file already exists.",
    )
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    holdout_root = Path(args.holdout_root)
    hold = {x.strip() for x in args.hold_rooms.split(",") if x.strip()}
    train_arrays = _parse_csv_ints(args.train_arrays)
    hold_arrays = _parse_csv_ints(args.hold_arrays)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for meta_path in sorted(dataset_root.rglob("rotated_metadata.json")):
        sample_dir = meta_path.parent
        wav = latest_rir_wav(sample_dir)
        if wav is None:
            continue
        meta = json.loads(meta_path.read_text())
        env = meta.get("environment") or {}
        scenario = meta.get("scenario") or {}
        arr = _array_id(meta, sample_dir)
        env_id = str(env.get("env_id") or "")
        env_name = str(env.get("env_name") or sample_dir.parent.parent.name)
        mic = str(scenario.get("microphone") or sample_dir.parent.name)
        spk = str(scenario.get("speaker") or sample_dir.name)
        npz_name = f"{env_name}_{mic}_{spk}.npz".replace("/", "_")
        npz_path = npz_dir / npz_name
        if args.force or not npz_path.is_file():
            angle_deg, radius_m = _pack_npz(wav, meta, npz_path)
        else:
            rot = meta["rotated"]
            angle_deg, radius_m = compute_angle_radius_rotated(
                float(rot["source_x"]),
                float(rot["source_y"]),
                float(rot["mic_x"]),
                float(rot["mic_y"]),
            )
        rows.append(
            {
                "env_id": env_id,
                "env_name": env_name,
                "array_id": arr,
                "holdout": env_id in hold or (hold_arrays is not None and arr in hold_arrays),
                "npz": os.path.normpath(npz_path.as_posix()),
                "angle_deg": angle_deg,
                "radius_m": radius_m,
                "sample_dir": sample_dir,
            }
        )

    if not rows:
        raise SystemExit(f"no dEchorate samples under {dataset_root}")

    train_rows = [
        r for r in rows
        if not r["holdout"] and (train_arrays is None or r["array_id"] in train_arrays)
    ]
    hold_rows = [r for r in rows if r["holdout"]]
    if not train_rows:
        raise SystemExit("hold-out left zero training rooms")
    if not hold_rows:
        raise SystemExit(f"no samples for hold-out rooms {sorted(hold)}")

    unique_train = [r["npz"] for r in train_rows]
    unique_all = [r["npz"] for r in rows]
    (out_dir / "dechorate-train-unique.txt").write_text("\n".join(unique_train) + "\n")
    (out_dir / "dechorate-all-unique.txt").write_text("\n".join(unique_all) + "\n")

    loc_path = out_dir / "rir_localization.csv"
    with loc_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rir_path", "angle_deg", "radius_m"])
        writer.writeheader()
        with open(args.sim_loc_csv, newline="") as sim_handle:
            for row in csv.DictReader(sim_handle):
                writer.writerow(
                    {
                        "rir_path": os.path.normpath(row["rir_path"]),
                        "angle_deg": row["angle_deg"],
                        "radius_m": row["radius_m"],
                    }
                )
        for row in train_rows:
            writer.writerow(
                {
                    "rir_path": row["npz"],
                    "angle_deg": f"{row['angle_deg']:.6f}",
                    "radius_m": f"{row['radius_m']:.6f}",
                }
            )

    sim = _read_lines(Path(args.sim_list))
    n_sim = len(sim)
    n_real = max(1, int(round(n_sim * args.mix_frac / max(1e-8, 1.0 - args.mix_frac))))
    reps = (n_real + len(unique_train) - 1) // len(unique_train)
    real_oversampled = (unique_train * reps)[:n_real]
    mixed = sim + real_oversampled
    mixed_path = out_dir / "reie-train-rir.txt"
    mixed_path.write_text("\n".join(mixed) + "\n")

    n_linked = 0
    for row in hold_rows:
        src = Path(row["sample_dir"])
        rel = src.relative_to(dataset_root)
        _symlink_sample(src, holdout_root / rel)
        n_linked += 1

    summary = {
        "n_total": len(rows),
        "n_train_rooms": len({r["env_id"] for r in train_rows}),
        "n_hold_rooms": len({r["env_id"] for r in hold_rows}),
        "train_arrays": sorted({r["array_id"] for r in train_rows if r["array_id"] is not None}),
        "hold_arrays": sorted({r["array_id"] for r in hold_rows if r["array_id"] is not None}),
        "n_train_rirs": len(train_rows),
        "n_hold_rirs": len(hold_rows),
        "hold_rooms": sorted(hold),
        "n_sim": n_sim,
        "n_real_oversampled": len(real_oversampled),
        "n_mixed": len(mixed),
        "frac_real": len(real_oversampled) / len(mixed),
        "npz_dir": str(npz_dir),
        "mixed_list": str(mixed_path),
        "localization_csv": str(loc_path),
        "holdout_root": str(holdout_root),
        "n_holdout_links": n_linked,
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
