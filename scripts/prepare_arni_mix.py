#!/usr/bin/env python3
"""Build Arni source-holdout mix: wideband sim + real Arni train + Arni twins.

Default hold-out speaker: S3. Train uses real S1/S2 Ambisonic-W RIRs plus
exact/dense gpuRIR twins. Evaluation tree is symlinked under --holdout-root.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset.rir_corrupt import peak_cut_dp  # noqa: E402
from eval_but import compute_angle_radius_rotated, latest_rir_wav  # noqa: E402


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path("/storage/reie/data/arni-mckenzie-nearwall"))
    p.add_argument("--npz-dir", type=Path, default=Path("/tmp/arni-real-npz"))
    p.add_argument(
        "--twin-manifest",
        type=Path,
        default=Path("/tmp/arni-twins-npz/manifest.json"),
    )
    p.add_argument(
        "--sim-list",
        type=Path,
        default=Path(
            "/storage/reie/REC-RIR-LOCALIZATION/config/"
            "rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w16-zfree-s2032/reie-train-rir.txt"
        ),
    )
    p.add_argument(
        "--sim-loc-csv",
        type=Path,
        default=Path(
            "/storage/reie/data/rec-rir/"
            "rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w16-zfree-s2032/rir_localization.csv"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/storage/reie/REC-RIR-LOCALIZATION/config/rir-zfree-wideband-w16-mix-arni-hold-S3"
        ),
    )
    p.add_argument(
        "--holdout-root",
        type=Path,
        default=Path("/storage/reie/data/arni-mckenzie-holdout-S3"),
    )
    p.add_argument("--hold-speakers", default="S3")
    p.add_argument(
        "--arni-frac",
        type=float,
        default=0.35,
        help="Fraction of mixed list coming from real+twin Arni npzs (rest = wideband sim).",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    hold = {x.strip() for x in args.hold_speakers.split(",") if x.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.npz_dir.mkdir(parents=True, exist_ok=True)
    args.holdout_root.mkdir(parents=True, exist_ok=True)

    # Pack measured Arni
    train_rows = []
    hold_rows = []
    for meta_path in sorted(args.dataset_root.rglob("rotated_metadata.json")):
        sample_dir = meta_path.parent
        if sample_dir.name.startswith("_"):
            continue
        wav = latest_rir_wav(sample_dir)
        if wav is None:
            continue
        meta = json.loads(meta_path.read_text())
        env = meta.get("environment") or {}
        scenario = meta.get("scenario") or {}
        env_name = str(env.get("env_name") or sample_dir.parent.parent.name)
        mic = str(scenario.get("microphone") or sample_dir.parent.name)
        spk = str(scenario.get("speaker") or sample_dir.name)
        npz_name = f"{env_name}_{mic}_{spk}.npz".replace("/", "_")
        npz_path = args.npz_dir / npz_name
        if args.force or not npz_path.is_file():
            angle, radius = _pack_npz(wav, meta, npz_path)
        else:
            rot = meta["rotated"]
            angle, radius = compute_angle_radius_rotated(
                float(rot["source_x"]),
                float(rot["source_y"]),
                float(rot["mic_x"]),
                float(rot["mic_y"]),
            )
        row = {
            "npz": os.path.normpath(npz_path.as_posix()),
            "angle_deg": angle,
            "radius_m": radius,
            "speaker": spk,
            "sample_dir": sample_dir,
            "kind": "real",
        }
        if spk in hold:
            hold_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows:
        raise SystemExit("no Arni train RIRs left after speaker hold-out")
    if not hold_rows:
        raise SystemExit(f"no hold-out samples for speakers {sorted(hold)}")

    # Twin manifest
    if not args.twin_manifest.is_file():
        raise SystemExit(f"missing twin manifest: {args.twin_manifest}")
    man = json.loads(args.twin_manifest.read_text())
    twin_rows = []
    for block in ("exact", "dense"):
        for r in man.get(block, []):
            twin_rows.append(
                {
                    "npz": os.path.normpath(r["npz"]),
                    "angle_deg": float(r["angle_deg"]),
                    "radius_m": float(r["radius_m"]),
                    "kind": r.get("kind", block),
                }
            )

    arni_pool = train_rows + twin_rows
    arni_paths = [r["npz"] for r in arni_pool]

    # Localization CSV = sim labels + all arni train/twin labels
    loc_path = args.out_dir / "rir_localization.csv"
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
        for row in arni_pool:
            writer.writerow(
                {
                    "rir_path": row["npz"],
                    "angle_deg": f"{float(row['angle_deg']):.6f}",
                    "radius_m": f"{float(row['radius_m']):.6f}",
                }
            )

    sim = _read_lines(args.sim_list)
    n_sim = len(sim)
    # arni_frac of mixed list from arni_pool; rest from sim
    n_arni = max(1, int(round(n_sim * args.arni_frac / max(1e-8, 1.0 - args.arni_frac))))
    # Keep sim size as-is and add oversampled arni — same pattern as dechorate mix
    # but with configurable arni fraction of final mix.
    # final = sim + arni_oversampled, with |arni|/(|sim|+|arni|) ≈ arni_frac
    n_arni = max(1, int(round(args.arni_frac / max(1e-8, 1.0 - args.arni_frac) * n_sim)))
    reps = (n_arni + len(arni_paths) - 1) // len(arni_paths)
    arni_over = (arni_paths * reps)[:n_arni]
    mixed = sim + arni_over
    mixed_path = args.out_dir / "reie-train-rir.txt"
    mixed_path.write_text("\n".join(mixed) + "\n")

    (args.out_dir / "arni-train-unique.txt").write_text(
        "\n".join(r["npz"] for r in train_rows) + "\n"
    )
    (args.out_dir / "arni-twin-unique.txt").write_text(
        "\n".join(r["npz"] for r in twin_rows) + "\n"
    )

    n_linked = 0
    for row in hold_rows:
        src = Path(row["sample_dir"])
        rel = src.relative_to(args.dataset_root)
        _symlink_sample(src, args.holdout_root / rel)
        n_linked += 1
    # prep_summary marker for eval_but-style loaders
    (args.holdout_root / "prep_summary.json").write_text(
        json.dumps(
            {
                "n_ok": n_linked,
                "hold_speakers": sorted(hold),
                "source": str(args.dataset_root),
            },
            indent=2,
        )
        + "\n"
    )

    summary = {
        "hold_speakers": sorted(hold),
        "n_real_train": len(train_rows),
        "n_real_hold": len(hold_rows),
        "n_twins": len(twin_rows),
        "n_arni_pool": len(arni_pool),
        "n_sim": n_sim,
        "n_arni_oversampled": len(arni_over),
        "n_mixed": len(mixed),
        "frac_arni": len(arni_over) / len(mixed),
        "mixed_list": str(mixed_path),
        "localization_csv": str(loc_path),
        "holdout_root": str(args.holdout_root),
        "n_holdout_links": n_linked,
        "npz_dir_real": str(args.npz_dir),
        "twin_manifest": str(args.twin_manifest),
    }
    (args.out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
