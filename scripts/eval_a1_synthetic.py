#!/usr/bin/env python3
"""A1: synthetic main results for the locked paper model.

Reports angle MAE (°) and distance MAE (m) on:
  1) full synthetic test split (config SNR range, test seed)
  2) fixed 1024-clip ablation reference at 20 dB SNR
  3) training-mean constant chance baselines on the same test labels

Usage (from repo root, with venv activated):
  python scripts/eval_a1_synthetic.py \\
    --config config/Rec-RIR-heads-only-free-mic-z-h100-miczfree-s2033-2gpu.toml \\
    --checkpoint /storage/reie/experiments/free-mic-z-h100-miczfree-s2033/ckpt/best.tar \\
    --out-dir /storage/reie/experiments/free-mic-z-h100-miczfree-s2033/a1_eval \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import toml
import torch
from tqdm import tqdm

# Repo root on sys.path when launched as scripts/eval_a1_synthetic.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trainer_inferencer.utils import initialize_module  # noqa: E402

import warnings

warnings.filterwarnings("ignore")


def _load_model(config: dict, checkpoint_path: Path, device: torch.device):
    model = initialize_module(config["model"]["path"], args=config["model"]["args"])
    ckpt = torch.load(checkpoint_path.as_posix(), map_location="cpu")
    epoch = ckpt.get("epoch", None)
    state_dict = {
        k.replace("module.", ""): v
        for k, v in ckpt["model"].items()
        if not any(x in k for x in ["ops", "params"])
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model.to(device).eval()
    return model, epoch


def _expand_dataset_to_all_rirs(dataset) -> int:
    """Training datasets are speech-indexed; for paper eval we want one clip per RIR.

    With ``rir_shuffle=False``, ``__getitem__`` uses ``rir_pathlist[index % n_rir]``.
    Expanding ``length`` / ``source_pathlist`` to ``n_rir`` therefore yields each RIR
    exactly once (speech cycled). Returns the new length.
    """
    n_rir = len(dataset.rir_pathlist)
    if n_rir == 0:
        raise RuntimeError("empty rir_pathlist")
    speech = list(dataset.source_pathlist)
    if not speech:
        raise RuntimeError("empty source_pathlist")
    reps = (n_rir + len(speech) - 1) // len(speech)
    dataset.source_pathlist = (speech * reps)[:n_rir]
    dataset.length = n_rir
    return n_rir


def _build_eval_dataloader(
    config: dict,
    *,
    snr_range: List[float],
    rir_pathlist: str,
    src_pathlist: str,
    noise_pathlist: str,
    batchsize: int,
    seed: int,
    seq_len: float,
    num_workers: int = 4,
):
    """Build a single-process eval loader that covers every RIR once."""
    from torch.utils.data import DataLoader

    from dataset.dataset_RecRIR import MyDataset
    from dataset.utils import MyDistributedSampler

    dl_args = config["dataloader"]["args"]
    ds = MyDataset(
        src_pathlist,
        rir_pathlist,
        noise_pathlist,
        list(snr_range),
        seq_len,
        config["acoustic"]["args"]["sr"],
        shuffle=False,  # deterministic RIR index = sample index % n_rir
        noisy_proportion=float(dl_args.get("noisy_proportion", 1)),
        repeat=False,
        localization_csv=dl_args.get("localization_csv"),
        localization_csv_columns=tuple(
            dl_args.get(
                "localization_csv_columns",
                ("rir_path", "angle_deg", "radius_m"),
            )
        ),
        num_angles=int(dl_args.get("num_angles", 181)),
        max_rad_value=float(dl_args.get("max_rad_value", 13.0)),
        rad_resolution=float(dl_args.get("rad_resolution", 0.25)),
        use_room_conditioning=bool(dl_args.get("use_room_conditioning", False)),
        volume_mode=str(dl_args.get("volume_mode", "drop")),
        volume_constant=float(dl_args.get("volume_constant", 0.5)),
        volume_shuffle_seed=int(dl_args.get("volume_shuffle_seed", 2024)),
    )
    n = _expand_dataset_to_all_rirs(ds)
    sampler = MyDistributedSampler(
        ds, seed=int(seed), shuffle=False, rank=0, num_replicas=1, drop_last=False
    )
    loader = DataLoader(
        ds,
        sampler=sampler,
        batch_size=batchsize,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )
    meta = {
        "n_rir": n,
        "snr_range": list(snr_range),
        "seed": int(seed),
        "seq_len": float(seq_len),
        "rir_pathlist": rir_pathlist,
        "src_pathlist": src_pathlist,
        "noise_pathlist": noise_pathlist,
        "batchsize": batchsize,
    }
    return loader, meta

@torch.no_grad()
def _eval_model(
    model: torch.nn.Module,
    dataloader,
    transformfunc,
    device: torch.device,
    *,
    rad_resolution: float,
    use_room_conditioning: bool,
    num_samples: Optional[int] = None,
    desc: str = "eval",
) -> Dict[str, Any]:
    angle_err_sum = 0.0
    radius_err_sum = 0.0
    n_angle = 0
    n_radius = 0
    seen = 0
    target = num_samples if num_samples is not None else len(dataloader.dataset)
    pbar = tqdm(total=target, desc=desc, unit="sample")

    for batch in dataloader:
        noisy_wav = batch[0].to(device)
        angle_class = batch[5].to(device)
        radius_class = batch[6].to(device)
        room_params = batch[7].to(device) if use_room_conditioning else None

        if num_samples is not None and seen + noisy_wav.shape[0] > num_samples:
            keep = num_samples - seen
            noisy_wav = noisy_wav[:keep]
            angle_class = angle_class[:keep]
            radius_class = radius_class[:keep]
            if room_params is not None:
                room_params = room_params[:keep]

        input_complex = transformfunc.stft(noisy_wav, output_type="complex")
        input_ft = transformfunc.preprocess(input_complex)
        outputs = model(input_ft, room_params=room_params)
        angle_preds = torch.argmax(outputs[3], dim=1)
        radius_preds = torch.argmax(outputs[4], dim=1)

        valid_a = angle_class >= 0
        if valid_a.any():
            angle_err_sum += torch.abs(
                angle_preds[valid_a] - angle_class[valid_a]
            ).float().sum().item()
            n_angle += int(valid_a.sum().item())

        valid_r = radius_class >= 0
        if valid_r.any():
            radius_err_sum += torch.abs(
                radius_preds[valid_r] - radius_class[valid_r]
            ).float().sum().item()
            n_radius += int(valid_r.sum().item())

        bs = int(noisy_wav.shape[0])
        seen += bs
        pbar.update(bs)
        if num_samples is not None and seen >= num_samples:
            break
    pbar.close()

    angle_mae = angle_err_sum / n_angle if n_angle else float("nan")
    radius_mae = (radius_err_sum / n_radius) * rad_resolution if n_radius else float("nan")
    return {
        "n_clips": seen,
        "n_angle_valid": n_angle,
        "n_radius_valid": n_radius,
        "angle_mae_deg": float(angle_mae),
        "distance_mae_m": float(radius_mae),
    }


def _read_pathlist(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _load_localization_map(csv_path: str) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["rir_path"]] = (float(row["angle_deg"]), float(row["radius_m"]))
    return out


def _training_mean_chance(
    train_rir_list: List[str],
    eval_rir_list: List[str],
    loc_map: Dict[str, Tuple[float, float]],
) -> Dict[str, Any]:
    train_angles = np.array([loc_map[p][0] for p in train_rir_list if p in loc_map])
    train_radii = np.array([loc_map[p][1] for p in train_rir_list if p in loc_map])
    eval_angles = np.array([loc_map[p][0] for p in eval_rir_list if p in loc_map])
    eval_radii = np.array([loc_map[p][1] for p in eval_rir_list if p in loc_map])
    if len(train_angles) == 0 or len(eval_angles) == 0:
        raise RuntimeError("Empty train/eval localization labels for chance baseline")
    mean_angle = float(train_angles.mean())
    mean_radius = float(train_radii.mean())
    return {
        "train_n": int(len(train_angles)),
        "eval_n": int(len(eval_angles)),
        "train_mean_angle_deg": mean_angle,
        "train_mean_distance_m": mean_radius,
        "chance_angle_mae_deg": float(np.abs(eval_angles - mean_angle).mean()),
        "chance_distance_mae_m": float(np.abs(eval_radii - mean_radius).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="A1 synthetic main results")
    parser.add_argument(
        "--config",
        required=True,
        help="Training/eval TOML (locked final model config)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best.tar (or other ckpt)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for JSON/summary outputs",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument(
        "--fixed1024-list",
        default=(
            "config/rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w09-miczfree-s2033/"
            "reie-test-rir-fixed1024.txt"
        ),
        help="Pathlist of the fixed 1024-clip ablation reference (from test)",
    )
    parser.add_argument(
        "--fixed1024-seed",
        type=int,
        default=2024,
        help="Dataloader seed for the fixed 1024-clip set (speech/noise draws)",
    )
    parser.add_argument(
        "--skip-full-test",
        action="store_true",
        help="Only run chance + fixed1024 (debug)",
    )
    parser.add_argument(
        "--skip-fixed1024",
        action="store_true",
        help="Skip the fixed 1024-clip @20 dB pass",
    )
    parser.add_argument(
        "--test-speech",
        default=None,
        help="Override test speech pathlist (default: config src_pathlist[2])",
    )
    parser.add_argument(
        "--test-noise",
        default=None,
        help="Override test noise pathlist (default: config noise_pathlist[2])",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().absolute()
    ckpt_path = Path(args.checkpoint).expanduser().absolute()
    out_dir = Path(args.out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(_REPO_ROOT)
    config = toml.load(config_path.as_posix())
    dl_args = config["dataloader"]["args"]
    model_args = config["model"]["args"]
    rad_resolution = float(
        model_args.get("rad_resolution", dl_args.get("rad_resolution", 0.25))
    )
    use_room = bool(dl_args.get("use_room_conditioning", False))
    train_snr = list(dl_args["snr_range"])
    test_seed = int(dl_args["seeds"][2])
    meta_seed = int(config["meta"]["seed"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} config={config_path} ckpt={ckpt_path}")

    transformfunc = initialize_module(
        config["acoustic"]["path"], config["acoustic"]["args"]
    )
    model, ckpt_epoch = _load_model(config, ckpt_path, device)
    print(f"checkpoint epoch={ckpt_epoch}")

    # ---- chance baselines on full test labels (CSV; no audio needed) ----
    train_rir = _read_pathlist(Path(dl_args["rir_pathlist"][0]))
    test_rir = _read_pathlist(Path(dl_args["rir_pathlist"][2]))
    fixed_list_path = Path(args.fixed1024_list)
    if not fixed_list_path.is_absolute():
        fixed_list_path = _REPO_ROOT / fixed_list_path
    fixed1024_rir = _read_pathlist(fixed_list_path)
    loc_map = _load_localization_map(dl_args["localization_csv"])

    chance_test = _training_mean_chance(train_rir, test_rir, loc_map)
    chance_1024 = _training_mean_chance(train_rir, fixed1024_rir, loc_map)
    print(
        f"[chance/test] angle={chance_test['chance_angle_mae_deg']:.3f}° "
        f"dist={chance_test['chance_distance_mae_m']:.3f} m "
        f"(train mean {chance_test['train_mean_angle_deg']:.3f}° / "
        f"{chance_test['train_mean_distance_m']:.3f} m)"
    )

    results: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt_epoch,
        "config": str(config_path),
        "meta_seed": meta_seed,
        "rad_resolution_m": rad_resolution,
        "num_room_params": int(model_args.get("num_room_params", 7)),
        "volume_mode": dl_args.get("volume_mode"),
        "corpus": {
            "localization_csv": dl_args["localization_csv"],
            "train_rir_list": dl_args["rir_pathlist"][0],
            "val_rir_list": dl_args["rir_pathlist"][1],
            "test_rir_list": dl_args["rir_pathlist"][2],
            "train_n": len(train_rir),
            "val_n": len(_read_pathlist(Path(dl_args["rir_pathlist"][1]))),
            "test_n": len(test_rir),
            "speech_lists": dl_args["src_pathlist"],
            "noise_lists": dl_args["noise_pathlist"],
        },
        "chance_training_mean_on_test": chance_test,
        "chance_training_mean_on_fixed1024": chance_1024,
    }

    # Speech/noise lists: default to index-2 entries from the locked training config;
    # optional CLI overrides for hygiene re-evals (C2).
    test_speech = args.test_speech or dl_args["src_pathlist"][2]
    test_noise = args.test_noise or dl_args["noise_pathlist"][2]
    test_seq_len = float(dl_args["seq_lenlist"][2])
    results["corpus"]["speech_lists_effective_test"] = test_speech
    results["corpus"]["noise_lists_effective_test"] = test_noise

    # ---- full test ----
    if not args.skip_full_test:
        test_loader, test_meta = _build_eval_dataloader(
            config,
            snr_range=train_snr,
            rir_pathlist=dl_args["rir_pathlist"][2],
            src_pathlist=test_speech,
            noise_pathlist=test_noise,
            batchsize=args.batchsize,
            seed=test_seed,
            seq_len=test_seq_len,
        )
        test_metrics = _eval_model(
            model,
            test_loader,
            transformfunc,
            device,
            rad_resolution=rad_resolution,
            use_room_conditioning=use_room,
            desc="full-test",
        )
        results["full_test"] = {
            **test_metrics,
            "split": "test",
            "snr_range_db": train_snr,
            "seed": test_seed,
            "seq_len_s": test_seq_len,
            "rir_pathlist": dl_args["rir_pathlist"][2],
            "src_pathlist": test_speech,
            "noise_pathlist": test_noise,
            "batchsize": args.batchsize,
            "protocol": (
                "one clip per test RIR (dataset length expanded from speech-indexed "
                f"to n_rir={test_meta['n_rir']}); rir_shuffle=False; drop_last=False; "
                f"SNR drawn U{train_snr} with dataloader seed {test_seed}"
            ),
        }
        print(
            f"[full-test] n={test_metrics['n_clips']} "
            f"angle={test_metrics['angle_mae_deg']:.3f}° "
            f"dist={test_metrics['distance_mae_m']:.3f} m"
        )

    # ---- fixed 1024 @ 20 dB ----
    if not args.skip_fixed1024:
        fixed_loader, fixed_meta = _build_eval_dataloader(
            config,
            snr_range=[20.0, 20.0],
            rir_pathlist=str(fixed_list_path),
            src_pathlist=test_speech,
            noise_pathlist=test_noise,
            batchsize=args.batchsize,
            seed=int(args.fixed1024_seed),
            seq_len=test_seq_len,
        )
        fixed_metrics = _eval_model(
            model,
            fixed_loader,
            transformfunc,
            device,
            rad_resolution=rad_resolution,
            use_room_conditioning=use_room,
            num_samples=1024,
            desc="fixed1024@20dB",
        )
        results["fixed1024_20db"] = {
            **fixed_metrics,
            "split": "test-subset-fixed1024",
            "snr_range_db": [20.0, 20.0],
            "seed": int(args.fixed1024_seed),
            "seq_len_s": test_seq_len,
            "rir_pathlist": str(fixed_list_path),
            "src_pathlist": test_speech,
            "noise_pathlist": test_noise,
            "selection": (
                "first 1024 paths from reie-test-rir.txt as committed in "
                "reie-test-rir-fixed1024.txt; dataset then sorts by basename; "
                "one clip per RIR at fixed SNR 20 dB"
            ),
            "batchsize": args.batchsize,
            "n_rir_in_list": fixed_meta["n_rir"],
        }
        print(
            f"[fixed1024@20dB] n={fixed_metrics['n_clips']} "
            f"angle={fixed_metrics['angle_mae_deg']:.3f}° "
            f"dist={fixed_metrics['distance_mae_m']:.3f} m"
        )

    summary_path = out_dir / "a1_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {summary_path}")

    # Human-readable one-pager
    lines = [
        "A1 synthetic main results",
        f"checkpoint: {ckpt_path} (epoch={ckpt_epoch})",
        f"config: {config_path}",
        f"meta_seed: {meta_seed}",
        "",
    ]
    if "full_test" in results:
        ft = results["full_test"]
        lines += [
            f"full test: angle MAE={ft['angle_mae_deg']:.4f}°  "
            f"distance MAE={ft['distance_mae_m']:.4f} m  "
            f"n={ft['n_clips']}  SNR={ft['snr_range_db']}  seed={ft['seed']}",
        ]
    if "fixed1024_20db" in results:
        f1024 = results["fixed1024_20db"]
        lines += [
            f"fixed1024@20dB: angle MAE={f1024['angle_mae_deg']:.4f}°  "
            f"distance MAE={f1024['distance_mae_m']:.4f} m  "
            f"n={f1024['n_clips']}  SNR={f1024['snr_range_db']}  seed={f1024['seed']}",
        ]
    lines += [
        f"chance (train-mean on test): angle={chance_test['chance_angle_mae_deg']:.4f}°  "
        f"distance={chance_test['chance_distance_mae_m']:.4f} m",
        f"train mean constants: angle={chance_test['train_mean_angle_deg']:.6f}°  "
        f"distance={chance_test['train_mean_distance_m']:.6f} m",
        f"corpus sizes: train={results['corpus']['train_n']}  "
        f"val={results['corpus']['val_n']}  test={results['corpus']['test_n']}",
        f"test speech: {test_speech}",
        f"test noise: {test_noise}",
    ]
    (out_dir / "a1_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
