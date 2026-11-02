#!/usr/bin/env python3
"""Leave-one-out / k-shot few-shot on labeled BUT ReverbDB.

Start from the z-free joint checkpoint. --split mic holds out one microphone
(room/mic). --split room holds out an entire environment so no sibling mic
from the same room is in train. --split array holds out an entire microphone
array and trains on the other array across all rooms. --split mic80 holds out
20% of global microphone IDs, stratified by array when possible. --split kshot
trains on k labeled RIRs at the test microphone and queries the rest (no other
rooms in train).

Headline numbers are the pooled held-out predictions plus FiLM-shuffle on
the same clips. Metrics stay grouped by microphone.

Gates (not MAE alone):
  * held-out angle MAE vs the 24.51° occupancy prior (fair) / 22.2° label median
  * r_within (must rise if this is localization)
  * FiLM-shuffle must not follow the donor

Default: freeze the trunk, train loc heads (+ head FiLM). Pass --freeze none
to unfreeze everything.

Usage (from REC-RIR-LOCALIZATION):
  /storage/reie/.venv_custom_ssh/bin/python scripts/train_but_loso.py
  /storage/reie/.venv_custom_ssh/bin/python scripts/train_but_loso.py --split room
  /storage/reie/.venv_custom_ssh/bin/python scripts/train_but_loso.py \\
      --split kshot --k 2 --draws 3 --kshot-rooms C236 --freeze none \\
      --batch 1 --eval-batch 4 --lr 3e-5 \\
      --exp /storage/reie/experiments/but-kshot-c236-zfree-unfrozen
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import toml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from eval_but import (  # noqa: E402
    ButRirDataset,
    _load_model,
    _split_by_film_volume_range,
    _split_by_model_radius_range,
    class_to_angle,
    class_to_radius,
    grouped_metrics,
    load_rotated_but_entries,
)
from model.lossF import GaussianLabelSmoothingLoss  # noqa: E402
from trainer_inferencer.utils import initialize_module  # noqa: E402

HEAD_KEYS = (
    "angle_fc",
    "angle_act",
    "angle_drop",
    "radius_head",
    "film_angle",
    "film_radius",
)

WANDB_PROJECT = "rec-rir-localization"


def _fold_key(row: dict, split: str) -> str:
    if split == "array":
        return _array_key(row)
    if split == "room":
        return str(row["room"])
    return str(row["group_key"])


def _mic_id(row: dict) -> str:
    mic = str(row.get("mic_id") or row.get("mic") or "")
    if not mic:
        mic = str(row.get("group_key") or "").split("/")[-1]
    if not mic:
        raise ValueError(f"cannot infer mic id from row keys {sorted(row)}")
    return mic


def _array_key(row: dict) -> str:
    mic = _mic_id(row)
    if "_" not in mic:
        raise ValueError(f"cannot infer array from mic id {mic!r}")
    return mic.split("_", 1)[0]


def _kshot_jobs(
    entries: list[dict],
    k: int,
    draws: int,
    seed: int,
    rooms: list[str] | None,
    min_query: int = 2,
) -> list[dict]:
    """Pure k-shot: support is k RIRs at one mic; query is the rest at that mic."""
    jobs: list[dict] = []
    mics = sorted({row["group_key"] for row in entries})
    for mi, mic in enumerate(mics):
        mic_rows = [r for r in entries if r["group_key"] == mic]
        room = str(mic_rows[0]["room"])
        if rooms is not None and not any(
            room == name or room.endswith("_" + name) or name in room for name in rooms
        ):
            continue
        if len(mic_rows) < k + min_query:
            continue
        donor = [dict(r) for r in entries if r["group_key"] != mic]
        for draw in range(draws):
            rng = np.random.default_rng(seed + 10007 * (mi + 1) + draw)
            pick = set(rng.choice(len(mic_rows), size=k, replace=False).tolist())
            train_entries = [dict(mic_rows[i]) for i in range(len(mic_rows)) if i in pick]
            test_entries = [dict(mic_rows[i]) for i in range(len(mic_rows)) if i not in pick]
            jobs.append(
                {
                    "hold": f"{mic}#d{draw}",
                    "mic": mic,
                    "draw": draw,
                    "train_entries": train_entries,
                    "test_entries": test_entries,
                    "donor_entries": donor,
                }
            )
    if not jobs:
        raise ValueError(
            f"no k-shot jobs: need mics with >= {k + min_query} RIRs"
            + (f" in rooms {rooms}" if rooms is not None else "")
        )
    return jobs


def _loo_jobs(entries: list[dict], split: str) -> list[dict]:
    groups = sorted({_fold_key(row, split) for row in entries})
    jobs = []
    for hold in groups:
        train_entries = [dict(r) for r in entries if _fold_key(r, split) != hold]
        test_entries = [dict(r) for r in entries if _fold_key(r, split) == hold]
        jobs.append(
            {
                "hold": hold,
                "mic": None,
                "draw": None,
                "train_entries": train_entries,
                "test_entries": test_entries,
                "donor_entries": train_entries,
            }
        )
    return jobs


def _mic80_jobs(entries: list[dict]) -> list[dict]:
    """Five-fold 80/20 over global mic IDs, balanced across arrays when possible."""
    mic_ids = sorted({_mic_id(row) for row in entries})
    by_array: dict[str, list[str]] = {}
    for mic in mic_ids:
        arr = mic.split("_", 1)[0] if "_" in mic else "mic"
        by_array.setdefault(arr, []).append(mic)

    hold_sets: list[list[str]] = []
    arrays = sorted(by_array)
    if len(arrays) == 2 and len(by_array[arrays[0]]) == len(by_array[arrays[1]]):
        for i in range(len(by_array[arrays[0]])):
            hold_sets.append([by_array[arrays[0]][i], by_array[arrays[1]][i]])
    else:
        fold_size = max(1, round(0.2 * len(mic_ids)))
        hold_sets = [mic_ids[i : i + fold_size] for i in range(0, len(mic_ids), fold_size)]

    jobs = []
    for hold_mics in hold_sets:
        hold = set(hold_mics)
        train_entries = [dict(r) for r in entries if _mic_id(r) not in hold]
        test_entries = [dict(r) for r in entries if _mic_id(r) in hold]
        jobs.append(
            {
                "hold": "mics_" + "+".join(hold_mics),
                "mic": None,
                "draw": None,
                "train_entries": train_entries,
                "test_entries": test_entries,
                "donor_entries": train_entries,
            }
        )
    return jobs


def _mic_fraction_jobs(
    entries: list[dict],
    train_fracs: list[float],
    draws: int,
    seed: int,
) -> list[dict]:
    """Balanced random train/test splits over microphone IDs for learning curves.

    With dEchorate A0/A1 there are five mics per array. For each requested
    train fraction, sample the same number of train mics from each array so the
    held-out set stays balanced across array geometry.
    """
    mic_ids = sorted({_mic_id(row) for row in entries})
    by_array: dict[str, list[str]] = {}
    for mic in mic_ids:
        arr = mic.split("_", 1)[0] if "_" in mic else "mic"
        by_array.setdefault(arr, []).append(mic)
    arrays = sorted(by_array)
    if not arrays:
        raise ValueError("no microphone IDs found for mic-fraction split")

    jobs: list[dict] = []
    seen: set[tuple[float, tuple[str, ...]]] = set()
    for frac in train_fracs:
        if not (0.0 < frac < 1.0):
            raise ValueError(f"train fraction must be in (0, 1), got {frac}")
        for draw in range(draws):
            rng = np.random.default_rng(seed + 1009 * draw + int(round(frac * 10000)))
            train_mics: list[str] = []
            for arr in arrays:
                mics = by_array[arr]
                n_train = int(round(frac * len(mics)))
                n_train = min(max(1, n_train), len(mics) - 1)
                pick = sorted(rng.choice(mics, size=n_train, replace=False).tolist())
                train_mics.extend(pick)
            train_key = tuple(sorted(train_mics))
            key = (round(frac, 6), train_key)
            if key in seen:
                continue
            seen.add(key)
            train = set(train_mics)
            test_mics = sorted(set(mic_ids) - train)
            train_entries = [dict(r) for r in entries if _mic_id(r) in train]
            test_entries = [dict(r) for r in entries if _mic_id(r) not in train]
            jobs.append(
                {
                    "hold": f"train{frac:.2f}_d{draw:02d}_test_" + "+".join(test_mics),
                    "mic": None,
                    "draw": draw,
                    "train_frac": frac,
                    "train_mics": sorted(train_mics),
                    "test_mics": test_mics,
                    "train_entries": train_entries,
                    "test_entries": test_entries,
                    "donor_entries": train_entries,
                }
            )
    if not jobs:
        raise ValueError("no mic-fraction jobs generated")
    return jobs


def _read_paths(txt: Path) -> list[str]:
    return [os.path.normpath(line.strip()) for line in txt.read_text().splitlines() if line.strip()]


def _filter_entries(entries: list[dict], max_rad_value: float) -> list[dict]:
    in_range, _ = _split_by_model_radius_range(entries, max_rad_value)
    vol_ok, _ = _split_by_film_volume_range(in_range)
    if not vol_ok:
        raise ValueError("no entries left after radius / FiLM-volume filters")
    return vol_ok


def _is_head_param(name: str) -> bool:
    return any(key in name for key in HEAD_KEYS)


def _freeze(model: nn.Module, mode: str) -> tuple[list[nn.Parameter], int, int]:
    trainable: list[nn.Parameter] = []
    n_train = n_frozen = 0
    for name, param in model.named_parameters():
        if mode == "none" or _is_head_param(name):
            param.requires_grad = True
            trainable.append(param)
            n_train += param.numel()
        else:
            param.requires_grad = False
            n_frozen += param.numel()
    return trainable, n_train, n_frozen


def _set_train_mode(model: nn.Module, freeze: str) -> None:
    """Keep frozen trunk in eval so its dropout stays off."""
    if freeze == "none":
        model.train()
        return
    model.eval()
    for name, module in model.named_modules():
        if name and _is_head_param(name):
            module.train()


def _attach_donors(test_entries: list[dict], donor_entries: list[dict]) -> list[dict]:
    """Give each test row a FiLM vector from a *train* microphone, not itself."""
    groups = sorted({row["group_key"] for row in donor_entries})
    if not groups:
        raise ValueError("shuffle needs at least one train microphone as donor")
    first: dict[str, dict] = {}
    for row in donor_entries:
        first.setdefault(row["group_key"], row)
    out = []
    for i, row in enumerate(test_entries):
        cloned = dict(row)
        cloned.pop("_film_row", None)
        donor = groups[i % len(groups)]
        cloned["_film_row"] = first[donor]
        out.append(cloned)
    return out


def _safe_slug(group: str) -> str:
    return group.replace("/", "__").replace(" ", "_")


@torch.no_grad()
def _score(
    model: nn.Module,
    entries: list[dict],
    *,
    speech_paths: list[str],
    noise_paths: list[str],
    transformfunc,
    device: torch.device,
    num_angles: int,
    max_rad_value: float,
    rad_resolution: float,
    seq_len: float,
    sr: int,
    clips_per_rir: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    film_mode: str,
    snr_range: tuple[float, float] = (20.0, 20.0),
) -> dict:
    model.eval()
    dataset = ButRirDataset(
        entries=entries,
        speech_paths=speech_paths,
        noise_paths=noise_paths,
        clips_per_rir=clips_per_rir,
        seq_len=seq_len,
        sr=sr,
        snr_range=snr_range,
        noisy_proportion=0.0,
        num_angles=num_angles,
        max_rad_value=max_rad_value,
        rad_resolution=rad_resolution,
        seed=seed,
        film_mode="normal" if film_mode == "shuffle" else film_mode,
        randomize=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    rows: list[dict] = []
    for batch in loader:
        reverb_wav, angle_cls_gt, radius_cls_gt, room_params, rir_paths, angle_degs, radius_ms, clip_idxs, group_keys, film_groups = batch
        reverb_wav = reverb_wav.to(device, non_blocking=True)
        rp = room_params.to(device, non_blocking=True)
        if film_mode == "off":
            rp = None
        input_ft = transformfunc.preprocess(transformfunc.stft(reverb_wav, output_type="complex"))
        outputs = model(input_ft, room_params=rp)
        angle_preds = torch.argmax(outputs[3], dim=1).cpu().numpy()
        radius_preds = torch.argmax(outputs[4], dim=1).cpu().numpy()
        B = reverb_wav.shape[0]
        for b in range(B):
            pred_ang = class_to_angle(int(angle_preds[b]))
            pred_rad = class_to_radius(int(radius_preds[b]), rad_resolution)
            gt_ang = float(angle_degs[b])
            gt_rad = float(radius_ms[b])
            rows.append(
                {
                    "rir_path": rir_paths[b],
                    "group_key": group_keys[b],
                    "film_group": film_groups[b],
                    "clip_idx": int(clip_idxs[b]),
                    "gt_angle_deg": gt_ang,
                    "pred_angle_deg": pred_ang,
                    "gt_radius_m": gt_rad,
                    "pred_radius_m": pred_rad,
                    "gt_angle_class": int(angle_cls_gt[b]),
                    "pred_angle_class": int(angle_preds[b]),
                    "gt_radius_class": int(radius_cls_gt[b]),
                    "pred_radius_class": int(radius_preds[b]),
                }
            )
    grp = np.array([r["group_key"] for r in rows])
    rir = np.array([r["rir_path"] for r in rows])
    ang = grouped_metrics(
        grp, rir,
        np.array([r["pred_angle_deg"] for r in rows]),
        np.array([r["gt_angle_deg"] for r in rows]),
        np.arange(0.0, 181.0, 1.0),
    )
    rad = grouped_metrics(
        grp, rir,
        np.array([r["pred_radius_m"] for r in rows]),
        np.array([r["gt_radius_m"] for r in rows]),
        np.arange(0.0, max_rad_value + 1e-9, rad_resolution),
    )
    return {"angle": ang, "radius": rad, "rows": rows, "film_mode": film_mode}


def _metrics_row(tag: str, scored: dict) -> dict:
    a, d = scored["angle"], scored["radius"]
    return {
        "tag": tag,
        "film": scored["film_mode"],
        "ang_mae": float(a["mae_clip"]),
        "rad_mae": float(d["mae_clip"]),
        "r_within": float(a["corr_within_group"]),
        "r_between": float(a["corr_between_groups"]),
        "pred_sd": float(a["pred_std_within_group"]),
        "n_rirs": int(a["n_rirs"]),
        "n_groups": int(a["n_groups"]),
        "oracle_group": float(a["baseline_group_constant_mae"]),
        "oracle_const": float(a["baseline_constant_mae"]),
        "beats_oracle": bool(a["mae_clip"] < a["baseline_group_constant_mae"]),
    }


def _read_pred_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "rir_path": row["rir_path"],
                    "group_key": row["group_key"],
                    "film_group": row["film_group"],
                    "clip_idx": int(row["clip_idx"]),
                    "gt_angle_deg": float(row["gt_angle_deg"]),
                    "pred_angle_deg": float(row["pred_angle_deg"]),
                    "gt_radius_m": float(row["gt_radius_m"]),
                    "pred_radius_m": float(row["pred_radius_m"]),
                    "gt_angle_class": int(float(row["gt_angle_class"])),
                    "pred_angle_class": int(float(row["pred_angle_class"])),
                    "gt_radius_class": int(float(row["gt_radius_class"])),
                    "pred_radius_class": int(float(row["pred_radius_class"])),
                }
            )
    return rows


def _scored_from_rows(
    rows: list[dict],
    film_mode: str,
    max_rad_value: float,
    rad_resolution: float,
) -> dict:
    grp = np.array([r["group_key"] for r in rows])
    rir = np.array([r["rir_path"] for r in rows])
    return {
        "angle": grouped_metrics(
            grp, rir,
            np.array([r["pred_angle_deg"] for r in rows]),
            np.array([r["gt_angle_deg"] for r in rows]),
            np.arange(0.0, 181.0, 1.0),
        ),
        "radius": grouped_metrics(
            grp, rir,
            np.array([r["pred_radius_m"] for r in rows]),
            np.array([r["gt_radius_m"] for r in rows]),
            np.arange(0.0, max_rad_value + 1e-9, rad_resolution),
        ),
        "rows": rows,
        "film_mode": film_mode,
    }


def _fmt(row: dict) -> str:
    rw = row["r_within"]
    rb = row["r_between"]
    rw_s = f"{rw:6.2f}" if rw == rw else "   nan"
    rb_s = f"{rb:6.2f}" if rb == rb else "   nan"
    return (
        f"{row['tag']:28s} {row['film']:8s} "
        f"{row['ang_mae']:7.2f}d {row['rad_mae']:6.3f}m "
        f"rw={rw_s} rb={rb_s}  sd={row['pred_sd']:5.1f}d  "
        f"n={row['n_rirs']}g{row['n_groups']}"
        f"{'  BEATS oracle' if row['beats_oracle'] else ''}"
    )


def _one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    angle_loss: nn.Module,
    radius_loss: nn.Module,
    transformfunc,
    device: torch.device,
    freeze: str,
    clip_grad: float,
) -> dict:
    _set_train_mode(model, freeze)
    total = 0.0
    n = 0
    ang_err = 0.0
    rad_err = 0.0
    n_item = 0
    for batch in loader:
        reverb_wav, angle_cls, radius_cls, room_params = batch[:4]
        reverb_wav = reverb_wav.to(device, non_blocking=True)
        angle_cls = angle_cls.to(device, non_blocking=True)
        radius_cls = radius_cls.to(device, non_blocking=True)
        rp = room_params.to(device, non_blocking=True)
        input_ft = transformfunc.preprocess(transformfunc.stft(reverb_wav, output_type="complex"))
        outputs = model(input_ft, room_params=rp)
        angle_logits, radius_logits = outputs[3], outputs[4]
        loss = angle_loss(angle_logits, angle_cls) + radius_loss(radius_logits, radius_cls)
        with torch.no_grad():
            ang_err += (angle_logits.argmax(1) - angle_cls).abs().float().sum().item()
            rad_err += (radius_logits.argmax(1) - radius_cls).abs().float().sum().item()
            n_item += int(angle_cls.numel())
        del outputs, angle_logits, radius_logits
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip_grad
            )
        opt.step()
        total += float(loss.item())
        n += 1
        del loss
    return {
        "loss": total / max(n, 1),
        "ang_mae": ang_err / max(n_item, 1),
        "rad_mae": (rad_err / max(n_item, 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="/storage/reie/experiments/but-loso-zfree-heads")
    p.add_argument(
        "--config",
        default="config/Rec-RIR-heads-only-epoch35-straight-to-15m-zfree-2gpu.toml",
    )
    p.add_argument(
        "--checkpoint",
        default="/storage/reie/experiments/epoch35-straight-to-15m-zfree-2gpu/ckpt/best.tar",
    )
    p.add_argument("--dataset-root", default="/storage/reie/Filteted and rotated BUT")
    p.add_argument("--train-speech", default="config/reie_train_speech_quick.txt")
    p.add_argument("--eval-speech", default="config/reie_validation_speech_quick.txt")
    p.add_argument("--noise", default="config/reie_train_noise.txt")
    p.add_argument("--freeze", choices=("heads", "none"), default="heads")
    p.add_argument(
        "--split",
        choices=("mic", "room", "array", "mic80", "micfrac", "kshot"),
        default="mic",
        help="Hold out a microphone, room, array, 80/20 mic-id fold, mic-fraction sweep, or k labeled RIRs.",
    )
    p.add_argument("--k", type=int, default=2, help="support RIRs per mic for --split kshot")
    p.add_argument("--draws", type=int, default=3, help="random support draws per mic for k-shot")
    p.add_argument(
        "--mic-train-fracs",
        default="0.8,0.6,0.4,0.2",
        help="Comma-separated train mic fractions for --split micfrac.",
    )
    p.add_argument(
        "--micfrac-draws",
        type=int,
        default=5,
        help="Balanced random draws per train fraction for --split micfrac.",
    )
    p.add_argument(
        "--kshot-rooms",
        default="C236",
        help="comma-separated rooms for k-shot, or 'all'",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=8, help="train batch size")
    p.add_argument(
        "--eval-batch",
        type=int,
        default=None,
        help="eval batch size (defaults to --batch). Unfrozen train may need 2-4.",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clips-train", type=int, default=8)
    p.add_argument("--clips-eval", type=int, default=20)
    p.add_argument("--seq-len", type=float, default=4.0)
    p.add_argument("--eval-every", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--eval-workers",
        type=int,
        default=0,
        help="DataLoader workers for eval/scoring; default preserves old single-process scoring.",
    )
    p.add_argument("--folds", default="all", help="'all' or comma-separated 0-based indices")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip folds that already have summary.json + pred CSVs; still re-score zero-shot.",
    )
    p.add_argument("--wandb-mode", default="online")
    p.add_argument(
        "--dataset-tag",
        default="but",
        help="Short name for logs / wandb (e.g. but, dechorate).",
    )
    p.add_argument("--run-name", default="", help="Override wandb run name.")
    p.add_argument("--wandb-group", default="", help="Override wandb group.")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    exp = Path(args.exp)
    exp.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = toml.load(config_path.as_posix())
    ckpt_path = Path(args.checkpoint)
    dataset_root = Path(args.dataset_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_args = config["model"]["args"]
    ac_args = config["acoustic"]["args"]
    num_angles = int(model_args.get("num_angles", 181))
    max_rad_value = float(model_args.get("max_rad_value", 13.0))
    rad_resolution = float(model_args.get("rad_resolution", 0.25))
    num_radius = int(max_rad_value / rad_resolution) + 1
    sr = int(ac_args.get("sr", 16000))
    transformfunc = initialize_module(config["acoustic"]["path"], ac_args)

    entries = _filter_entries(load_rotated_but_entries(dataset_root), max_rad_value)
    train_speech = _read_paths(root / args.train_speech)
    eval_speech = _read_paths(root / args.eval_speech)
    noise_paths = _read_paths(root / args.noise) if args.noise else []

    tag = str(args.dataset_tag).strip() or "but"
    if args.split == "kshot":
        rooms = None if args.kshot_rooms.strip().lower() == "all" else [
            x.strip() for x in args.kshot_rooms.split(",") if x.strip()
        ]
        jobs = _kshot_jobs(entries, args.k, args.draws, args.seed, rooms)
        split_name = f"k-shot k={args.k} draws={args.draws}"
        wandb_group = args.wandb_group.strip() or f"{tag}-kshot-fewshot"
    elif args.split == "mic80":
        jobs = _mic80_jobs(entries)
        split_name = "mic-ID 80/20"
        wandb_group = args.wandb_group.strip() or f"{tag}-mic80-fewshot"
    elif args.split == "micfrac":
        train_fracs = [
            float(x.strip()) for x in args.mic_train_fracs.split(",") if x.strip()
        ]
        jobs = _mic_fraction_jobs(entries, train_fracs, args.micfrac_draws, args.seed)
        split_name = "mic-ID fraction sweep"
        wandb_group = args.wandb_group.strip() or f"{tag}-micfrac-fewshot"
    else:
        jobs = _loo_jobs(entries, args.split)
        split_name = (
            "LORO" if args.split == "room"
            else "array hold-out" if args.split == "array"
            else "LOSO"
        )
        wandb_group = args.wandb_group.strip() or (
            f"{tag}-loro-fewshot" if args.split == "room"
            else f"{tag}-array-fewshot" if args.split == "array"
            else f"{tag}-loso-fewshot"
        )

    if args.folds == "all":
        fold_idx = list(range(len(jobs)))
    else:
        fold_idx = [int(x) for x in args.folds.split(",") if x.strip() != ""]

    print(
        f"{tag} {split_name}  split={args.split}  jobs={len(jobs)}  "
        f"rirs={len(entries)}  folds={fold_idx}\n"
        f"ckpt={ckpt_path}  freeze={args.freeze}  device={device}",
        flush=True,
    )
    for i, job in enumerate(jobs):
        mark = " *" if i in fold_idx else ""
        print(
            f"  [{i:02d}] {job['hold']:40s}  "
            f"train={len(job['train_entries'])}  test={len(job['test_entries'])}{mark}",
            flush=True,
        )

    import wandb

    run_name = args.run_name.strip() or (
        f"{tag}-{args.split}-zfree-{args.freeze}{'-resume' if args.resume else ''}"
    )
    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        group=wandb_group,
        job_type="fewshot",
        mode=args.wandb_mode,
        dir=str(exp),
        tags=[tag, args.split, "fewshot", "zfree", args.freeze],
        notes=(
            f"{split_name} few-shot on labeled {tag}. "
            "Pooled held-out MAE / r_within / FiLM-shuffle are the gates. "
            "Metrics stay grouped by microphone. "
            f"freeze={args.freeze}."
        ),
        config={
            "split": args.split,
            "k": args.k,
            "draws": args.draws,
            "mic_train_fracs": args.mic_train_fracs,
            "micfrac_draws": args.micfrac_draws,
            "freeze": args.freeze,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch": args.batch,
            "clips_train": args.clips_train,
            "clips_eval": args.clips_eval,
            "n_jobs": len(jobs),
            "n_rirs": len(entries),
            "checkpoint": str(ckpt_path),
            "folds": fold_idx,
        },
    )

    angle_loss = GaussianLabelSmoothingLoss(num_angles, sigma=10.0).to(device)
    radius_loss = GaussianLabelSmoothingLoss(num_radius, sigma=2.0).to(device)

    eval_batch = args.eval_batch if args.eval_batch is not None else args.batch
    score_kw = dict(
        speech_paths=eval_speech,
        noise_paths=[],
        transformfunc=transformfunc,
        device=device,
        num_angles=num_angles,
        max_rad_value=max_rad_value,
        rad_resolution=rad_resolution,
        seq_len=args.seq_len,
        sr=sr,
        clips_per_rir=args.clips_eval,
        batch_size=eval_batch,
        num_workers=args.eval_workers,
        seed=args.seed,
    )

    pooled_zero_n: list[dict] = []
    pooled_ft_n: list[dict] = []
    pooled_ft_s: list[dict] = []
    fold_summaries: list[dict] = []

    def _tag_job_rows(rows: list[dict], job: dict) -> list[dict]:
        out = []
        for row in rows:
            cloned = dict(row)
            if "train_frac" in job:
                cloned["train_frac"] = job["train_frac"]
                cloned["draw"] = job.get("draw")
            out.append(cloned)
        return out

    for fi in fold_idx:
        job = jobs[fi]
        hold = job["hold"]
        train_entries = job["train_entries"]
        test_entries = job["test_entries"]
        donor_entries = job["donor_entries"]
        slug = _safe_slug(hold)
        fold_dir = exp / f"fold_{fi:02d}_{slug}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n===== fold {fi} hold-out {hold}  "
            f"train_rirs={len(train_entries)} test_rirs={len(test_entries)}  "
            f"test_mics={len({r['group_key'] for r in test_entries})} =====",
            flush=True,
        )

        model = _load_model(config, ckpt_path, device)
        trainable, n_train, n_frozen = _freeze(model, args.freeze)
        print(f"trainable {n_train:,}  frozen {n_frozen:,}  tensors={len(trainable)}", flush=True)

        zero_n = _score(model, test_entries, film_mode="normal", **score_kw)
        shuffle_entries = _attach_donors(test_entries, donor_entries)
        zero_s = _score(model, shuffle_entries, film_mode="shuffle", **score_kw)
        z_n_row = _metrics_row(f"fold{fi}-zero-normal", zero_n)
        z_s_row = _metrics_row(f"fold{fi}-zero-shuffle", zero_s)
        print(_fmt(z_n_row), flush=True)
        print(_fmt(z_s_row), flush=True)
        pooled_zero_n.extend(_tag_job_rows(zero_n["rows"], job))
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pred_n_path = fold_dir / "preds_normal.csv"
        pred_s_path = fold_dir / "preds_shuffle.csv"
        if (
            args.resume
            and (fold_dir / "summary.json").exists()
            and pred_n_path.exists()
            and pred_s_path.exists()
        ):
            last_n = _scored_from_rows(
                _read_pred_csv(pred_n_path), "normal", max_rad_value, rad_resolution
            )
            last_s = _scored_from_rows(
                _read_pred_csv(pred_s_path), "shuffle", max_rad_value, rad_resolution
            )
            pooled_ft_n.extend(_tag_job_rows(last_n["rows"], job))
            pooled_ft_s.extend(_tag_job_rows(last_s["rows"], job))
            saved = json.loads((fold_dir / "summary.json").read_text())
            saved["zero_normal"] = z_n_row
            saved["zero_shuffle"] = z_s_row
            saved["ft_normal"] = _metrics_row(f"fold{fi}-ft-normal", last_n)
            saved["ft_shuffle"] = _metrics_row(f"fold{fi}-ft-shuffle", last_s)
            fold_summaries.append(saved)
            print(
                f"  resume skip train  "
                f"{_fmt(saved['ft_normal'])}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        train_set = ButRirDataset(
            entries=train_entries,
            speech_paths=train_speech,
            noise_paths=noise_paths,
            clips_per_rir=args.clips_train,
            seq_len=args.seq_len,
            sr=sr,
            snr_range=(5.0, 20.0),
            noisy_proportion=1.0 if noise_paths else 0.0,
            num_angles=num_angles,
            max_rad_value=max_rad_value,
            rad_resolution=rad_resolution,
            seed=args.seed + 1000 * fi,
            randomize=True,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=False,
        )
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=1e-6)

        last_n = last_s = None
        history = []
        for epoch in range(1, args.epochs + 1):
            tr = _one_epoch(
                model, train_loader, opt, angle_loss, radius_loss,
                transformfunc, device, args.freeze, clip_grad=1.0,
            )
            tr["rad_mae"] *= rad_resolution
            sched.step()
            do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
            log = {
                "fold": fi,
                "epoch": epoch,
                "train/loss": tr["loss"],
                "train/ang_mae": tr["ang_mae"],
                "train/rad_mae": tr["rad_mae"],
                "lr": float(opt.param_groups[0]["lr"]),
            }
            line = (
                f"  ep {epoch:02d}  train {tr['loss']:.3f}  "
                f"{tr['ang_mae']:6.2f}° {tr['rad_mae']:.3f}m"
            )
            if do_eval:
                last_n = _score(model, test_entries, film_mode="normal", **score_kw)
                last_s = _score(
                    model, _attach_donors(test_entries, donor_entries),
                    film_mode="shuffle", **score_kw,
                )
                n_row = _metrics_row(f"fold{fi}-ep{epoch}-normal", last_n)
                s_row = _metrics_row(f"fold{fi}-ep{epoch}-shuffle", last_s)
                log.update(
                    {
                        "test/ang_mae": n_row["ang_mae"],
                        "test/rad_mae": n_row["rad_mae"],
                        "test/r_within": n_row["r_within"],
                        "test/shuffle_ang_mae": s_row["ang_mae"],
                    }
                )
                line += (
                    f"  | test {n_row['ang_mae']:6.2f}° rw={n_row['r_within']:.2f}"
                    f"  shuffle {s_row['ang_mae']:6.2f}°"
                )
                history.append({"epoch": epoch, "normal": n_row, "shuffle": s_row})
            print(line, flush=True)
            wandb.log(log)

        torch.save(
            {"model": model.state_dict(), "hold": hold, "freeze": args.freeze, "epoch": args.epochs},
            fold_dir / "last.pt",
        )
        assert last_n is not None and last_s is not None
        pooled_ft_n.extend(_tag_job_rows(last_n["rows"], job))
        pooled_ft_s.extend(_tag_job_rows(last_s["rows"], job))

        with (fold_dir / "preds_normal.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(last_n["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(last_n["rows"])
        with (fold_dir / "preds_shuffle.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(last_s["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(last_s["rows"])

        fold_summaries.append(
            {
                "fold": fi,
                "hold": hold,
                "split": args.split,
                "k": args.k if args.split == "kshot" else None,
                "draw": job["draw"],
                "train_frac": job.get("train_frac"),
                "train_mics": job.get("train_mics"),
                "test_mics": job.get("test_mics"),
                "n_train_rirs": len(train_entries),
                "n_test_rirs": len(test_entries),
                "n_test_mics": len({r["group_key"] for r in test_entries}),
                "zero_normal": z_n_row,
                "zero_shuffle": z_s_row,
                "ft_normal": _metrics_row(f"fold{fi}-ft-normal", last_n),
                "ft_shuffle": _metrics_row(f"fold{fi}-ft-shuffle", last_s),
                "history": history,
            }
        )
        (fold_dir / "summary.json").write_text(json.dumps(fold_summaries[-1], indent=2, default=str))

    def _pool(rows: list[dict], tag: str, film: str) -> dict:
        fake = {
            "angle": grouped_metrics(
                np.array([r["group_key"] for r in rows]),
                np.array([r["rir_path"] for r in rows]),
                np.array([r["pred_angle_deg"] for r in rows]),
                np.array([r["gt_angle_deg"] for r in rows]),
                np.arange(0.0, 181.0, 1.0),
            ),
            "radius": grouped_metrics(
                np.array([r["group_key"] for r in rows]),
                np.array([r["rir_path"] for r in rows]),
                np.array([r["pred_radius_m"] for r in rows]),
                np.array([r["gt_radius_m"] for r in rows]),
                np.arange(0.0, max_rad_value + 1e-9, rad_resolution),
            ),
            "film_mode": film,
            "rows": rows,
        }
        return _metrics_row(tag, fake)

    lines = [
        f"{tag} {split_name} few-shot",
        f"ckpt={ckpt_path}",
        f"split={args.split}  freeze={args.freeze}  epochs={args.epochs}  lr={args.lr}",
        f"train clips/RIR={args.clips_train}  eval clips/RIR={args.clips_eval}",
        "",
        f"{'condition':28s} {'film':8s} {'ang':>8s} {'rad':>7s}  r_within r_between",
        "-" * 88,
    ]
    for fs in fold_summaries:
        lines.append(_fmt(fs["zero_normal"]))
        lines.append(_fmt(fs["ft_normal"]))
        lines.append(_fmt(fs["ft_shuffle"]))
        lines.append("")

    if pooled_zero_n and pooled_ft_n:
        pool_z = _pool(pooled_zero_n, "POOLED zero-shot", "normal")
        pool_n = _pool(pooled_ft_n, "POOLED few-shot", "normal")
        pool_s = _pool(pooled_ft_s, "POOLED few-shot", "shuffle")
        lines += [
            "-" * 88,
            _fmt(pool_z),
            _fmt(pool_n),
            _fmt(pool_s),
            "-" * 88,
            (
                f"oracle per-group constant  {pool_n['oracle_group']:.2f}d   "
                f"(never hears the audio)"
            ),
            f"oracle one constant         {pool_n['oracle_const']:.2f}d",
            "",
            "Success if POOLED few-shot normal MAE < per-group oracle, "
            "r_within clearly above zero-shot, and shuffle stays near chance.",
        ]
        wandb.summary["pooled/zero_ang_mae"] = pool_z["ang_mae"]
        wandb.summary["pooled/ft_ang_mae"] = pool_n["ang_mae"]
        wandb.summary["pooled/ft_r_within"] = pool_n["r_within"]
        wandb.summary["pooled/shuffle_ang_mae"] = pool_s["ang_mae"]
        wandb.summary["pooled/beats_oracle"] = pool_n["beats_oracle"]

        frac_vals = sorted({
            float(r["train_frac"]) for r in pooled_ft_n
            if r.get("train_frac") is not None
        })
        if frac_vals:
            lines += [
                "",
                "By train-mic fraction",
                f"{'fraction':>8s} {'n_rows':>8s} {'normal_ang':>11s} {'shuffle':>9s} {'rad':>7s}  r_within",
                "-" * 88,
            ]
            for frac in frac_vals:
                n_rows = [r for r in pooled_ft_n if r.get("train_frac") == frac]
                s_rows = [r for r in pooled_ft_s if r.get("train_frac") == frac]
                frac_n = _pool(n_rows, f"frac{frac:.2f} normal", "normal")
                frac_s = _pool(s_rows, f"frac{frac:.2f} shuffle", "shuffle")
                lines.append(
                    f"{frac:8.2f} {len(n_rows):8d} "
                    f"{frac_n['ang_mae']:11.2f} {frac_s['ang_mae']:9.2f} "
                    f"{frac_n['rad_mae']:7.3f}  {frac_n['r_within']:8.3f}"
                )
                wandb.summary[f"frac/{frac:.2f}/ang_mae"] = frac_n["ang_mae"]
                wandb.summary[f"frac/{frac:.2f}/shuffle_ang_mae"] = frac_s["ang_mae"]
                wandb.summary[f"frac/{frac:.2f}/r_within"] = frac_n["r_within"]

    text = "\n".join(lines) + "\n"
    (exp / "summary.txt").write_text(text)
    (exp / "folds.json").write_text(json.dumps(fold_summaries, indent=2, default=str))
    print("\n" + text, flush=True)
    print(f"wrote {exp / 'summary.txt'}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
