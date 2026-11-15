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

Optional Arm-B mix: ``--mix-synth-csv`` (+ optional ``--mix-synth-list``) appends
labeled gpuRIR ``.npz`` entries into each fold's *train* set only. Real
train/test membership is unchanged. Default mix ratio 1.0 → equal #synth as
#real train RIRs (50/50 by count).

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
import time
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


def _source_id(row: dict) -> str:
    src = str(row.get("spk_setup") or row.get("source_id") or row.get("speaker") or "")
    if not src:
        # Fallback: <room>/<mic>/<speaker>/RIR.wav → speaker is parent of RIR dir's parent… 
        # entries store sample path indirectly; prefer spk_setup from loader.
        raise ValueError(f"cannot infer source id from row keys {sorted(row)}")
    return src


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


def _mic_source_fraction_jobs(
    entries: list[dict],
    mic_train_fracs: list[float],
    src_train_fracs: list[float],
    draws: int,
    seed: int,
) -> list[dict]:
    """Product holdout over mic IDs × source IDs.

    Train only on train-mics × train-sources. Primary test is the joint corner
    hold-mics × hold-sources. Extra eval_sets cover the other three quadrants
    for diagnostics (seen-mic/new-src, new-mic/seen-src, sanity ceiling).
    """
    mic_ids = sorted({_mic_id(row) for row in entries})
    src_ids = sorted({_source_id(row) for row in entries})
    by_array: dict[str, list[str]] = {}
    for mic in mic_ids:
        arr = mic.split("_", 1)[0] if "_" in mic else "mic"
        by_array.setdefault(arr, []).append(mic)
    arrays = sorted(by_array)
    if not arrays:
        raise ValueError("no microphone IDs found for mic×source split")
    if len(src_ids) < 2:
        raise ValueError(f"need >=2 sources for source holdout, got {src_ids}")

    jobs: list[dict] = []
    seen: set[tuple[float, float, tuple[str, ...], tuple[str, ...]]] = set()
    for mic_frac in mic_train_fracs:
        if not (0.0 < mic_frac < 1.0):
            raise ValueError(f"mic train fraction must be in (0, 1), got {mic_frac}")
        for src_frac in src_train_fracs:
            if not (0.0 < src_frac < 1.0):
                raise ValueError(f"source train fraction must be in (0, 1), got {src_frac}")
            for draw in range(draws):
                rng = np.random.default_rng(
                    seed
                    + 1009 * draw
                    + int(round(mic_frac * 10000))
                    + 17 * int(round(src_frac * 10000))
                )
                train_mics: list[str] = []
                for arr in arrays:
                    mics = by_array[arr]
                    n_train = int(round(mic_frac * len(mics)))
                    n_train = min(max(1, n_train), len(mics) - 1)
                    pick = sorted(rng.choice(mics, size=n_train, replace=False).tolist())
                    train_mics.extend(pick)
                n_src_train = int(round(src_frac * len(src_ids)))
                n_src_train = min(max(1, n_src_train), len(src_ids) - 1)
                train_srcs = sorted(
                    rng.choice(src_ids, size=n_src_train, replace=False).tolist()
                )
                train_mics_t = tuple(sorted(train_mics))
                train_srcs_t = tuple(train_srcs)
                key = (round(mic_frac, 6), round(src_frac, 6), train_mics_t, train_srcs_t)
                if key in seen:
                    continue
                seen.add(key)

                train_mic_set = set(train_mics)
                train_src_set = set(train_srcs)
                hold_mics = sorted(set(mic_ids) - train_mic_set)
                hold_srcs = sorted(set(src_ids) - train_src_set)

                def _pick(mics: set[str], srcs: set[str]) -> list[dict]:
                    return [
                        dict(r)
                        for r in entries
                        if _mic_id(r) in mics and _source_id(r) in srcs
                    ]

                train_entries = _pick(train_mic_set, train_src_set)
                q_tm_ts = train_entries
                q_tm_hs = _pick(train_mic_set, set(hold_srcs))
                q_hm_ts = _pick(set(hold_mics), train_src_set)
                q_hm_hs = _pick(set(hold_mics), set(hold_srcs))
                if not train_entries or not q_hm_hs:
                    continue
                jobs.append(
                    {
                        "hold": (
                            f"mic{mic_frac:.2f}_src{src_frac:.2f}_d{draw:02d}"
                            f"_holdM_{'+'.join(hold_mics)}"
                            f"_holdS_{'+'.join(hold_srcs)}"
                        ),
                        "mic": None,
                        "draw": draw,
                        "train_frac": mic_frac,
                        "src_train_frac": src_frac,
                        "train_mics": sorted(train_mics),
                        "test_mics": hold_mics,
                        "train_srcs": train_srcs,
                        "test_srcs": hold_srcs,
                        "train_entries": train_entries,
                        "test_entries": q_hm_hs,  # primary: joint holdout
                        "donor_entries": train_entries,
                        "eval_sets": {
                            "train_mic_train_src": q_tm_ts,
                            "train_mic_hold_src": q_tm_hs,
                            "hold_mic_train_src": q_hm_ts,
                            "hold_mic_hold_src": q_hm_hs,
                        },
                    }
                )
    if not jobs:
        raise ValueError("no mic×source jobs generated")
    return jobs


# dEchorate paper Table-5 RT60s cluster into three clear bands (unique values):
#   low  <0.20 → {0.14, 0.18, 0.19}
#   mid  [0.20, 0.45) → {0.26, 0.33, 0.34}
#   high ≥0.45 → {0.56, 0.73}
_RT60_BAND_EDGES: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 0.20),
    ("mid", 0.20, 0.45),
    ("high", 0.45, float("inf")),
)


def _rt60_band(rt60: float) -> str:
    for name, lo, hi in _RT60_BAND_EDGES:
        if lo <= float(rt60) < hi:
            return name
    raise ValueError(f"RT60 {rt60} falls outside configured bands")


def _rt60_band_jobs(entries: list[dict]) -> list[dict]:
    """Leave-one-RT60-band-out (3 folds: hold low / mid / high).

    Primary test = held band. ``eval_sets`` also scores the train-band corner
    so in-distribution recovery can be compared to the holdout failure mode.
    """
    by_band: dict[str, list[dict]] = {name: [] for name, _, _ in _RT60_BAND_EDGES}
    for row in entries:
        band = _rt60_band(float(row["rt60"]))
        by_band[band].append(dict(row))
    missing = [n for n, rows in by_band.items() if not rows]
    if missing:
        raise ValueError(f"RT60 bands empty after labeling: {missing}")

    jobs: list[dict] = []
    for hold_band, _, _ in _RT60_BAND_EDGES:
        test_entries = by_band[hold_band]
        train_entries = [
            dict(r)
            for band, rows in by_band.items()
            if band != hold_band
            for r in rows
        ]
        hold_vals = sorted({round(float(r["rt60"]), 4) for r in test_entries})
        train_vals = sorted({round(float(r["rt60"]), 4) for r in train_entries})
        jobs.append(
            {
                "hold": f"rt60_{hold_band}_hold{hold_vals}_train{train_vals}",
                "mic": None,
                "draw": None,
                "hold_band": hold_band,
                "train_rt60": train_vals,
                "test_rt60": hold_vals,
                "train_entries": train_entries,
                "test_entries": test_entries,
                "donor_entries": train_entries,
                "eval_sets": {
                    "train_rt60": train_entries,
                    "hold_rt60": test_entries,
                },
            }
        )
    return jobs


def _read_paths(txt: Path) -> list[str]:
    return [os.path.normpath(line.strip()) for line in txt.read_text().splitlines() if line.strip()]


def _load_synth_npz_pool(
    loc_csv: Path,
    rir_list: Path | None,
    max_rad_value: float,
) -> list[dict]:
    """CSV angle/radius rows for labeled gpuRIR ``.npz`` (real dEchorate skipped)."""
    allow: set[str] | None = None
    if rir_list is not None:
        allow = {os.path.normpath(p) for p in _read_paths(rir_list)}

    rows: list[dict] = []
    with loc_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = os.path.normpath(str(row["rir_path"]).strip())
            if allow is not None and path not in allow:
                continue
            if "dechorate" in path.lower():
                continue
            if not path.lower().endswith(".npz"):
                continue
            try:
                ang = float(row["angle_deg"])
                rad = float(row["radius_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if rad > max_rad_value:
                continue
            rows.append({"rir_path": path, "angle_deg": ang, "radius_m": rad})
    if not rows:
        raise ValueError(f"No synth npz rows under {loc_csv} (list={rir_list})")
    return rows


def _materialize_synth_entries(
    pool_rows: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Sample ``n`` synth CSV rows and attach FiLM metadata from each npz."""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    if n >= len(pool_rows):
        chosen = list(pool_rows)
        rng.shuffle(chosen)
    else:
        idx = rng.choice(len(pool_rows), size=n, replace=False)
        chosen = [pool_rows[int(i)] for i in idx]

    out: list[dict] = []
    for row in chosen:
        path = row["rir_path"]
        with np.load(path, allow_pickle=False) as z:
            rt60 = float(np.asarray(z["RT60"]).reshape(-1)[0])
            room_sz = np.asarray(z["room_sz"], dtype=np.float64).reshape(-1)
            pos_rcv = np.asarray(z["pos_rcv"], dtype=np.float64).reshape(-1, 3)[0]
            pos_src = np.asarray(z["pos_src"], dtype=np.float64).reshape(-1, 3)[0]
        stem = Path(path).stem
        mic_id = f"synth_{stem}"
        out.append(
            {
                "rir_path": path,
                "angle_deg": f"{float(row['angle_deg']):.4f}",
                "radius_m": f"{float(row['radius_m']):.4f}",
                "rt60": str(rt60),
                "room_sz_x": str(float(room_sz[0])),
                "room_sz_y": str(float(room_sz[1])),
                "room_sz_z": str(float(room_sz[2])),
                "mic_x": str(float(pos_rcv[0])),
                "mic_y": str(float(pos_rcv[1])),
                "mic_z": str(float(pos_rcv[2])),
                "source_x": str(float(pos_src[0])),
                "source_y": str(float(pos_src[1])),
                "source_z": str(float(pos_src[2])),
                "room": "synth_s2030",
                "mic_id": mic_id,
                "group_key": f"synth_s2030/{mic_id}",
                "spk_setup": "synth",
                "applied_rotation": "synth",
                "is_synth": "1",
            }
        )
    return out


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
    volume_mode: str = "drop",
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
        volume_mode=volume_mode,
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
    n_total = len(dataset)
    n_seen = 0
    t0 = time.time()
    print(
        f"  score[{film_mode}] start  n={n_total}  batch={batch_size}  workers={num_workers}",
        flush=True,
    )
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
        n_seen += B
        if n_seen == B or n_seen % max(batch_size * 20, 160) == 0 or n_seen >= n_total:
            elapsed = max(time.time() - t0, 1e-3)
            print(
                f"  score[{film_mode}] {n_seen}/{n_total}  "
                f"{n_seen / elapsed:.1f} samp/s",
                flush=True,
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
        choices=("mic", "room", "array", "mic80", "micfrac", "micsrc", "kshot", "rt60"),
        default="mic",
        help=(
            "Hold out a microphone, room, array, 80/20 mic-id fold, mic-fraction "
            "sweep, mic×source product holdout, RT60 band, or k labeled RIRs."
        ),
    )
    p.add_argument("--k", type=int, default=2, help="support RIRs per mic for --split kshot")
    p.add_argument("--draws", type=int, default=3, help="random support draws per mic for k-shot")
    p.add_argument(
        "--mic-train-fracs",
        default="0.8,0.6,0.4,0.2",
        help="Comma-separated train mic fractions for --split micfrac / micsrc.",
    )
    p.add_argument(
        "--src-train-fracs",
        default="0.43,0.71,0.86",
        help="Comma-separated train source fractions for --split micsrc (≈3/7,5/7,6/7).",
    )
    p.add_argument(
        "--micfrac-draws",
        type=int,
        default=5,
        help="Balanced random draws per train fraction for --split micfrac / micsrc.",
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
    p.add_argument(
        "--mix-synth-csv",
        default="",
        help=(
            "Localization CSV for labeled gpuRIR npz (Arm B). When set, samples "
            "synth RIRs into each fold's train set only; real membership unchanged."
        ),
    )
    p.add_argument(
        "--mix-synth-list",
        default="",
        help="Optional path list restricting which npz from --mix-synth-csv are eligible.",
    )
    p.add_argument(
        "--mix-ratio",
        type=float,
        default=1.0,
        help="Synth:real train count ratio (1.0 = equal counts / 50-50 by RIR).",
    )
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

    synth_pool: list[dict] | None = None
    if str(args.mix_synth_csv).strip():
        mix_csv = Path(args.mix_synth_csv)
        if not mix_csv.is_absolute():
            mix_csv = root / mix_csv
        mix_list = None
        if str(args.mix_synth_list).strip():
            mix_list = Path(args.mix_synth_list)
            if not mix_list.is_absolute():
                mix_list = root / mix_list
        synth_pool = _load_synth_npz_pool(mix_csv, mix_list, max_rad_value)
        print(
            f"Arm-B synth mix pool: {len(synth_pool)} npz  "
            f"ratio={args.mix_ratio:.2f}  csv={mix_csv}",
            flush=True,
        )

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
    elif args.split == "micsrc":
        mic_fracs = [
            float(x.strip()) for x in args.mic_train_fracs.split(",") if x.strip()
        ]
        src_fracs = [
            float(x.strip()) for x in args.src_train_fracs.split(",") if x.strip()
        ]
        jobs = _mic_source_fraction_jobs(
            entries, mic_fracs, src_fracs, args.micfrac_draws, args.seed
        )
        split_name = "mic×source fraction sweep"
        wandb_group = args.wandb_group.strip() or f"{tag}-micsrc-fewshot"
    elif args.split == "rt60":
        jobs = _rt60_band_jobs(entries)
        split_name = "RT60 band leave-one-out"
        wandb_group = args.wandb_group.strip() or f"{tag}-rt60-holdout"
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
        tags=[
            tag,
            args.split,
            "fewshot",
            "zfree",
            args.freeze,
            *(["mix-synth"] if synth_pool is not None else []),
        ],
        notes=(
            f"{split_name} few-shot on labeled {tag}. "
            "Pooled held-out MAE / r_within / FiLM-shuffle are the gates. "
            "Metrics stay grouped by microphone. "
            f"freeze={args.freeze}."
            + (
                f" mix_synth_ratio={args.mix_ratio}."
                if synth_pool is not None
                else ""
            )
        ),
        config={
            "split": args.split,
            "k": args.k,
            "draws": args.draws,
            "mic_train_fracs": args.mic_train_fracs,
            "src_train_fracs": args.src_train_fracs,
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
            "mix_synth_csv": args.mix_synth_csv,
            "mix_synth_list": args.mix_synth_list,
            "mix_ratio": args.mix_ratio,
            "n_synth_pool": len(synth_pool) if synth_pool is not None else 0,
        },
    )

    angle_loss = GaussianLabelSmoothingLoss(num_angles, sigma=10.0).to(device)
    radius_loss = GaussianLabelSmoothingLoss(num_radius, sigma=2.0).to(device)

    eval_batch = args.eval_batch if args.eval_batch is not None else args.batch
    volume_mode = str(
        config.get("dataloader", {}).get("args", {}).get("volume_mode", "drop")
    ).strip().lower()
    print(
        f"volume_mode={volume_mode}  num_room_params={model_args.get('num_room_params')}",
        flush=True,
    )
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
        volume_mode=volume_mode,
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
        train_entries = list(job["train_entries"])
        n_real_train = len(train_entries)
        n_synth_train = 0
        if synth_pool is not None:
            n_synth_train = max(0, int(round(n_real_train * float(args.mix_ratio))))
            synth_entries = _materialize_synth_entries(
                synth_pool,
                n_synth_train,
                seed=args.seed + 17_000 + 97 * fi,
            )
            n_synth_train = len(synth_entries)
            train_entries = train_entries + synth_entries
        test_entries = job["test_entries"]
        donor_entries = job["donor_entries"]
        slug = _safe_slug(hold)
        fold_dir = exp / f"fold_{fi:02d}_{slug}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n===== fold {fi} hold-out {hold}  "
            f"train_rirs={n_real_train}"
            + (f"+{n_synth_train}synth" if n_synth_train else "")
            + f" test_rirs={len(test_entries)}  "
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
            volume_mode=volume_mode,
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

        quadrant_metrics: dict[str, dict] = {}
        eval_sets = job.get("eval_sets") or {}
        for qname, q_entries in eval_sets.items():
            if not q_entries:
                continue
            # Primary holdout already scored as last_n / last_s.
            if qname in ("hold_mic_hold_src", "hold_rt60"):
                quadrant_metrics[qname] = {
                    "normal": _metrics_row(f"fold{fi}-{qname}-normal", last_n),
                    "shuffle": _metrics_row(f"fold{fi}-{qname}-shuffle", last_s),
                    "n_rirs": len(q_entries),
                }
                continue
            q_n = _score(model, q_entries, film_mode="normal", **score_kw)
            q_s = _score(
                model,
                _attach_donors(q_entries, donor_entries),
                film_mode="shuffle",
                **score_kw,
            )
            quadrant_metrics[qname] = {
                "normal": _metrics_row(f"fold{fi}-{qname}-normal", q_n),
                "shuffle": _metrics_row(f"fold{fi}-{qname}-shuffle", q_s),
                "n_rirs": len(q_entries),
            }
            print(
                f"  quadrant {qname:22s}  "
                f"{quadrant_metrics[qname]['normal']['ang_mae']:6.2f}°  "
                f"shuffle {quadrant_metrics[qname]['shuffle']['ang_mae']:6.2f}°  "
                f"n={len(q_entries)}",
                flush=True,
            )

        fold_summaries.append(
            {
                "fold": fi,
                "hold": hold,
                "split": args.split,
                "k": args.k if args.split == "kshot" else None,
                "draw": job["draw"],
                "train_frac": job.get("train_frac"),
                "src_train_frac": job.get("src_train_frac"),
                "train_mics": job.get("train_mics"),
                "test_mics": job.get("test_mics"),
                "train_srcs": job.get("train_srcs"),
                "test_srcs": job.get("test_srcs"),
                "n_train_rirs": n_real_train,
                "n_train_rirs_with_synth": len(train_entries),
                "n_synth_train": n_synth_train,
                "mix_ratio": float(args.mix_ratio) if synth_pool is not None else 0.0,
                "n_test_rirs": len(test_entries),
                "n_test_mics": len({r["group_key"] for r in test_entries}),
                "zero_normal": z_n_row,
                "zero_shuffle": z_s_row,
                "ft_normal": _metrics_row(f"fold{fi}-ft-normal", last_n),
                "ft_shuffle": _metrics_row(f"fold{fi}-ft-shuffle", last_s),
                "quadrants": quadrant_metrics,
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
        (
            f"mix_synth=on  ratio={args.mix_ratio}  pool={len(synth_pool)}"
            if synth_pool is not None
            else "mix_synth=off"
        ),
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
