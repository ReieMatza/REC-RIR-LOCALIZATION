#!/usr/bin/env python3
"""
eval_but.py  —  Localization evaluation on BUT ReverbDB real-room RIRs.

Builds a speech-convolution dataset (multiple speech clips per RIR), runs the
trained BiSpatialNet model, and produces the same confusion-matrix / MAE
artefacts as confusion_matrix.py, plus a per-entry prediction CSV.

room_params (FiLM conditioning) come from metadata rather than synthetic .npz
files.  Edit the variables in main() to choose:

  dataset_root  filtered-and-rotated BUT tree (rotated_metadata.json + RIR/*.wav)
  but_csv       legacy matched CSV (e.g. ACE single-sample feed); used only
                when dataset_root is None

Usage:
  cd /storage/reie/REC-RIR-LOCALIZATION
  python scripts/eval_but.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings
from pathlib import Path

# Add project root so that trainer_inferencer, dataset, etc. resolve correctly
# when the script is invoked as `python scripts/eval_but.py` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trainer_inferencer.utils import initialize_module

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Normalization constants — must match MyDataset._extract_room_params exactly
# ---------------------------------------------------------------------------
_RT60_MIN:     float = 0.1
_RT60_MAX:     float = 1.5
_ROOM_SZ_MAX:  float = 15.0
_LOG_VOL_MIN:  float = math.log(10.0)
_LOG_VOL_MAX:  float = math.log(1000.0)
_EPS:          float = 1e-8
_NUM_ROOM_PARAMS: int = 8

FILM_MODES = ("normal", "off", "zero", "shuffle")
# Volume-channel-only policy (index 7). Independent of film_mode whole-vector ablations.
# "true" keeps log-volume; others match dataloader volume_mode for train/eval parity.
VOLUME_MODES = ("true", "drop", "constant", "shuffle", "random", "zero")


# ---------------------------------------------------------------------------
# RT60 estimation from a measured RIR
# ---------------------------------------------------------------------------

def estimate_rt60_from_rir(rir: torch.Tensor, sr: int) -> float:
    """Estimate RT60 (seconds) from an impulse response using the Schroeder
    backward-integration EDC and a T20 linear-regression extrapolation.

    The T20 method fits a line to the EDC between the −5 dB and −25 dB
    points and extrapolates the 60 dB decay time (T60 = T20 × 3).  This is
    more robust than trying to hit −65 dB (noise floor issues) and is the
    standard measurement method (ISO 3382).

    Returns the raw RT60 estimate in seconds; the caller should clamp it to
    the FiLM normalisation range [_RT60_MIN, _RT60_MAX] before use.
    """
    wav = rir.numpy() if isinstance(rir, torch.Tensor) else np.asarray(rir)
    wav = wav.ravel().astype(np.float64)
    if wav.size == 0:
        return float("nan")

    energy = wav ** 2
    edc    = np.cumsum(energy[::-1])[::-1]          # Schroeder integration
    peak   = edc[0]
    if peak <= 0:
        return float("nan")
    edc_db = 10.0 * np.log10(np.maximum(edc / peak, 1e-14))

    t = np.arange(len(edc_db)) / float(sr)

    # Locate the −5 dB and −25 dB crossings
    idx5  = np.where(edc_db <= -5.0)[0]
    idx25 = np.where(edc_db <= -25.0)[0]
    if idx5.size == 0 or idx25.size == 0 or idx25[0] <= idx5[0]:
        return float("nan")

    t_region   = t  [idx5[0] : idx25[0]]
    edc_region = edc_db[idx5[0] : idx25[0]]

    if t_region.size < 4:
        return float("nan")

    # Linear fit on the 20 dB window, extrapolate to 60 dB
    slope, _ = np.polyfit(t_region, edc_region, 1)
    if slope >= 0:
        return float("nan")
    T20  = 20.0 / abs(slope)           # seconds to drop 20 dB
    return T20 * 3.0                    # T60 = T20 × 3


# ---------------------------------------------------------------------------
# room_params from metadata
# ---------------------------------------------------------------------------

def _compute_room_params(
    rt60: float,
    Lx: float,
    Ly: float,
    Lz: float,
    mx: float,
    my: float,
    mz: float,
) -> torch.Tensor:
    """Build the 8-element FiLM vector. Matches MyDataset._extract_room_params.

    Vector layout:
        [0]   rt60_norm           clip((RT60 - 0.1) / 1.4,       0, 1)
        [1-3] room_sz_norm        clip([Lx,Ly,Lz] / 15,          0, 1)
        [4-6] mic_center_norm     clip([mx/Lx, my/Ly, mz/Lz],    0, 1)
        [7]   volume_norm         clip((ln(vol)−ln10)/(ln1000−ln10), 0, 1)
    """
    rt60_norm = float(np.clip((rt60 - _RT60_MIN) / (_RT60_MAX - _RT60_MIN), 0.0, 1.0))

    room_sz = np.array([Lx, Ly, Lz], dtype=np.float64)
    room_sz_norm = np.clip(room_sz / _ROOM_SZ_MAX, 0.0, 1.0)

    mic_center = np.array([mx, my, mz], dtype=np.float64)
    mic_center_norm = np.clip(mic_center / (room_sz + _EPS), 0.0, 1.0)

    volume_m3 = Lx * Ly * Lz
    log_vol = math.log(max(volume_m3, _EPS))
    volume_norm = float(np.clip(
        (log_vol - _LOG_VOL_MIN) / (_LOG_VOL_MAX - _LOG_VOL_MIN), 0.0, 1.0
    ))

    params = np.concatenate(
        [[rt60_norm], room_sz_norm, mic_center_norm, [volume_norm]]
    ).astype(np.float32)
    return torch.from_numpy(params)  # [8]


def compute_room_params_from_row(
    row: Dict[str, str],
    rt60_override: Optional[float] = None,
) -> torch.Tensor:
    """Build room_params from a dataset entry dict.

    Rotated-BUT entries already store training-frame geometry in
    room_sz_{x,y,z} / mic_{x,y,z} — use those directly.

    Legacy CSV rows (ACE / match_but_dataset.py) use room_depth/width and
    a nearest-wall remap.  The ACE single-sample feed hard-codes the
    depthMax-style mapping used when that CSV was produced.
    """
    rt60 = rt60_override if (rt60_override is not None and math.isfinite(rt60_override)) \
           else float(row["rt60"])

    if row.get("room_sz_x"):
        return _compute_room_params(
            rt60,
            float(row["room_sz_x"]),
            float(row["room_sz_y"]),
            float(row["room_sz_z"]),
            float(row["mic_x"]),
            float(row["mic_y"]),
            float(row["mic_z"]),
        )

    room_d = float(row["room_depth"])
    room_w = float(row["room_width"])
    room_h = float(row["room_height"])
    mic_d  = float(row["mic_depth"])
    mic_w  = float(row["mic_width"])
    mic_h  = float(row["mic_height"])

    # ACE CSV fallback: keep the mapping used when but_matched.csv was written.
    Lx, Ly = room_w, room_d
    mx, my = room_w - mic_w, mic_d
    return _compute_room_params(rt60, Lx, Ly, room_h, mx, my, mic_h)


def compute_angle_radius_rotated(
    source_x: float,
    source_y: float,
    mic_x: float,
    mic_y: float,
) -> Tuple[float, float]:
    """Folded angle [0, 180] and horizontal radius in the training frame.

    Matches gen_rir.py::compute_angle_radius with half-space restriction:
    angle = atan2(Δy, Δx) % 360, then fold angles > 180.
    """
    dx = source_x - mic_x
    dy = source_y - mic_y
    radius_m = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    if angle_deg > 180.0:
        angle_deg = 360.0 - angle_deg
    return angle_deg, radius_m


def latest_rir_wav(sample_dir: Path) -> Optional[Path]:
    """Lexicographically last *.wav in sample_dir/RIR/ (highest v## index)."""
    rir_dir = sample_dir / "RIR"
    if not rir_dir.is_dir():
        return None
    wavs = sorted(rir_dir.glob("*.wav"))
    return wavs[-1] if wavs else None


def load_rotated_but_entries(root: Path) -> List[Dict[str, str]]:
    """Walk a filtered-and-rotated BUT tree and build ButRirDataset entries.

    Each sample is ``<room>/<mic>/<speaker>/`` with ``rotated_metadata.json``
    and ``RIR/*.wav``.  Coordinates in ``rotated`` are already in the training
    frame (mic near y=0).
    """
    meta_paths = sorted(root.rglob("rotated_metadata.json"))
    if not meta_paths:
        raise ValueError(f"No rotated_metadata.json files under {root}")

    entries: List[Dict[str, str]] = []
    skipped: List[str] = []
    for meta_path in meta_paths:
        sample_dir = meta_path.parent
        wav = latest_rir_wav(sample_dir)
        if wav is None:
            skipped.append(str(sample_dir))
            continue

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        rot = meta["rotated"]
        env = meta.get("environment") or {}
        scenario = meta.get("scenario") or {}
        rotation = meta.get("rotation") or {}

        angle_deg, radius_m = compute_angle_radius_rotated(
            float(rot["source_x"]),
            float(rot["source_y"]),
            float(rot["mic_x"]),
            float(rot["mic_y"]),
        )
        room_name = str(env.get("env_name", sample_dir.parent.parent.name))
        mic_name = str(scenario.get("microphone", sample_dir.parent.name))
        entries.append({
            "rir_path": str(wav.resolve()),
            "angle_deg": f"{angle_deg:.4f}",
            "radius_m": f"{radius_m:.4f}",
            "rt60": str(rot["RT60"]),
            "room_sz_x": str(rot["room_sz_x"]),
            "room_sz_y": str(rot["room_sz_y"]),
            "room_sz_z": str(rot["room_sz_z"]),
            "mic_x": str(rot["mic_x"]),
            "mic_y": str(rot["mic_y"]),
            "mic_z": str(rot["mic_z"]),
            "source_x": str(rot["source_x"]),
            "source_y": str(rot["source_y"]),
            "source_z": str(rot["source_z"]),
            "room": room_name,
            "mic_id": mic_name,
            "group_key": f"{room_name}/{mic_name}",
            "spk_setup": str(scenario.get("speaker", sample_dir.name)),
            "applied_rotation": str(rotation.get("mode", "")),
        })

    if skipped:
        print(f"[warn] skipped {len(skipped)} samples with no RIR wav:")
        for s in skipped:
            print(f"  {s}")
    if not entries:
        raise ValueError(f"No usable RIR samples under {root}")
    return entries


def _split_by_model_radius_range(
    entries: List[Dict[str, str]],
    max_rad_value: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Tag entries with in_model_range and split on the model's radius cap.

    The cut is the config/checkpoint ``max_rad_value`` (not a hardcoded metre
    limit), so a different model filters itself.
    """
    in_range: List[Dict[str, str]] = []
    oor: List[Dict[str, str]] = []
    for row in entries:
        try:
            keep = float(row["radius_m"]) <= max_rad_value
        except (KeyError, TypeError, ValueError):
            keep = False
        row["in_model_range"] = "1" if keep else "0"
        (in_range if keep else oor).append(row)
    return in_range, oor


def _entry_room_volume_m3(row: Dict[str, str]) -> Optional[float]:
    """Room volume from rotated room_sz_* or legacy depth/width/height columns."""
    try:
        if row.get("room_sz_x"):
            return (
                float(row["room_sz_x"])
                * float(row["room_sz_y"])
                * float(row["room_sz_z"])
            )
        return (
            float(row["room_depth"])
            * float(row["room_width"])
            * float(row["room_height"])
        )
    except (KeyError, TypeError, ValueError):
        return None


def _entry_room_dims_m(row: Dict[str, str]) -> Optional[Tuple[float, float, float]]:
    """(Lx, Ly, Lz) from rotated or legacy CSV columns."""
    try:
        if row.get("room_sz_x"):
            return (
                float(row["room_sz_x"]),
                float(row["room_sz_y"]),
                float(row["room_sz_z"]),
            )
        return (
            float(row["room_width"]),
            float(row["room_depth"]),
            float(row["room_height"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _split_by_film_volume_range(
    entries: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Keep rooms whose FiLM volume / size channels are not saturated.

    Uses the same windows as ``_compute_room_params`` / MyDataset:
      volume in [exp(_LOG_VOL_MIN), exp(_LOG_VOL_MAX)] = [10, 1000] m^3
      each room dim <= _ROOM_SZ_MAX (15 m)

    These are the normalisation constants the model was trained with, not a
    separate hardcoded eval cut — change the FiLM constants and the filter
    follows.
    """
    vol_min = math.exp(_LOG_VOL_MIN)
    vol_max = math.exp(_LOG_VOL_MAX)
    in_range: List[Dict[str, str]] = []
    oor: List[Dict[str, str]] = []
    for row in entries:
        vol = _entry_room_volume_m3(row)
        dims = _entry_room_dims_m(row)
        keep = (
            vol is not None
            and dims is not None
            and vol_min <= vol <= vol_max
            and max(dims) <= _ROOM_SZ_MAX
        )
        row["in_volume_range"] = "1" if keep else "0"
        if vol is not None:
            row["volume_m3"] = f"{vol:.4f}"
        (in_range if keep else oor).append(row)
    return in_range, oor


# ---------------------------------------------------------------------------
# Class label helpers — match MyDataset._angle_to_class / _radius_to_class
# ---------------------------------------------------------------------------

def angle_to_class(angle_deg: float, num_angles: int) -> int:
    angle_deg = angle_deg % 360.0
    if num_angles <= 180:
        angle_deg = min(angle_deg, 360.0 - angle_deg)
    return max(0, min(num_angles - 1, int(angle_deg)))


def radius_to_class(radius_m: float, max_rad_value: float, rad_resolution: float) -> int:
    num_rad = int(max_rad_value / rad_resolution) + 1
    rad_idx = int(round(radius_m / rad_resolution))
    return max(0, min(num_rad - 1, rad_idx))


def class_to_angle(cls: int) -> float:
    """Centre of angle bin (1 class == 1 degree)."""
    return float(cls)


def class_to_radius(cls: int, rad_resolution: float) -> float:
    """Centre of radius bin."""
    return cls * rad_resolution


# ---------------------------------------------------------------------------
# Grouped metrics
#
# One BUT sample is a (room, microphone, source) triple scored with several
# speech clips.  Every source recorded at one microphone shares the same room
# geometry, so they all get the *same* FiLM vector: within a microphone
# position the only thing that changes is the audio.  That makes the
# microphone position the grouping that separates localization from room
# identification:
#
#   between-group corr    can the model tell the microphone positions apart
#   within-group corr     can it tell sources at one microphone apart -- this
#                         is localization, everything else is a room prior
#   group-constant MAE    score of an oracle that answers one number per
#                         microphone position and never listens to the audio
#   pred spread in group  how far the predictions move when only audio changes
#
# A plain MAE hides all of this: a model that memorises a per-room answer
# scores well on it as long as the rooms differ from each other.
# ---------------------------------------------------------------------------

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _best_constant(values: np.ndarray, grid: np.ndarray) -> Tuple[float, float]:
    """Constant that minimises MAE over `values`, and that MAE."""
    err = np.abs(values[None, :] - grid[:, None]).mean(axis=1)
    i = int(np.argmin(err))
    return float(grid[i]), float(err[i])


def grouped_metrics(
    group: np.ndarray,
    rir: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    grid: np.ndarray,
    seed: int = 0,
    n_boot: int = 2000,
) -> dict:
    """Split the error into between-group and within-group parts.

    `group` / `rir` / `pred` / `gt` are per clip.  Predictions are pooled to one
    value per RIR (median over clips) before any correlation, so the clip count
    never inflates the sample size.  The bootstrap resamples RIRs for the same
    reason.
    """
    uniq_rir = sorted(set(rir))
    rir_pred = np.array([float(np.median(pred[rir == k])) for k in uniq_rir])
    rir_gt = np.array([float(gt[rir == k][0]) for k in uniq_rir])
    rir_grp = np.array([group[rir == k][0] for k in uniq_rir])
    rir_spread = np.array([float(np.std(pred[rir == k])) for k in uniq_rir])

    uniq_grp = sorted(set(rir_grp))
    grp_pred = np.array([rir_pred[rir_grp == g].mean() for g in uniq_grp])
    grp_gt = np.array([rir_gt[rir_grp == g].mean() for g in uniq_grp])

    # Oracle baselines that ignore the recording.
    const, mae_const = _best_constant(gt, grid)
    group_const_err = 0.0
    for g in uniq_grp:
        m = group == g
        _, e = _best_constant(gt[m], grid)
        group_const_err += e * m.sum()
    mae_group_const = group_const_err / len(gt)

    # Within-group: centre both sides on the group mean, keep groups that hold
    # more than one source position.
    multi = np.array([(rir_grp == g).sum() > 1 for g in rir_grp])
    pc, gc = rir_pred.copy(), rir_gt.copy()
    for g in uniq_grp:
        m = rir_grp == g
        if m.sum() > 1:
            pc[m] -= rir_pred[m].mean()
            gc[m] -= rir_gt[m].mean()

    rng = np.random.default_rng(seed)
    idx_by_rir = [np.where(rir == k)[0] for k in uniq_rir]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq_rir), len(uniq_rir))
        idx = np.concatenate([idx_by_rir[j] for j in pick])
        boots[i] = np.abs(pred[idx] - gt[idx]).mean()

    per_group = []
    for g in uniq_grp:
        m = rir_grp == g
        c = group == g
        per_group.append({
            "group": g,
            "n_sources": int(m.sum()),
            "gt_mean": float(rir_gt[m].mean()),
            "gt_std": float(rir_gt[m].std()),
            "pred_mean": float(rir_pred[m].mean()),
            "pred_std_across_sources": float(rir_pred[m].std()),
            "mae": float(np.abs(pred[c] - gt[c]).mean()),
        })

    return {
        "n_clips": int(len(gt)),
        "n_rirs": len(uniq_rir),
        "n_groups": len(uniq_grp),
        "n_groups_multi": int(sum(1 for g in uniq_grp if (rir_grp == g).sum() > 1)),
        "mae_clip": float(np.abs(pred - gt).mean()),
        "mae_rir_pooled": float(np.abs(rir_pred - rir_gt).mean()),
        "mae_ci_lo": float(np.percentile(boots, 2.5)),
        "mae_ci_hi": float(np.percentile(boots, 97.5)),
        "baseline_constant": const,
        "baseline_constant_mae": mae_const,
        "baseline_group_constant_mae": mae_group_const,
        "corr_all": _safe_corr(rir_pred, rir_gt),
        "corr_between_groups": _safe_corr(grp_pred, grp_gt),
        "corr_within_group": _safe_corr(pc[multi], gc[multi]),
        "pred_std_within_group": float(
            np.mean([rir_pred[rir_grp == g].std() for g in uniq_grp if (rir_grp == g).sum() > 1])
        ),
        "pred_std_across_clips": float(np.mean(rir_spread)),
        "gt_std_within_group": float(
            np.mean([rir_gt[rir_grp == g].std() for g in uniq_grp if (rir_grp == g).sum() > 1])
        ),
        "per_group": per_group,
        "_rir_pred": rir_pred,
        "_rir_gt": rir_gt,
        "_rir_grp": rir_grp,
        "_uniq_rir": uniq_rir,
    }


def format_grouped_metrics(m: dict, head: str, unit: str) -> List[str]:
    beats_group = m["mae_clip"] < m["baseline_group_constant_mae"]
    return [
        f"[{head}: grouped by microphone position]",
        f"  rirs / groups:               {m['n_rirs']} in {m['n_groups']} "
        f"({m['n_groups_multi']} with >1 source)",
        f"  MAE (per clip):              {m['mae_clip']:.3f}{unit}"
        f"  95% CI over rirs [{m['mae_ci_lo']:.2f}, {m['mae_ci_hi']:.2f}]",
        f"  MAE (pooled over clips):     {m['mae_rir_pooled']:.3f}{unit}",
        f"  baseline, one constant:      {m['baseline_constant_mae']:.3f}{unit}"
        f"  (at {m['baseline_constant']:.1f}{unit})",
        f"  baseline, constant / group:  {m['baseline_group_constant_mae']:.3f}{unit}"
        f"   <- oracle that never hears the audio"
        f"{'' if beats_group else '   *** model does not beat it ***'}",
        f"  corr, all rirs:              {m['corr_all']:.3f}",
        f"  corr, between groups:        {m['corr_between_groups']:.3f}"
        f"   (room identification)",
        f"  corr, within group:          {m['corr_within_group']:.3f}"
        f"   <- localization",
        f"  pred spread within group:    {m['pred_std_within_group']:.3f}{unit}"
        f"   (truth spreads {m['gt_std_within_group']:.3f}{unit})",
        f"  pred spread across clips:    {m['pred_std_across_clips']:.3f}{unit}"
        f"   (same rir, different speech)",
    ]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_mono_resampled(path: str, target_sr: int) -> torch.Tensor:
    """Load WAV or labeled gpuRIR ``.npz`` → mono float32 at target_sr.

    Returns a 1-D tensor. NPZ files use the ``rir`` / ``fs`` fields written by
    the synthetic / mix packers (shape ``[n_src, n_mic, T]`` or ``[n_mic, T]``).
    """
    if str(path).lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            rir = np.asarray(z["rir"], dtype=np.float32)
            sr = int(np.asarray(z["fs"]).reshape(-1)[0])
        # Collapse source/mic dims → mono [T]
        while rir.ndim > 1:
            rir = rir.mean(axis=0)
        wav = torch.from_numpy(np.ascontiguousarray(rir)).unsqueeze(0)  # [1, T]
    else:
        wav, sr = torchaudio.load(path)  # [C, T]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        sr = int(sr)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)  # [T]


def _cut_or_loop(wav: torch.Tensor, target_len: int, rng: np.random.Generator) -> torch.Tensor:
    """Trim or loop+trim a 1-D waveform to exactly target_len samples."""
    T = wav.shape[-1]
    if T == target_len:
        return wav
    if T > target_len:
        start = int(rng.integers(0, T - target_len))
        return wav[start : start + target_len]
    # loop until long enough, then trim
    while wav.shape[-1] < target_len:
        wav = torch.cat([wav, wav], dim=-1)
    start = int(rng.integers(0, wav.shape[-1] - target_len))
    return wav[start : start + target_len]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ButRirDataset(Dataset):
    """Each BUT RIR entry is paired with `clips_per_rir` different speech clips.

    Returns per-item:
        reverb_wav   : [1, seq_samples]  (mono, float32, normalised to ±1)
        angle_class  : scalar long
        radius_class : scalar long
        room_params  : [8]  float32 FiLM conditioning vector
        rir_path     : str
        angle_deg    : float
        radius_m     : float
        clip_idx     : int  (which speech clip within this RIR)
        group_key    : str  room/microphone the audio came from
        film_group   : str  room/microphone the FiLM vector came from

    `film_mode` controls the conditioning, so the same audio can be scored with
    the wrong room vector:
        normal   the entry's own geometry
        zero     an all-zero vector
        shuffle  the vector of a different microphone position (set up by the
                 caller through the private "_film_row" key)
    'off' is not a dataset mode: the caller passes room_params=None instead.
    """

    def __init__(
        self,
        entries: List[Dict[str, str]],
        speech_paths: List[str],
        noise_paths: List[str],
        clips_per_rir: int,
        seq_len: float,
        sr: int,
        snr_range: Tuple[float, float],
        noisy_proportion: float,
        num_angles: int,
        max_rad_value: float,
        rad_resolution: float,
        seed: int = 42,
        estimate_rt60: bool = False,
        film_mode: str = "normal",
        volume_mode: str = "true",
        volume_constant: float = 0.5,
        volume_shuffle_seed: int = 2024,
        randomize: bool = False,
    ) -> None:
        super().__init__()
        if not entries:
            raise ValueError("BUT CSV produced zero entries — nothing to evaluate.")
        if not speech_paths:
            raise ValueError("Speech path list is empty.")
        vmode = str(volume_mode or "true").strip().lower()
        if vmode not in VOLUME_MODES:
            raise ValueError(f"volume_mode must be one of {VOLUME_MODES}, got {volume_mode!r}")
        self.entries = entries
        self.speech_paths = speech_paths
        self.noise_paths = noise_paths
        self.clips_per_rir = clips_per_rir
        self.seq_len = seq_len
        self.sr = sr
        self.snr_range = snr_range
        self.noisy_proportion = noisy_proportion
        self.num_angles = num_angles
        self.max_rad_value = max_rad_value
        self.rad_resolution = rad_resolution
        self.seed = seed
        self.seq_samples = int(seq_len * sr)
        self.estimate_rt60 = estimate_rt60
        self.film_mode = film_mode
        self.volume_mode = vmode
        self.volume_constant = float(volume_constant)
        self.volume_shuffle_seed = int(volume_shuffle_seed)
        self.randomize = randomize
        self._volume_shuffle_donors: Dict[int, float] = {}
        if self.volume_mode == "shuffle" and self.film_mode != "zero":
            vols = []
            for row in entries:
                rp = compute_room_params_from_row(row)
                vols.append(float(rp[7].item()))
            if len(vols) >= 2:
                rng_v = np.random.default_rng(self.volume_shuffle_seed)
                perm = rng_v.permutation(len(vols))
                self._volume_shuffle_donors = {
                    i: float(vols[perm[i]]) for i in range(len(vols))
                }
        # NoiseX files are multi-minute WAVs; reloading them every sample dominates
        # epoch time on slow shared FS. Cache once (15 files ≈ 200MB).
        self._noise_cache: Dict[str, torch.Tensor] = {}
        if noise_paths and noisy_proportion > 0:
            for path in noise_paths:
                try:
                    self._noise_cache[path] = _load_mono_resampled(path, sr)
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.entries) * self.clips_per_rir

    def __getitem__(self, idx: int):
        rir_idx  = idx // self.clips_per_rir
        clip_idx = idx %  self.clips_per_rir
        row = self.entries[rir_idx]

        rng = np.random.default_rng() if self.randomize else np.random.default_rng(self.seed + idx)

        # ── Load RIR ────────────────────────────────────────────────────────
        rir = _load_mono_resampled(row["rir_path"], self.sr)
        scale = rir.abs().max()
        if scale > 0:
            rir = rir / scale

        # Optionally estimate RT60 from the RIR before convolving
        rt60_est: Optional[float] = None
        if self.estimate_rt60:
            rt60_est = estimate_rt60_from_rir(rir, self.sr)

        # ── Load speech ──────────────────────────────────────────────────────
        if self.randomize:
            speech_idx = int(rng.integers(len(self.speech_paths)))
        else:
            speech_idx = (rir_idx * self.clips_per_rir + clip_idx) % len(self.speech_paths)
        speech = _load_mono_resampled(self.speech_paths[speech_idx], self.sr)
        speech = _cut_or_loop(speech, self.seq_samples, rng)
        if speech.abs().max() > 0:
            speech = speech / speech.abs().max()

        # ── Convolve ──────────────────────────────────────────────────────────
        reverb = AF.fftconvolve(speech, rir, mode="full")[: self.seq_samples]
        if reverb.shape[-1] < self.seq_samples:
            reverb = torch.nn.functional.pad(
                reverb, (0, self.seq_samples - reverb.shape[-1])
            )

        # ── Noise ────────────────────────────────────────────────────────────
        if self.noise_paths and rng.random() < self.noisy_proportion:
            noise_path = self.noise_paths[int(rng.integers(len(self.noise_paths)))]
            try:
                noise = self._noise_cache.get(noise_path)
                if noise is None:
                    noise = _load_mono_resampled(noise_path, self.sr)
                    self._noise_cache[noise_path] = noise
                noise = _cut_or_loop(noise, self.seq_samples, rng)
                dp_power    = reverb.pow(2).mean()
                noise_power = noise.pow(2).mean()
                if noise_power > 1e-12 and dp_power > 1e-12:
                    snr_db = float(rng.uniform(self.snr_range[0], self.snr_range[1]))
                    mix = math.sqrt(10 ** (-snr_db / 10) * dp_power.item() / noise_power.item())
                    reverb = reverb + noise * mix
            except Exception:
                pass  # skip noise for this sample rather than crash

        # ── Normalise ────────────────────────────────────────────────────────
        scale = reverb.abs().max()
        if scale > 1e-12:
            reverb = reverb / scale

        # ── Labels & conditioning ────────────────────────────────────────────
        if self.film_mode == "zero":
            film_row = None
            room_params = torch.zeros(_NUM_ROOM_PARAMS, dtype=torch.float32)
        else:
            film_row = row.get("_film_row") or row
            room_params = compute_room_params_from_row(
                film_row,
                rt60_override=rt60_est if film_row is row else None,
            )
            # Volume-channel-only ablation (does not touch RT60 / size / mic).
            vmode = self.volume_mode
            if vmode == "drop":
                room_params = room_params[:7].clone()
            elif vmode == "zero":
                room_params = room_params.clone()
                room_params[7] = 0.0
            elif vmode == "constant":
                room_params = room_params.clone()
                room_params[7] = self.volume_constant
            elif vmode == "random":
                room_params = room_params.clone()
                room_params[7] = float(rng.uniform(0.0, 1.0))
            elif vmode == "shuffle":
                room_params = room_params.clone()
                if rir_idx in self._volume_shuffle_donors:
                    room_params[7] = self._volume_shuffle_donors[rir_idx]
                else:
                    room_params[7] = float(
                        np.random.default_rng(
                            self.volume_shuffle_seed + rir_idx
                        ).uniform(0.0, 1.0)
                    )
        angle_cls    = angle_to_class(float(row["angle_deg"]), self.num_angles)
        radius_cls   = radius_to_class(
            float(row["radius_m"]), self.max_rad_value, self.rad_resolution
        )

        return (
            reverb.unsqueeze(0),                          # [1, T]
            torch.tensor(angle_cls,  dtype=torch.long),
            torch.tensor(radius_cls, dtype=torch.long),
            room_params,                                  # [8]
            row["rir_path"],
            float(row["angle_deg"]),
            float(row["radius_m"]),
            clip_idx,
            row.get("group_key", "?"),
            (film_row or {}).get("group_key", self.film_mode),
        )


# ---------------------------------------------------------------------------
# Confusion-matrix helpers  (copied from confusion_matrix.py to keep this
# script self-contained)
# ---------------------------------------------------------------------------

def _accumulate(matrix: np.ndarray, preds: torch.Tensor, labels: torch.Tensor) -> int:
    num_classes = matrix.shape[0]
    valid = labels >= 0
    skipped = int((~valid).sum().item())
    if valid.any():
        lbl = labels[valid].detach().cpu().numpy()
        prd = preds[valid].detach().cpu().numpy()
        ok  = (lbl >= 0) & (lbl < num_classes) & (prd >= 0) & (prd < num_classes)
        np.add.at(matrix, (lbl[ok], prd[ok]), 1)
    return skipped


def _global_mae(matrix: np.ndarray, unit_scale: float) -> float:
    total = matrix.sum()
    if total == 0:
        return float("nan")
    i_idx, j_idx = np.indices(matrix.shape)
    return float((np.abs(i_idx - j_idx) * matrix).sum() / total) * unit_scale


def _per_true_class_mae(matrix: np.ndarray, unit_scale: float) -> np.ndarray:
    n = matrix.shape[0]
    idx = np.arange(n)
    row_totals = matrix.sum(axis=1).astype(np.float64)
    abs_diff   = np.abs(idx[None, :] - idx[:, None])
    err_sum    = (abs_diff * matrix).sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.where(row_totals > 0, err_sum / row_totals, np.nan)
    return mae * unit_scale


def _plot_matrix(
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    normalize: bool,
    axis_label: str,
    unit_scale: float,
    unit_suffix: str,
) -> None:
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = matrix.astype(np.float64)
    if normalize:
        rs = disp.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        disp = disp / rs
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, None
    im = ax.imshow(disp, origin="lower", aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel(f"Predicted {axis_label}")
    ax.set_ylabel(f"True {axis_label}")
    ax.set_title(title)
    step = max(1, n // 12)
    ticks = np.arange(0, n, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t * unit_scale:g}{unit_suffix}" for t in ticks],
                       rotation=45, ha="right")
    ax.set_yticklabels([f"{t * unit_scale:g}{unit_suffix}" for t in ticks])
    fig.colorbar(im, ax=ax, label="proportion" if normalize else "count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_per_class_mae(
    mae_per_class: np.ndarray,
    row_totals: np.ndarray,
    out_path: Path,
    title: str,
    axis_label: str,
    unit_scale: float,
    unit_suffix: str,
    global_mae: float,
) -> None:
    n = mae_per_class.shape[0]
    x = np.arange(n) * unit_scale
    fig, (ax_mae, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax_mae.bar(x, mae_per_class, width=unit_scale, align="edge",
               color="#1f77b4", edgecolor="none")
    if not math.isnan(global_mae):
        ax_mae.axhline(global_mae, color="crimson", linestyle="--", linewidth=1.2,
                       label=f"overall MAE = {global_mae:.2f}{unit_suffix}")
        ax_mae.legend(loc="upper right")
    ax_mae.set_ylabel(f"MAE ({unit_suffix})")
    ax_mae.set_title(title)
    ax_mae.grid(axis="y", linestyle=":", alpha=0.5)
    ax_hist.bar(x, row_totals, width=unit_scale, align="edge",
                color="#888", edgecolor="none")
    ax_hist.set_ylabel("# samples")
    ax_hist.set_xlabel(f"True {axis_label}")
    ax_hist.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model loading (mirrors confusion_matrix.py)
# ---------------------------------------------------------------------------

def _load_model(config: dict, checkpoint_path: Path, device: torch.device):
    model = initialize_module(config["model"]["path"], args=config["model"]["args"])
    ckpt  = torch.load(checkpoint_path.as_posix(), map_location="cpu")
    sd    = {
        k.replace("module.", ""): v
        for k, v in ckpt["model"].items()
        if not any(x in k for x in ["ops", "params"])
    }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_head(
    prefix: str,
    matrix: np.ndarray,
    out_dir: Path,
    checkpoint_name: str,
    axis_label: str,
    unit_scale: float,
    unit_suffix: str,
    head_name: str,
) -> dict:
    total   = int(matrix.sum())
    correct = int(np.trace(matrix))
    acc     = correct / total if total > 0 else float("nan")
    mae     = _global_mae(matrix, unit_scale)
    mae_per_class = _per_true_class_mae(matrix, unit_scale)
    row_totals    = matrix.sum(axis=1)

    np.save(out_dir / f"{prefix}_confusion_matrix.npy", matrix)
    np.savetxt(out_dir / f"{prefix}_confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    np.save(out_dir / f"{prefix}_mae_per_true_class.npy", mae_per_class)

    title = (
        f"{head_name} confusion matrix (BUT real-room)\n"
        f"ckpt={checkpoint_name}  N={total}  acc={acc:.3f}  MAE={mae:.2f}{unit_suffix}"
    )
    _plot_matrix(matrix, out_dir / f"{prefix}_confusion_matrix.png",
                 title, normalize=False,
                 axis_label=axis_label, unit_scale=unit_scale, unit_suffix=unit_suffix)
    _plot_matrix(matrix, out_dir / f"{prefix}_confusion_matrix_normalized.png",
                 title + " (row-normalized)", normalize=True,
                 axis_label=axis_label, unit_scale=unit_scale, unit_suffix=unit_suffix)
    _plot_per_class_mae(
        mae_per_class, row_totals,
        out_dir / f"{prefix}_mae_per_true_class.png",
        title=f"{head_name} MAE per true class (BUT real-room, ckpt={checkpoint_name})",
        axis_label=axis_label, unit_scale=unit_scale, unit_suffix=unit_suffix,
        global_mae=mae,
    )
    print(f"[{head_name}] N={total}  acc={acc:.4f}  MAE={mae:.3f}{unit_suffix}")
    return {"total": total, "acc": acc, "mae": mae}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(
    config_path: Path,
    checkpoint_path: Path,
    but_csv: Optional[Path],
    dataset_root: Optional[Path],
    speech_txt: Path,
    noise_txt: Optional[Path],
    clips_per_rir: int,
    seq_len: float,
    snr_range: Tuple[float, float],
    noisy_proportion: float,
    out_dir: Path,
    device_str: str,
    no_room_conditioning: bool,
    estimate_rt60: bool,
    batch_size: int,
    num_workers: int,
    seed: int,
    filter_to_film_volume_range: bool = True,
    film_mode: str = "normal",
    volume_mode: Optional[str] = None,
    volume_constant: Optional[float] = None,
    volume_shuffle_seed: Optional[int] = None,
) -> dict:
    # ── Setup ────────────────────────────────────────────────────────────────
    if film_mode not in FILM_MODES:
        raise ValueError(f"film_mode must be one of {FILM_MODES}, got {film_mode!r}")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    config = toml.load(config_path.as_posix())
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    model = _load_model(config, checkpoint_path, device)

    # Resolve evaluation settings from config
    model_args = config["model"]["args"]
    dl_args    = config["dataloader"]["args"]
    sr         = int(config["acoustic"]["args"].get("sr", 16000))
    num_angles     = int(model_args.get("num_angles", 181))
    max_rad_value  = float(model_args.get("max_rad_value", dl_args.get("max_rad_value", 6.0)))
    rad_resolution = float(model_args.get("rad_resolution", dl_args.get("rad_resolution", 0.1)))
    num_radius = int(max_rad_value / rad_resolution) + 1
    # Prefer explicit CLI/API args; else mirror train config dataloader.args.
    resolved_volume_mode = str(
        volume_mode if volume_mode is not None else dl_args.get("volume_mode", "true")
    ).strip().lower()
    if resolved_volume_mode not in VOLUME_MODES:
        raise ValueError(
            f"volume_mode must be one of {VOLUME_MODES}, got {resolved_volume_mode!r}"
        )
    resolved_volume_constant = float(
        volume_constant
        if volume_constant is not None
        else dl_args.get("volume_constant", 0.5)
    )
    resolved_volume_shuffle_seed = int(
        volume_shuffle_seed
        if volume_shuffle_seed is not None
        else dl_args.get("volume_shuffle_seed", seed)
    )

    use_room_cond = (
        bool(dl_args.get("use_room_conditioning", False))
        and not no_room_conditioning
        and film_mode != "off"
    )
    print(
        f"num_angle_classes={num_angles}, "
        f"num_radius_classes={num_radius} (max={max_rad_value}m, res={rad_resolution}m), "
        f"use_room_conditioning={use_room_cond}, "
        f"film_mode={film_mode}, "
        f"volume_mode={resolved_volume_mode}, "
        f"estimate_rt60={estimate_rt60}"
    )

    # STFT frontend
    transformfunc = initialize_module(
        config["acoustic"]["path"], config["acoustic"]["args"]
    )

    # ── Read entries ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    if dataset_root is not None:
        entries = load_rotated_but_entries(dataset_root)
        data_src = f"dataset_root={dataset_root}"
    else:
        if but_csv is None:
            raise ValueError("Either dataset_root or but_csv must be provided.")
        with open(but_csv, newline="", encoding="utf-8") as f:
            entries = list(csv.DictReader(f))
        data_src = f"but_csv={but_csv}"

    n_loaded = len(entries)
    for row in entries:
        row.setdefault("group_key", f"{row.get('room', '?')}/{row.get('mic_id', '?')}")
    # Tag radius-range + volume-range on the full table before dropping anything.
    _, _ = _split_by_model_radius_range(entries, max_rad_value)
    _, _ = _split_by_film_volume_range(entries)

    if dataset_root is not None and entries:
        entries_csv = out_dir / "rotated_entries.csv"
        fieldnames = list(entries[0].keys())
        with open(entries_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)
        print(f"Wrote loaded entries: {entries_csv}")

    in_range_entries, oor_entries = _split_by_model_radius_range(entries, max_rad_value)
    n_skipped_oor = len(oor_entries)
    if oor_entries:
        print(
            f"Skipping {n_skipped_oor}/{n_loaded} RIRs with radius_m > "
            f"{max_rad_value:g} m (model max_rad_value):"
        )
        for row in oor_entries:
            print(
                f"  {row.get('room', '')} {row.get('mic_id', '')} "
                f"{row.get('spk_setup', '')}  radius={row['radius_m']} m  "
                f"{row.get('rir_path', '')}"
            )
    entries = in_range_entries
    n_after_radius = len(entries)

    n_skipped_vol = 0
    vol_min = math.exp(_LOG_VOL_MIN)
    vol_max = math.exp(_LOG_VOL_MAX)
    if filter_to_film_volume_range:
        vol_ok, vol_oor = _split_by_film_volume_range(entries)
        n_skipped_vol = len(vol_oor)
        if vol_oor:
            print(
                f"Skipping {n_skipped_vol}/{n_after_radius} RIRs outside FiLM volume "
                f"window [{vol_min:g}, {vol_max:g}] m^3 or with a room dim > "
                f"{_ROOM_SZ_MAX:g} m:"
            )
            for row in vol_oor:
                print(
                    f"  {row.get('room', '')} {row.get('mic_id', '')} "
                    f"{row.get('spk_setup', '')}  vol={row.get('volume_m3', '?')} m^3  "
                    f"dims=({row.get('room_sz_x') or row.get('room_width')},"
                    f"{row.get('room_sz_y') or row.get('room_depth')},"
                    f"{row.get('room_sz_z') or row.get('room_height')})  "
                    f"{row.get('rir_path', '')}"
                )
        entries = vol_ok

    if not entries:
        raise ValueError(
            "No RIR entries left after radius / FiLM-volume filters "
            f"(max_rad_value={max_rad_value:g} m, "
            f"volume=[{vol_min:g}, {vol_max:g}] m^3)."
        )
    # ── FiLM ablation wiring ─────────────────────────────────────────────────
    # "shuffle" hands each microphone position the room vector of the next one,
    # so audio and conditioning disagree.  Predictions that follow the donor
    # instead of the audio mean the output is read off the geometry, not heard.
    if film_mode == "shuffle":
        groups = sorted({row["group_key"] for row in entries})
        if len(groups) < 2:
            raise ValueError("film_mode='shuffle' needs at least 2 microphone positions")
        donor_of = {g: groups[(i + 1) % len(groups)] for i, g in enumerate(groups)}
        first_row_of = {}
        for row in entries:
            first_row_of.setdefault(row["group_key"], row)
        for row in entries:
            row["_film_row"] = first_row_of[donor_of[row["group_key"]]]
        print("FiLM shuffle (audio group -> conditioning group):")
        for g in groups:
            print(f"  {g}  ->  {donor_of[g]}")

    print(
        f"BUT data: {data_src}  "
        f"({n_loaded} loaded, {n_after_radius} radius-in-range, "
        f"{len(entries)} after volume filter, "
        f"{n_skipped_oor} radius-OOR, {n_skipped_vol} volume-OOR; "
        f"{len(entries)} × {clips_per_rir} clips = "
        f"{len(entries) * clips_per_rir} total samples)"
    )

    # ── Speech + noise paths ─────────────────────────────────────────────────
    with open(speech_txt, encoding="utf-8") as f:
        speech_paths = [os.path.normpath(l.strip()) for l in f if l.strip()]
    print(f"Speech paths: {len(speech_paths)} files from {speech_txt}")

    noise_paths: List[str] = []
    if noise_txt is not None:
        with open(noise_txt, encoding="utf-8") as f:
            noise_paths = [os.path.normpath(l.strip()) for l in f if l.strip()]
        print(f"Noise paths: {len(noise_paths)} files (noisy_proportion={noisy_proportion})")
    else:
        print("Noise: disabled (noise_txt is None)")

    # ── Build dataset & dataloader ────────────────────────────────────────────
    dataset = ButRirDataset(
        entries=entries,
        speech_paths=speech_paths,
        noise_paths=noise_paths,
        clips_per_rir=clips_per_rir,
        seq_len=seq_len,
        sr=sr,
        snr_range=snr_range,
        noisy_proportion=noisy_proportion if noise_paths else 0.0,
        num_angles=num_angles,
        max_rad_value=max_rad_value,
        rad_resolution=rad_resolution,
        seed=seed,
        estimate_rt60=estimate_rt60,
        film_mode=film_mode,
        volume_mode=resolved_volume_mode,
        volume_constant=resolved_volume_constant,
        volume_shuffle_seed=resolved_volume_shuffle_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    angle_matrix  = np.zeros((num_angles, num_angles),   dtype=np.int64)
    radius_matrix = np.zeros((num_radius, num_radius),   dtype=np.int64)

    per_entry_rows: List[dict] = []
    seen = skipped_angle = skipped_radius = 0

    pbar = tqdm(total=len(dataset), desc="BUT eval", unit="sample")

    for batch in loader:
        (
            reverb_wav,     # [B, 1, T]
            angle_cls_gt,   # [B]
            radius_cls_gt,  # [B]
            room_params,    # [B, 8]
            rir_paths,      # tuple of str (B)
            angle_degs,     # [B] float
            radius_ms,      # [B] float
            clip_idxs,      # [B] int
            group_keys,     # tuple of str (B)
            film_groups,    # tuple of str (B)
        ) = batch

        reverb_wav = reverb_wav.to(device)
        rp = room_params.to(device) if use_room_cond else None

        input_complex = transformfunc.stft(reverb_wav, output_type="complex")
        input_ft      = transformfunc.preprocess(input_complex)

        outputs = model(input_ft, room_params=rp)
        # returns (y_spch, y_CTF, y_rev, angle_logits, radius_logits)
        angle_logits  = outputs[3]
        radius_logits = outputs[4]

        angle_preds  = torch.argmax(angle_logits,  dim=1)   # [B]
        radius_preds = torch.argmax(radius_logits, dim=1)   # [B]

        skipped_angle  += _accumulate(angle_matrix,  angle_preds,  angle_cls_gt.to(device))
        skipped_radius += _accumulate(radius_matrix, radius_preds, radius_cls_gt.to(device))

        # Collect per-entry predictions
        B = reverb_wav.shape[0]
        ap_np = angle_preds.cpu().numpy()
        rp_np = radius_preds.cpu().numpy()
        ag_np = angle_cls_gt.numpy()
        rg_np = radius_cls_gt.numpy()
        for b in range(B):
            pred_angle_deg  = class_to_angle(int(ap_np[b]))
            pred_radius_m   = class_to_radius(int(rp_np[b]), rad_resolution)
            gt_angle_deg_f  = float(angle_degs[b])
            gt_radius_m_f   = float(radius_ms[b])
            per_entry_rows.append({
                "rir_path":        rir_paths[b],
                "group_key":       group_keys[b],
                "film_group":      film_groups[b],
                "clip_idx":        int(clip_idxs[b]),
                "gt_angle_deg":    gt_angle_deg_f,
                "gt_angle_class":  int(ag_np[b]),
                "pred_angle_class":int(ap_np[b]),
                "pred_angle_deg":  pred_angle_deg,
                "angle_error_deg": abs(pred_angle_deg - gt_angle_deg_f),
                "gt_radius_m":     gt_radius_m_f,
                "gt_radius_class": int(rg_np[b]),
                "pred_radius_class":int(rp_np[b]),
                "pred_radius_m":   pred_radius_m,
                "radius_error_m":  abs(pred_radius_m - gt_radius_m_f),
            })

        seen += B
        pbar.update(B)

    pbar.close()

    # ── Write outputs ─────────────────────────────────────────────────────────
    ckpt_name = checkpoint_path.name

    angle_stats = _save_head(
        "angle", angle_matrix, out_dir, ckpt_name,
        axis_label="angle", unit_scale=1.0, unit_suffix="\u00b0", head_name="Angle",
    )
    radius_stats = _save_head(
        "radius", radius_matrix, out_dir, ckpt_name,
        axis_label="radius", unit_scale=rad_resolution, unit_suffix="m", head_name="Radius",
    )

    # ── Grouped metrics ──────────────────────────────────────────────────────
    grp   = np.array([r["group_key"] for r in per_entry_rows])
    rir_k = np.array([r["rir_path"]  for r in per_entry_rows])
    angle_grouped = grouped_metrics(
        grp, rir_k,
        np.array([r["pred_angle_deg"] for r in per_entry_rows], dtype=np.float64),
        np.array([r["gt_angle_deg"]   for r in per_entry_rows], dtype=np.float64),
        grid=np.arange(0.0, 180.5, 0.5), seed=seed,
    )
    radius_grouped = grouped_metrics(
        grp, rir_k,
        np.array([r["pred_radius_m"] for r in per_entry_rows], dtype=np.float64),
        np.array([r["gt_radius_m"]   for r in per_entry_rows], dtype=np.float64),
        grid=np.arange(0.0, max_rad_value + 1e-9, rad_resolution), seed=seed,
    )

    grouped_lines: List[str] = []
    for head, unit, m in [("angle", "deg", angle_grouped), ("radius", "m", radius_grouped)]:
        grouped_lines += format_grouped_metrics(m, head, unit) + [""]

    # With shuffled conditioning, does the answer follow the donor geometry?
    if film_mode == "shuffle":
        donor_of = {r["group_key"]: r["film_group"] for r in per_entry_rows}
        m = angle_grouped
        gt_of_group = {
            g: float(m["_rir_gt"][m["_rir_grp"] == g].mean()) for g in set(m["_rir_grp"])
        }
        groups = sorted(gt_of_group)
        pred_g = np.array([m["_rir_pred"][m["_rir_grp"] == g].mean() for g in groups])
        own_g  = np.array([gt_of_group[g] for g in groups])
        don_g  = np.array([gt_of_group.get(donor_of.get(g, g), np.nan) for g in groups])
        grouped_lines += [
            "[angle: shuffled conditioning]",
            f"  corr(pred, own group truth):    {_safe_corr(pred_g, own_g):.3f}",
            f"  corr(pred, donor group truth):  {_safe_corr(pred_g, don_g):.3f}"
            f"   <- higher means the answer follows the room vector",
            "",
        ]

    for head, m in [("angle", angle_grouped), ("radius", radius_grouped)]:
        csv_path = out_dir / f"{head}_per_group.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(m["per_group"][0].keys()))
            w.writeheader()
            w.writerows(m["per_group"])
    print("\n" + "\n".join(grouped_lines))

    # per_entry_preds.csv
    pred_csv = out_dir / "per_entry_preds.csv"
    if per_entry_rows:
        fieldnames = list(per_entry_rows[0].keys())
        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_entry_rows)
        print(f"Per-entry predictions: {pred_csv}")

    # summary.txt
    summary_lines = [
        f"BUT ReverbDB Localization Evaluation",
        f"=====================================",
        f"config:          {config_path}",
        f"checkpoint:      {checkpoint_path}",
        f"data:            {data_src}",
        f"max_rad_value_m:         {max_rad_value}",
        f"film_volume_m3:          [{vol_min:g}, {vol_max:g}]",
        f"film_room_sz_max_m:      {_ROOM_SZ_MAX}",
        f"filter_to_film_volume:   {filter_to_film_volume_range}",
        f"rir_entries_loaded:      {n_loaded}",
        f"rir_entries_radius_ok:   {n_after_radius}",
        f"rir_entries_skipped_oor: {n_skipped_oor}",
        f"rir_entries_vol_ok:      {len(entries)}",
        f"rir_entries_skipped_vol: {n_skipped_vol}",
        f"clips_per_rir:   {clips_per_rir}",
        f"total_samples:   {seen}",
        f"seq_len:         {seq_len} s",
        f"snr_range:       {snr_range} dB",
        f"use_room_cond:   {use_room_cond}",
        f"film_mode:       {film_mode}",
        f"volume_mode:     {resolved_volume_mode}",
        f"volume_constant: {resolved_volume_constant}",
        f"estimate_rt60:   {estimate_rt60}",
        f"",
        f"[angle]",
        f"  num_classes:             {num_angles}",
        f"  samples_counted:         {angle_stats['total']}",
        f"  samples_skipped:         {skipped_angle}",
        f"  top1_accuracy:           {angle_stats['acc']:.6f}",
        f"  mean_abs_error_deg:      {angle_stats['mae']:.4f}",
        f"",
        f"[radius]",
        f"  num_classes:             {num_radius}",
        f"  max_rad_value_m:         {max_rad_value}",
        f"  rad_resolution_m:        {rad_resolution}",
        f"  samples_counted:         {radius_stats['total']}",
        f"  samples_skipped:         {skipped_radius}",
        f"  top1_accuracy:           {radius_stats['acc']:.6f}",
        f"  mean_abs_error_m:        {radius_stats['mae']:.4f}",
        f"",
    ] + grouped_lines
    summary_txt = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_txt + "\n")
    print(f"\nSaved all artefacts to: {out_dir}")
    print("\n" + summary_txt)

    return {
        "checkpoint": str(checkpoint_path),
        "out_dir": str(out_dir),
        "film_mode": film_mode,
        "volume_mode": resolved_volume_mode,
        "angle": angle_grouped,
        "radius": radius_grouped,
    }


# ---------------------------------------------------------------------------
# Entry point — edit these variables, then run: python scripts/eval_but.py
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    # ── models ─────────────────────────────────────────────────────────────
    # Both jobs use the one-shot toml: the OpenSLR fine-tune toml sets
    # use_room_conditioning=false, but BUT eval must keep FiLM on.
    one_shot_toml = project_root / "config/Rec-RIR-heads-only-epoch35-straight-to-15m-2gpu.toml"
    zfree_toml = project_root / "config/Rec-RIR-heads-only-epoch35-straight-to-15m-zfree-2gpu.toml"
    mix_toml = project_root / "config/Rec-RIR-heads-only-epoch35-straight-to-15m-zfree-mix-openslr-2gpu.toml"
    corrupt_toml = project_root / "config/Rec-RIR-heads-only-epoch35-straight-to-15m-zfree-mix-ir-corrupt-2gpu.toml"
    models = {
        "one-shot": (
            one_shot_toml,
            Path("/storage/reie/experiments/epoch35-straight-to-15m-2gpu/ckpt/best.tar"),
            project_root / "results/but_epoch35_straight_to_15m",
        ),
        "z-free": (
            zfree_toml,
            Path("/storage/reie/experiments/epoch35-straight-to-15m-zfree-2gpu/ckpt/best.tar"),
            project_root / "results/but_epoch35_straight_to_15m_zfree",
        ),
        "mix-openslr": (
            mix_toml,
            Path("/storage/reie/experiments/epoch35-zfree-mix-openslr-2gpu/ckpt/best.tar"),
            project_root / "results/but_epoch35_zfree_mix_openslr",
        ),
        "mix-ir-corrupt": (
            corrupt_toml,
            Path("/storage/reie/experiments/epoch35-mix-ir-corrupt-2gpu/ckpt/best.tar"),
            project_root / "results/but_epoch35_mix_ir_corrupt",
        ),
    }

    # The measured rooms, and their simulated twins from
    # scripts/gen_but_twin_rirs.py: same geometry and labels, synthetic
    # acoustics, walking toward the training distribution.
    datasets = {
        "measured":     Path("/storage/reie/Filteted and rotated BUT"),
        "twin-exact":   Path("/storage/reie/data/but-twin/exact"),
        "twin-rt60":    Path("/storage/reie/data/but-twin/rt60"),
        "twin-heights": Path("/storage/reie/data/but-twin/heights"),
    }

    # FiLM modes:
    #   normal   the entry's own geometry
    #   off      no conditioning at all (room_params=None, FiLM is identity)
    #   zero     an all-zero room vector
    #   shuffle  another microphone position's room vector
    # Together they say whether the output is read from the audio or from the
    # geometry the model is handed.
    #
    # Each (model, dataset, film_mode) writes to the model's out_dir with the
    # dataset and mode appended; measured + normal keeps the bare name.
    # z-free / one-shot / mix-openslr already scored. IR-corrupt FT from mix
    # needs the BUT 45-slice + twin-exact gates (normal + shuffle).
    jobs = [
        ("mix-ir-corrupt", "measured",   ["normal", "shuffle"]),
        ("mix-ir-corrupt", "twin-exact", ["normal", "shuffle"]),
    ]

    # ── data ───────────────────────────────────────────────────────────────
    # Rotated-BUT style trees come from `datasets` above.  Set but_csv and
    # clear `datasets` to use the legacy matched-CSV path instead.
    but_csv: Optional[Path] = None  # e.g. project_root / "config/but_matched.csv"

    speech_txt = project_root / "config/reie_validation_speech_quick.txt"
    noise_txt: Optional[Path] = None  # e.g. project_root / "config/reie_train_noise.txt"
    clips_per_rir = 20
    seq_len = 4.0
    snr_range: Tuple[float, float] = (20.0, 20.0)
    noisy_proportion = 1.0

    # ── runtime ────────────────────────────────────────────────────────────
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    no_room_conditioning = False
    estimate_rt60 = False
    # Match but_filtered_rotated: radius ≤ max_rad_value, then vol/dim in the
    # training FiLM window (45 rooms × 20 clips).
    filter_to_film_volume_range = True
    batch_size = 4
    num_workers = 0
    seed = 42

    # ── validate ───────────────────────────────────────────────────────────
    for model_name, dataset_name, _modes in jobs:
        if model_name not in models:
            print(f"Error: unknown model {model_name!r}", file=sys.stderr)
            sys.exit(1)
        if dataset_name not in datasets:
            print(f"Error: unknown dataset {dataset_name!r}", file=sys.stderr)
            sys.exit(1)
        config_path, checkpoint_path, _out = models[model_name]
        for path, label in [
            (config_path, "config_path"),
            (checkpoint_path, "checkpoint_path"),
            (datasets[dataset_name], "dataset_root"),
        ]:
            if not path.exists():
                print(f"Error: {label} does not exist: {path}", file=sys.stderr)
                sys.exit(1)
    if not speech_txt.exists():
        print(f"Error: speech_txt does not exist: {speech_txt}", file=sys.stderr)
        sys.exit(1)

    if noise_txt is not None and not noise_txt.exists():
        print(f"Error: noise_txt does not exist: {noise_txt}", file=sys.stderr)
        sys.exit(1)

    results = []
    for model_name, dataset_name, film_modes in jobs:
        config_path, checkpoint_path, out_dir = models[model_name]
        dataset_root = datasets[dataset_name]
        for film_mode in film_modes:
            suffix = "_".join(
                ([] if dataset_name == "measured" else [dataset_name.replace("-", "_")])
                + ([] if film_mode == "normal" else [film_mode])
            )
            mode_dir = out_dir if not suffix else Path(f"{out_dir}_{suffix}")
            print(
                f"\n===== {model_name}  data={dataset_name}  film={film_mode}"
                f"  -> {mode_dir.name} ====="
            )
            res = run(
                config_path          = config_path.expanduser().absolute(),
                checkpoint_path      = checkpoint_path.expanduser().absolute(),
                but_csv              = but_csv.expanduser().absolute() if but_csv else None,
                dataset_root         = dataset_root.expanduser().absolute(),
                speech_txt           = speech_txt.expanduser().absolute(),
                noise_txt            = noise_txt.expanduser().absolute() if noise_txt else None,
                clips_per_rir        = clips_per_rir,
                seq_len              = seq_len,
                snr_range            = snr_range,
                noisy_proportion     = noisy_proportion,
                out_dir              = mode_dir.expanduser().absolute(),
                device_str           = device_str,
                no_room_conditioning = no_room_conditioning,
                estimate_rt60        = estimate_rt60,
                batch_size           = batch_size,
                num_workers          = num_workers,
                seed                 = seed,
                filter_to_film_volume_range = filter_to_film_volume_range,
                film_mode            = film_mode,
            )
            res["name"] = model_name
            res["dataset"] = dataset_name
            results.append(res)

    table = comparison_table(results)
    print(table)
    out_table = project_root / "results/but_mix_ir_corrupt_comparison.txt"
    out_table.write_text(table + "\n")
    print(f"wrote {out_table}")


def comparison_table(results: List[dict]) -> str:
    """One row per (checkpoint, dataset, film_mode), plus the audio-free oracles."""
    hdr = (
        f"{'model':15s} {'data':13s} {'film':8s} {'ang MAE':>9s} {'95% CI':>14s} "
        f"{'r within':>9s} {'r between':>10s} {'pred sd':>8s} "
        f"{'rad MAE':>8s} {'r within':>9s}"
    )
    lines = ["", "=" * len(hdr), hdr, "-" * len(hdr)]
    for r in results:
        a, d = r["angle"], r["radius"]
        lines.append(
            f"{r['name']:15s} {r.get('dataset', 'measured'):13s} {r['film_mode']:8s} "
            f"{a['mae_clip']:8.2f}d "
            f"{'[%5.1f,%5.1f]' % (a['mae_ci_lo'], a['mae_ci_hi']):>14s} "
            f"{a['corr_within_group']:9.2f} {a['corr_between_groups']:10.2f} "
            f"{a['pred_std_within_group']:7.1f}d "
            f"{d['mae_clip']:7.2f}m {d['corr_within_group']:9.2f}"
        )
    # With shuffled conditioning, how close is the answer to what the donor
    # microphone position gets with its own audio?  Small numbers mean the
    # recording is not being used.
    normal_of = {
        (r["name"], r.get("dataset", "measured")):
            {g["group"]: g["pred_mean"] for g in r["angle"]["per_group"]}
        for r in results if r["film_mode"] == "normal"
    }
    for r in results:
        key = (r["name"], r.get("dataset", "measured"))
        if r["film_mode"] != "shuffle" or key not in normal_of:
            continue
        ref = normal_of[key]
        groups = sorted(ref)
        donor = {g: groups[(i + 1) % len(groups)] for i, g in enumerate(groups)}
        shuf = np.array([g["pred_mean"] for g in r["angle"]["per_group"]])
        keys = [g["group"] for g in r["angle"]["per_group"]]
        lines.append(
            f"{r['name']:15s} {key[1]:13s} {'shuffle vs donor':17s} "
            f"{np.abs(shuf - np.array([ref[donor[k]] for k in keys])).mean():7.1f}d"
            f"   from the donor's own answer,"
            f" {np.abs(shuf - np.array([ref[k] for k in keys])).mean():.1f}d from its own"
        )

    a0 = results[0]["angle"]
    lines += [
        "-" * len(hdr),
        f"{'oracle':15s} {'per-group constant':31s} {a0['baseline_group_constant_mae']:7.2f}d"
        f"   (angle, never hears the audio)",
        f"{'oracle':15s} {'one constant':31s} {a0['baseline_constant_mae']:7.2f}d",
        "=" * len(hdr),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
