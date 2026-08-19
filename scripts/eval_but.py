#!/usr/bin/env python3
"""
eval_but.py  —  Localization evaluation on BUT ReverbDB real-room RIRs.

Reads the matched BUT CSV produced by match_but_dataset.py, builds a
speech-convolution dataset (multiple speech clips per RIR), runs the trained
BiSpatialNet model, and produces the same confusion-matrix / MAE artefacts as
confusion_matrix.py, plus a per-entry prediction CSV for detailed inspection.

The critical difference from the standard evaluation pipeline is that room_params
(needed for FiLM conditioning) are derived directly from CSV metadata columns
instead of being read from the synthetic .npz files.

Usage:
  cd /storage/reie/REC-RIR-LOCALIZATION
  python scripts/eval_but.py \\
      -c config/Rec-RIR-heads-only-fullroom-y0.toml \\
      -k /storage/reie/experiments/<exp>/ckpt/best.tar \\
      --out_dir results/but_eval

Full options:
  python scripts/eval_but.py --help
"""

from __future__ import annotations

import argparse
import csv
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
# room_params from CSV row
# ---------------------------------------------------------------------------

def compute_room_params_from_row(
    row: Dict[str, str],
    rt60_override: Optional[float] = None,
) -> torch.Tensor:
    """Build the 8-element room_params tensor from a but_matched.csv row.

    Replicates MyDataset._extract_room_params (dataset_RecRIR.py lines
    151-180) using CSV columns instead of npz fields.

    The training model (fullroom-edge-y0) always places the mic near the
    y=0 wall, so room_sz=[Lx,Ly,Lz] and mic_center=[mx,my,mz] with my
    small.  BUT mics can sit near any of the four walls.  We rotate the
    horizontal plane so the near-wall axis always becomes y (index 1),
    matching compute_angle_radius_local() in match_but_dataset.py which
    builds angle labels in the same local frame.

    Wall → training-frame mapping (BUT depth=x-axis, width=y-axis):
        depth0   : Lx=room_width,  Ly=room_depth,  mx=mic_w,  my=mic_d
        depthMax : Lx=room_width,  Ly=room_depth,  mx=mic_w,  my=room_d−mic_d
        width0   : Lx=room_depth,  Ly=room_width,  mx=mic_d,  my=mic_w  [identity]
        widthMax : Lx=room_depth,  Ly=room_width,  mx=mic_d,  my=room_w−mic_w

    If rt60_override is not None it replaces the CSV 'rt60' column value,
    allowing the caller to supply an RT60 estimated directly from the RIR.

    Vector layout:
        [0]   rt60_norm           clip((RT60 - 0.1) / 1.4,       0, 1)
        [1-3] room_sz_norm        clip([Lx,Ly,Lz] / 15,          0, 1)
        [4-6] mic_center_norm     clip([mx/Lx, my/Ly, mz/Lz],    0, 1)
        [7]   volume_norm         clip((ln(vol)−ln10)/(ln1000−ln10), 0, 1)
    """
    rt60 = rt60_override if (rt60_override is not None and math.isfinite(rt60_override)) \
           else float(row["rt60"])
    room_d = float(row["room_depth"])
    room_w = float(row["room_width"])
    room_h = float(row["room_height"])
    mic_d  = float(row["mic_depth"])
    mic_w  = float(row["mic_width"])
    mic_h  = float(row["mic_height"])
    nearest_wall = row["nearest_wall"].strip()

    # Rotate horizontal axes so the near-wall direction maps to y (index 1),
    # matching the training data convention.
    # if nearest_wall == "depth0":
    #     Lx, Ly = room_w, room_d
    #     mx, my = mic_w, mic_d
    # elif nearest_wall == "depthMax":
    Lx, Ly = room_w, room_d
    mx, my = room_w - mic_w, mic_d
    # elif nearest_wall == "width0":
    #     Lx, Ly = room_d, room_w
    #     mx, my = mic_d, mic_w
    # else:  # widthMax
    #     Lx, Ly = room_d, room_w
    #     mx, my = mic_d, room_w - mic_w

    rt60_norm = float(np.clip((rt60 - _RT60_MIN) / (_RT60_MAX - _RT60_MIN), 0.0, 1.0))

    room_sz = np.array([Lx, Ly, room_h], dtype=np.float64)
    room_sz_norm = np.clip(room_sz / _ROOM_SZ_MAX, 0.0, 1.0)

    mic_center = np.array([mx, my, mic_h], dtype=np.float64)
    mic_center_norm = np.clip(mic_center / (room_sz + _EPS), 0.0, 1.0)

    volume_m3 = room_d * room_w * room_h  # volume is rotation-invariant
    log_vol = math.log(max(volume_m3, _EPS))
    volume_norm = float(np.clip(
        (log_vol - _LOG_VOL_MIN) / (_LOG_VOL_MAX - _LOG_VOL_MIN), 0.0, 1.0
    ))

    params = np.concatenate(
        [[rt60_norm], room_sz_norm, mic_center_norm, [volume_norm]]
    ).astype(np.float32)
    return torch.from_numpy(params)  # [8]


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
# Audio helpers
# ---------------------------------------------------------------------------

def _load_mono_resampled(path: str, target_sr: int) -> torch.Tensor:
    """Load any WAV → mono float32 at target_sr. Returns 1-D tensor."""
    wav, sr = torchaudio.load(path)          # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mix to mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)                    # [T]


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
    ) -> None:
        super().__init__()
        if not entries:
            raise ValueError("BUT CSV produced zero entries — nothing to evaluate.")
        if not speech_paths:
            raise ValueError("Speech path list is empty.")
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

    def __len__(self) -> int:
        return len(self.entries) * self.clips_per_rir

    def __getitem__(self, idx: int):
        rir_idx  = idx // self.clips_per_rir
        clip_idx = idx %  self.clips_per_rir
        row = self.entries[rir_idx]

        rng = np.random.default_rng(self.seed + idx)

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
                noise = _load_mono_resampled(noise_path, self.sr)
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
        room_params  = compute_room_params_from_row(row, rt60_override=rt60_est)
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
    but_csv: Path,
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
) -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
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

    use_room_cond = (
        bool(dl_args.get("use_room_conditioning", False))
        and not no_room_conditioning
    )
    print(
        f"num_angle_classes={num_angles}, "
        f"num_radius_classes={num_radius} (max={max_rad_value}m, res={rad_resolution}m), "
        f"use_room_conditioning={use_room_cond}, "
        f"estimate_rt60={estimate_rt60}"
    )

    # STFT frontend
    transformfunc = initialize_module(
        config["acoustic"]["path"], config["acoustic"]["args"]
    )

    # ── Read BUT CSV ─────────────────────────────────────────────────────────
    with open(but_csv, newline="", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))
    print(f"BUT CSV: {but_csv}  ({len(entries)} entries × {clips_per_rir} clips = "
          f"{len(entries) * clips_per_rir} total samples)")

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
        print("Noise: disabled (--noise_txt not provided)")

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
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = checkpoint_path.name

    angle_stats = _save_head(
        "angle", angle_matrix, out_dir, ckpt_name,
        axis_label="angle", unit_scale=1.0, unit_suffix="\u00b0", head_name="Angle",
    )
    radius_stats = _save_head(
        "radius", radius_matrix, out_dir, ckpt_name,
        axis_label="radius", unit_scale=rad_resolution, unit_suffix="m", head_name="Radius",
    )

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
        f"but_csv:         {but_csv}  ({len(entries)} RIR entries)",
        f"clips_per_rir:   {clips_per_rir}",
        f"total_samples:   {seen}",
        f"seq_len:         {seq_len} s",
        f"snr_range:       {snr_range} dB",
        f"use_room_cond:   {use_room_cond}",
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
    ]
    summary_txt = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_txt + "\n")
    print(f"\nSaved all artefacts to: {out_dir}")
    print("\n" + summary_txt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── required / model ───────────────────────────────────────────────────
    p.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to training config .toml (e.g. config/Rec-RIR-heads-only-fullroom-y0.toml)",
    )
    p.add_argument(
        "-k", "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint .tar",
    )

    # ── data ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--but_csv",
        type=Path,
        default=Path("config/but_matched.csv"),
        help="BUT matched CSV (default: config/but_matched.csv)",
    )
    p.add_argument(
        "--speech_txt",
        type=Path,
        default=Path("config/reie_train_speech_quick.txt"),
        help="Speech pathlist .txt (default: config/reie_train_speech_quick.txt)",
    )
    p.add_argument(
        "--noise_txt",
        type=Path,
        default=None,
        help="Noise pathlist .txt; omit to disable noise (default: None)",
    )
    p.add_argument(
        "--clips_per_rir",
        type=int,
        default=20,
        help="Number of different speech clips per RIR (default: 20)",
    )
    p.add_argument(
        "--seq_len",
        type=float,
        default=4.0,
        help="Audio segment length in seconds (default: 4.0)",
    )
    p.add_argument(
        "--snr",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[20.0, 20.0],
        help="SNR range in dB when noise is enabled (default: 20 20)",
    )
    p.add_argument(
        "--noisy_proportion",
        type=float,
        default=1.0,
        help="Fraction of samples that receive additive noise (default: 1.0)",
    )

    # ── output ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/but_eval"),
        help="Output directory for artefacts (default: results/but_eval)",
    )

    # ── runtime ────────────────────────────────────────────────────────────
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cpu, cuda, or cuda:N (default: cuda if available)",
    )
    p.add_argument(
        "--no_room_conditioning",
        action="store_true",
        help="Disable FiLM room conditioning even if the config enables it",
    )
    p.add_argument(
        "--estimate_rt60",
        action="store_true",
        help=(
            "Estimate RT60 from each RIR (Schroeder T20×3) and use it in "
            "the FiLM room_params instead of the CSV metadata value. "
            "Useful when metadata RT60 is known to differ from the real "
            "room decay (e.g. BUT metadata vs wideband EDC)."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="DataLoader batch size (default: 4)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for speech/noise selection (default: 42)",
    )

    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    # Validate paths
    for attr, label in [
        ("config", "--config"), ("checkpoint", "--checkpoint"),
        ("but_csv", "--but_csv"), ("speech_txt", "--speech_txt"),
    ]:
        p = getattr(args, attr)
        if not p.exists():
            print(f"Error: {label} path does not exist: {p}", file=sys.stderr)
            sys.exit(1)
    if args.noise_txt is not None and not args.noise_txt.exists():
        print(f"Error: --noise_txt path does not exist: {args.noise_txt}", file=sys.stderr)
        sys.exit(1)

    run(
        config_path          = args.config.expanduser().absolute(),
        checkpoint_path      = args.checkpoint.expanduser().absolute(),
        but_csv              = args.but_csv.expanduser().absolute(),
        speech_txt           = args.speech_txt.expanduser().absolute(),
        noise_txt            = args.noise_txt.expanduser().absolute() if args.noise_txt else None,
        clips_per_rir        = args.clips_per_rir,
        seq_len              = args.seq_len,
        snr_range            = tuple(args.snr),
        noisy_proportion     = args.noisy_proportion,
        out_dir              = args.out_dir.expanduser().absolute(),
        device_str           = args.device,
        no_room_conditioning = args.no_room_conditioning,
        estimate_rt60        = args.estimate_rt60,
        batch_size           = args.batch_size,
        num_workers          = args.num_workers,
        seed                 = args.seed,
    )


if __name__ == "__main__":
    main()
