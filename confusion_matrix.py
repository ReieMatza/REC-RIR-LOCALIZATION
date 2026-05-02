# @description: Build confusion matrices + per-bin MAE plots for the
# angle and radius classification heads.
#
# Given an experiment directory produced by train.py, this script:
#   1. Locates the config .toml saved inside the experiment dir (or uses --config).
#   2. Loads the requested checkpoint (default: ckpt/best.tar).
#   3. Runs the model on the validation dataloader defined in the config.
#   4. Accumulates argmax predictions vs. ground-truth classes for BOTH heads.
#   5. Writes into <experiment_dir>/confusion_matrix/:
#        angle_confusion_matrix.{npy,csv,png}            (raw counts)
#        angle_confusion_matrix_normalized.png           (row-normalized)
#        angle_mae_per_true_class.{npy,png}              (per-true-bin MAE, deg)
#        radius_confusion_matrix.{npy,csv,png}           (raw counts)
#        radius_confusion_matrix_normalized.png          (row-normalized)
#        radius_mae_per_true_class.{npy,png}             (per-true-bin MAE, m)
#        summary.txt
#
# Usage:
#   python confusion_matrix.py /storage/reie/experiments/<exp_name>
#   python confusion_matrix.py <exp_dir> --checkpoint epoch90.tar --num-samples 2000
#
# Run from the REC-RIR-LOCALIZATION folder so that `dataset.*` and
# `trainer_inferencer.*` imports resolve.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import toml
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trainer_inferencer.utils import initialize_module

import warnings
warnings.filterwarnings("ignore")


def _find_config(experiment_dir: Path, config_override: Optional[str]) -> Path:
    """Return the config .toml to use.

    If --config is given, use it. Otherwise pick the most recently modified
    .toml at the top level of the experiment dir (train.py dumps a
    timestamped config there on every run).
    """
    if config_override:
        p = Path(config_override).expanduser().absolute()
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        return p

    candidates = sorted(
        experiment_dir.glob("*.toml"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .toml config found at top of {experiment_dir}. "
            f"Pass --config explicitly."
        )
    return candidates[0]


def _resolve_checkpoint(experiment_dir: Path, checkpoint: str) -> Path:
    """Accept either a bare filename (relative to <exp>/ckpt/) or a full path."""
    p = Path(checkpoint).expanduser()
    if p.is_absolute() and p.exists():
        return p
    in_ckpt_dir = experiment_dir / "ckpt" / checkpoint
    if in_ckpt_dir.exists():
        return in_ckpt_dir
    rel = experiment_dir / checkpoint
    if rel.exists():
        return rel
    raise FileNotFoundError(
        f"Checkpoint not found. Tried: {p}, {in_ckpt_dir}, {rel}"
    )


def _load_model(config: dict, checkpoint_path: Path, device: torch.device):
    model = initialize_module(config["model"]["path"], args=config["model"]["args"])
    ckpt = torch.load(checkpoint_path.as_posix(), map_location="cpu")
    state_dict = ckpt["model"]
    # Strip DDP "module." prefix and drop any flop-counter bookkeeping tensors.
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
        if not any(x in k for x in ["ops", "params"])
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model.to(device).eval()
    return model


def _build_valid_dataloader(config: dict, snr_override: Optional[Tuple[float, float]]):
    """Build the validation dataloader exactly as the trainer would (single-process)."""
    dl_cfg = config["dataloader"]
    dl_cfg["args"]["rank"] = 0
    dl_cfg["args"]["sr"] = config["acoustic"]["args"]["sr"]
    if snr_override is not None:
        dl_cfg["args"]["snr_range"] = list(snr_override)
    dataloader = initialize_module(dl_cfg["path"], args=dl_cfg["args"])
    return dataloader.valid_dataloader


def _accumulate(
    matrix: np.ndarray,
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> int:
    """Add (label, pred) pairs into `matrix`. Returns number of invalid labels skipped.

    Invalid labels are those with value < 0 (the dataset uses -1 as the
    "no label" sentinel when a RIR doesn't appear in the localization CSV).
    """
    num_classes = matrix.shape[0]
    valid = labels >= 0
    skipped = int((~valid).sum().item())
    if valid.any():
        labels_np = labels[valid].detach().cpu().numpy()
        preds_np = preds[valid].detach().cpu().numpy()
        in_range = (
            (labels_np >= 0)
            & (labels_np < num_classes)
            & (preds_np >= 0)
            & (preds_np < num_classes)
        )
        np.add.at(matrix, (labels_np[in_range], preds_np[in_range]), 1)
    return skipped


def _global_mae(matrix: np.ndarray, unit_scale: float) -> float:
    """Mean absolute error over all counted samples, in (class-index * unit_scale)."""
    total = matrix.sum()
    if total == 0:
        return float("nan")
    i_idx, j_idx = np.indices(matrix.shape)
    return float((np.abs(i_idx - j_idx) * matrix).sum() / total) * unit_scale


def _per_true_class_mae(matrix: np.ndarray, unit_scale: float) -> np.ndarray:
    """MAE for each true class (row), in (class-index * unit_scale).

    Returns an array of length num_classes. Rows with zero samples are NaN
    so they don't drag the plot to zero.
    """
    num_classes = matrix.shape[0]
    idx = np.arange(num_classes)
    row_totals = matrix.sum(axis=1).astype(np.float64)
    # |i - j| abs-difference per row: shape (num_classes, num_classes)
    abs_diff = np.abs(idx[None, :] - idx[:, None])  # [num_classes, num_classes]
    # Weighted sum of errors per row.
    err_sum = (abs_diff * matrix).sum(axis=1).astype(np.float64)
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
):
    """Plot a class x class confusion matrix.

    Tick labels are rendered in physical units (e.g. degrees or meters) by
    multiplying the class index by `unit_scale` and annotating with
    `unit_suffix`. The matrix itself is still indexed by class.
    """
    num_classes = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    display = matrix.astype(np.float64)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display = display / row_sums
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, None
    im = ax.imshow(
        display,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel(f"Predicted {axis_label}")
    ax.set_ylabel(f"True {axis_label}")
    ax.set_title(title)
    # At most ~12 ticks to stay readable for large matrices (e.g. 181x181).
    step = max(1, num_classes // 12)
    ticks = np.arange(0, num_classes, step)
    tick_labels = [f"{t * unit_scale:g}{unit_suffix}" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
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
):
    """Plot MAE(class) as a bar chart; annotate class ticks in physical units."""
    num_classes = mae_per_class.shape[0]
    x = np.arange(num_classes)
    x_units = x * unit_scale

    fig, (ax_mae, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax_mae.bar(x_units, mae_per_class, width=unit_scale, align="edge",
               color="#1f77b4", edgecolor="none")
    if not np.isnan(global_mae):
        ax_mae.axhline(
            global_mae, color="crimson", linestyle="--", linewidth=1.2,
            label=f"overall MAE = {global_mae:.2f}{unit_suffix}",
        )
        ax_mae.legend(loc="upper right")
    ax_mae.set_ylabel(f"MAE ({unit_suffix})")
    ax_mae.set_title(title)
    ax_mae.grid(axis="y", linestyle=":", alpha=0.5)

    ax_hist.bar(x_units, row_totals, width=unit_scale, align="edge",
                color="#888888", edgecolor="none")
    ax_hist.set_ylabel("# samples")
    ax_hist.set_xlabel(f"True {axis_label}")
    ax_hist.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def run(
    experiment_dir: Path,
    checkpoint: str,
    config_override: Optional[str],
    num_samples: Optional[int],
    device_str: str,
    snr_override: Optional[Tuple[float, float]],
    out_subdir_name: str,
):
    config_path = _find_config(experiment_dir, config_override)
    print(f"Using config: {config_path}")
    config = toml.load(config_path.as_posix())

    checkpoint_path = _resolve_checkpoint(experiment_dir, checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")

    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # acoustic frontend (STFT preprocess)
    transformfunc = initialize_module(
        config["acoustic"]["path"], config["acoustic"]["args"]
    )

    model = _load_model(config, checkpoint_path, device)

    valid_dataloader = _build_valid_dataloader(config, snr_override)

    # ---- resolve class counts & unit scales from config ----------------
    model_args = config["model"]["args"]
    dl_args = config["dataloader"]["args"]

    num_angles = int(model_args.get("num_angles", 181))
    # Angle axis is folded to [0, 180] in the dataset (see _angle_to_class),
    # and with num_angles == 181 one class == one degree.
    angle_unit_scale = 1.0
    angle_unit_suffix = "\u00b0"  # degree sign

    max_rad_value = float(
        model_args.get("max_rad_value", dl_args.get("max_rad_value", 6.0))
    )
    rad_resolution = float(
        model_args.get("rad_resolution", dl_args.get("rad_resolution", 0.1))
    )
    num_radius = int(max_rad_value / rad_resolution) + 1
    radius_unit_scale = rad_resolution
    radius_unit_suffix = "m"

    use_room_conditioning = bool(
        dl_args.get("use_room_conditioning", False)
    )
    print(
        f"num_angle_classes={num_angles}, "
        f"num_radius_classes={num_radius} "
        f"(max_rad={max_rad_value}m, res={rad_resolution}m), "
        f"use_room_conditioning={use_room_conditioning}"
    )

    # ---- run inference & accumulate ------------------------------------
    angle_matrix = np.zeros((num_angles, num_angles), dtype=np.int64)
    radius_matrix = np.zeros((num_radius, num_radius), dtype=np.int64)
    seen = 0
    skipped_angle = 0
    skipped_radius = 0
    target = num_samples if num_samples else len(valid_dataloader.dataset)
    pbar = tqdm(total=target, desc="Confusion matrix", unit="sample")

    for batch in valid_dataloader:
        # dataset returns 8 items:
        # (noisy, reverb, target, src_fpath, rir_path, angle_class, radius_class, room_params)
        noisy_wav = batch[0].to(device)
        angle_class = batch[5].to(device)
        radius_class = batch[6].to(device)
        room_params = batch[7].to(device) if use_room_conditioning else None

        input_complex = transformfunc.stft(noisy_wav, output_type="complex")
        input_ft = transformfunc.preprocess(input_complex)

        outputs = model(input_ft, room_params=room_params)
        # Forward returns (est_spch, est_ctf, est_reverb, angle_logits, radius_logits)
        angle_logits = outputs[3]
        radius_logits = outputs[4]
        angle_preds = torch.argmax(angle_logits, dim=1)
        radius_preds = torch.argmax(radius_logits, dim=1)

        skipped_angle += _accumulate(angle_matrix, angle_preds, angle_class)
        skipped_radius += _accumulate(radius_matrix, radius_preds, radius_class)

        batch_size = noisy_wav.shape[0]
        seen += batch_size
        pbar.update(batch_size)
        if num_samples is not None and seen >= num_samples:
            break
    pbar.close()

    # ---- write outputs -------------------------------------------------
    out_dir = experiment_dir / out_subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_head(
        prefix: str,
        matrix: np.ndarray,
        axis_label: str,
        unit_scale: float,
        unit_suffix: str,
        head_name: str,
    ):
        total = int(matrix.sum())
        correct = int(np.trace(matrix))
        acc = correct / total if total > 0 else float("nan")
        mae = _global_mae(matrix, unit_scale)
        mae_per_class = _per_true_class_mae(matrix, unit_scale)
        row_totals = matrix.sum(axis=1)

        # Raw artifacts
        np.save(out_dir / f"{prefix}_confusion_matrix.npy", matrix)
        np.savetxt(
            out_dir / f"{prefix}_confusion_matrix.csv",
            matrix, fmt="%d", delimiter=",",
        )
        np.save(out_dir / f"{prefix}_mae_per_true_class.npy", mae_per_class)

        title_common = (
            f"{head_name} confusion matrix\n"
            f"ckpt={checkpoint_path.name}  "
            f"N={total}  acc={acc:.3f}  MAE={mae:.2f}{unit_suffix}"
        )
        _plot_matrix(
            matrix, out_dir / f"{prefix}_confusion_matrix.png",
            title_common, normalize=False,
            axis_label=axis_label, unit_scale=unit_scale, unit_suffix=unit_suffix,
        )
        _plot_matrix(
            matrix, out_dir / f"{prefix}_confusion_matrix_normalized.png",
            title_common + " (row-normalized)", normalize=True,
            axis_label=axis_label, unit_scale=unit_scale, unit_suffix=unit_suffix,
        )
        _plot_per_class_mae(
            mae_per_class, row_totals,
            out_dir / f"{prefix}_mae_per_true_class.png",
            title=f"{head_name} MAE per true class (ckpt={checkpoint_path.name})",
            axis_label=axis_label,
            unit_scale=unit_scale,
            unit_suffix=unit_suffix,
            global_mae=mae,
        )

        print(
            f"[{head_name}] counted={total}  acc={acc:.4f}  "
            f"MAE={mae:.3f}{unit_suffix}"
        )
        return {"total": total, "acc": acc, "mae": mae}

    angle_stats = _save_head(
        "angle", angle_matrix,
        axis_label="angle", unit_scale=angle_unit_scale,
        unit_suffix=angle_unit_suffix, head_name="Angle",
    )
    radius_stats = _save_head(
        "radius", radius_matrix,
        axis_label="radius", unit_scale=radius_unit_scale,
        unit_suffix=radius_unit_suffix, head_name="Radius",
    )

    # ---- summary -------------------------------------------------------
    summary = (
        f"experiment_dir: {experiment_dir}\n"
        f"config: {config_path}\n"
        f"checkpoint: {checkpoint_path}\n"
        f"samples_seen: {seen}\n"
        f"snr_range: {config['dataloader']['args'].get('snr_range')}\n"
        f"\n"
        f"[angle]\n"
        f"  num_classes: {num_angles}\n"
        f"  samples_counted: {angle_stats['total']}\n"
        f"  samples_skipped_invalid_label: {skipped_angle}\n"
        f"  top1_accuracy: {angle_stats['acc']:.6f}\n"
        f"  mean_abs_error_deg: {angle_stats['mae']:.6f}\n"
        f"\n"
        f"[radius]\n"
        f"  num_classes: {num_radius}\n"
        f"  max_rad_value_m: {max_rad_value}\n"
        f"  rad_resolution_m: {rad_resolution}\n"
        f"  samples_counted: {radius_stats['total']}\n"
        f"  samples_skipped_invalid_label: {skipped_radius}\n"
        f"  top1_accuracy: {radius_stats['acc']:.6f}\n"
        f"  mean_abs_error_m: {radius_stats['mae']:.6f}\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(f"\nSaved confusion matrix artifacts to: {out_dir}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build angle+radius confusion matrices and per-bin MAE plots "
            "from a trained experiment dir."
        ),
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to an experiment directory created by train.py",
    )
    parser.add_argument(
        "--checkpoint", "-k",
        type=str, default="best.tar",
        help="Checkpoint filename under <exp>/ckpt/ or absolute path (default: best.tar)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str, default=None,
        help="Override config .toml (default: most recent .toml in experiment dir)",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int, default=None,
        help="Limit on samples to evaluate (default: full validation set)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cpu, cuda, or cuda:N (default: cuda if available)",
    )
    parser.add_argument(
        "--snr", type=float, nargs=2, default=None,
        metavar=("SNR_MIN", "SNR_MAX"),
        help="Override dataloader snr_range (default: use config value)",
    )
    parser.add_argument(
        "--out-subdir", type=str, default="confusion_matrix",
        help="Subdir (under experiment dir) to write outputs (default: confusion_matrix)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    exp_dir = Path(args.experiment_dir).expanduser().absolute()
    if not exp_dir.is_dir():
        print(f"Experiment dir does not exist: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    run(
        experiment_dir=exp_dir,
        checkpoint=args.checkpoint,
        config_override=args.config,
        num_samples=args.num_samples,
        device_str=args.device,
        snr_override=tuple(args.snr) if args.snr is not None else None,
        out_subdir_name=args.out_subdir,
    )
