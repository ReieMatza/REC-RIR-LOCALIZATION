# @description: Inference-only FiLM parameter importance analysis for BiSpatialNet.
#
# Two analyses are run on the validation set:
#   1. Permutation importance — globally shuffle one room_param dim at a time,
#      measure angle/radius MAE degradation vs baseline.
#   2. Gradient sensitivity — mean |d(true-class logit)/d(room_params_j)|
#      over the validation set (local linear sensitivity, no training).
#
# The 8 FiLM inputs (indices into room_params) are:
#   0: RT60
#   1: room_sz_x,  2: room_sz_y,  3: room_sz_z
#   4: mic_center_x, 5: mic_center_y, 6: mic_center_z
#   7: volume (log-normalised)
#
# Outputs are written to <experiment_dir>/film_importance/:
#   permutation_importance.png
#   gradient_sensitivity.png
#   summary.txt
#
# Usage:
#   python film_importance.py /path/to/experiment_dir [options]
#
# Run from the REC-RIR-LOCALIZATION folder so that dataset.* imports resolve.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import toml
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trainer_inferencer.utils import initialize_module
from confusion_matrix import (
    _find_config,
    _resolve_checkpoint,
    _load_model,
    _build_valid_dataloader,
    _global_mae,
    _accumulate,
)

import warnings
warnings.filterwarnings("ignore")


PARAM_NAMES = [
    "RT60",
    "room_sz_x",
    "room_sz_y",
    "room_sz_z",
    "mic_x",
    "mic_y",
    "mic_z",
    "volume",
]
NUM_PARAMS = len(PARAM_NAMES)


# ---------------------------------------------------------------------------
# Inference pass (no_grad)
# ---------------------------------------------------------------------------

def _inference_pass(
    model: torch.nn.Module,
    valid_dataloader,
    transformfunc,
    device: torch.device,
    num_angles: int,
    num_radius: int,
    num_samples: Optional[int],
    use_room_conditioning: bool,
    room_params_all: Optional[torch.Tensor] = None,
    desc: str = "inference",
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    """Run a single no_grad forward pass over the validation set.

    When ``room_params_all`` is ``None`` (baseline pass) and
    ``use_room_conditioning`` is True, room_params are taken from the batch and
    also collected into a [N, 8] tensor that is returned as the third element.

    When ``room_params_all`` is provided (permutation pass), it overrides the
    batch room_params slice-by-slice in dataloader order. The third return value
    is None in that case.
    """
    angle_matrix = np.zeros((num_angles, num_angles), dtype=np.int64)
    radius_matrix = np.zeros((num_radius, num_radius), dtype=np.int64)
    rp_list: List[torch.Tensor] = []
    is_baseline = room_params_all is None
    offset = 0
    seen = 0

    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=desc, leave=False):
            noisy_wav = batch[0].to(device)
            angle_class = batch[5].to(device)
            radius_class = batch[6].to(device)
            B = noisy_wav.shape[0]

            if room_params_all is not None:
                rp = room_params_all[offset : offset + B].to(device)
                offset += B
            elif use_room_conditioning:
                rp = batch[7].to(device)
                if is_baseline:
                    rp_list.append(batch[7].cpu())
            else:
                rp = None

            input_complex = transformfunc.stft(noisy_wav, output_type="complex")
            input_ft = transformfunc.preprocess(input_complex)

            outputs = model(input_ft, room_params=rp)
            angle_preds = torch.argmax(outputs[3], dim=1)
            radius_preds = torch.argmax(outputs[4], dim=1)

            _accumulate(angle_matrix, angle_preds, angle_class)
            _accumulate(radius_matrix, radius_preds, radius_class)

            seen += B
            if num_samples is not None and seen >= num_samples:
                break

    collected = torch.cat(rp_list, dim=0) if rp_list else None
    return angle_matrix, radius_matrix, collected


# ---------------------------------------------------------------------------
# Gradient sensitivity pass
# ---------------------------------------------------------------------------

def _gradient_pass(
    model: torch.nn.Module,
    valid_dataloader,
    transformfunc,
    device: torch.device,
    num_samples: Optional[int],
    use_room_conditioning: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean |d(true-class logit) / d(room_params_j)| for each head.

    Processes ONE sample at a time to minimise GPU memory: the computation
    graph (which stores all layer activations for backprop) is built and freed
    per-sample. This costs ~2× forward-pass memory for a single sample, rather
    than holding graphs for an entire batch simultaneously.

    Two separate backward passes share the same graph via ``retain_graph=True``
    (angle head first, then radius head), after which the graph is released.

    Returns:
        angle_sens:  [NUM_PARAMS] float64 — per-dim mean |grad|, angle head
        radius_sens: [NUM_PARAMS] float64 — per-dim mean |grad|, radius head
    """
    angle_accum = np.zeros(NUM_PARAMS, dtype=np.float64)
    radius_accum = np.zeros(NUM_PARAMS, dtype=np.float64)
    n = 0

    if not use_room_conditioning:
        return angle_accum, radius_accum

    # Release any cache fragmentation left over from the no_grad baseline pass.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for batch in tqdm(valid_dataloader, desc="Gradient sensitivity", leave=False):
        # Keep raw batch tensors on CPU; move to device one sample at a time.
        noisy_cpu = batch[0]        # [B, 1, T]
        angle_cpu = batch[5]        # [B]
        radius_cpu = batch[6]       # [B]
        rp_cpu = batch[7]           # [B, 8]
        B = noisy_cpu.shape[0]

        for b in range(B):
            a_cls = int(angle_cpu[b].item())
            r_cls = int(radius_cpu[b].item())

            noisy_b = noisy_cpu[b : b + 1].to(device)
            with torch.no_grad():
                input_complex = transformfunc.stft(noisy_b, output_type="complex")
                input_ft = transformfunc.preprocess(input_complex)
            input_ft = input_ft.detach()

            # Leaf we differentiate through: shape [1, 8]
            rp = rp_cpu[b : b + 1].to(device).detach().clone().requires_grad_(True)

            # Single forward — builds computation graph only for this one sample
            outputs = model(input_ft, room_params=rp)
            angle_logits = outputs[3]   # [1, num_angles]
            radius_logits = outputs[4]  # [1, num_radius]

            # --- angle gradient --------------------------------------------
            grads_a = torch.zeros(NUM_PARAMS, device=device)
            if a_cls >= 0:
                score_a = angle_logits[0, a_cls]
                (ga,) = torch.autograd.grad(score_a, rp, retain_graph=True)
                grads_a = ga[0]   # [8]

            # --- radius gradient -------------------------------------------
            grads_r = torch.zeros(NUM_PARAMS, device=device)
            if r_cls >= 0:
                score_r = radius_logits[0, r_cls]
                (gr,) = torch.autograd.grad(score_r, rp)
                grads_r = gr[0]   # [8]

            angle_accum += grads_a.detach().abs().cpu().numpy()
            radius_accum += grads_r.detach().abs().cpu().numpy()

            # Explicitly free the computation graph and sample tensors.
            del outputs, input_ft, input_complex, noisy_b, rp, grads_a, grads_r
            if device.type == "cuda":
                torch.cuda.empty_cache()

            n += 1
            if num_samples is not None and n >= num_samples:
                break

        if num_samples is not None and n >= num_samples:
            break

    if n > 0:
        angle_accum /= n
        radius_accum /= n

    return angle_accum, radius_accum


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_permutation_importance(
    angle_delta: np.ndarray,
    radius_delta: np.ndarray,
    baseline_angle_mae: float,
    baseline_radius_mae: float,
    out_path: Path,
    checkpoint_name: str,
    n_samples: int,
):
    """Two-panel horizontal bar chart showing MAE degradation per feature."""
    # Sort by combined normalised importance (both panels share this order)
    norm_a = angle_delta / (baseline_angle_mae + 1e-12)
    norm_r = radius_delta / (baseline_radius_mae + 1e-12)
    combined = (norm_a + norm_r) / 2.0
    order = np.argsort(combined)          # ascending → least important first (bottom of chart)
    names = [PARAM_NAMES[i] for i in order]

    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Permutation Feature Importance  |  ckpt={checkpoint_name}  N={n_samples}",
        fontsize=11,
    )

    y = np.arange(NUM_PARAMS)
    bar_kw = dict(height=0.6, align="center")

    # Angle panel
    vals_a = angle_delta[order]
    colors_a = ["#d7191c" if v > 0 else "#abd9e9" for v in vals_a]
    ax_a.barh(y, vals_a, color=colors_a, **bar_kw)
    ax_a.axvline(0, color="k", linewidth=0.8)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(names)
    ax_a.set_xlabel("Angle MAE Δ (°)")
    ax_a.set_title(f"Angle head  (baseline MAE={baseline_angle_mae:.2f}°)")
    ax_a.grid(axis="x", linestyle=":", alpha=0.5)

    # Radius panel
    vals_r = radius_delta[order]
    colors_r = ["#d7191c" if v > 0 else "#abd9e9" for v in vals_r]
    ax_r.barh(y, vals_r, color=colors_r, **bar_kw)
    ax_r.axvline(0, color="k", linewidth=0.8)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels(names)
    ax_r.set_xlabel("Radius MAE Δ (m)")
    ax_r.set_title(f"Radius head  (baseline MAE={baseline_radius_mae:.4f} m)")
    ax_r.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_gradient_sensitivity(
    angle_sens: np.ndarray,
    radius_sens: np.ndarray,
    out_path: Path,
    checkpoint_name: str,
    n_samples: int,
):
    """Two-panel horizontal bar chart of mean |gradient| per feature."""
    # Sort by combined sensitivity (normalised to [0,1] per head)
    def _norm01(x):
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-12)

    combined = (_norm01(angle_sens) + _norm01(radius_sens)) / 2.0
    order = np.argsort(combined)
    names = [PARAM_NAMES[i] for i in order]

    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Gradient Sensitivity  |d(logit)/d(param)|  |  ckpt={checkpoint_name}  N={n_samples}",
        fontsize=11,
    )

    y = np.arange(NUM_PARAMS)
    bar_kw = dict(height=0.6, align="center", color="#4393c3")

    ax_a.barh(y, angle_sens[order], **bar_kw)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(names)
    ax_a.set_xlabel("|d(angle logit) / d(room_param)|")
    ax_a.set_title("Angle head")
    ax_a.grid(axis="x", linestyle=":", alpha=0.5)

    ax_r.barh(y, radius_sens[order], color="#d6604d", height=0.6, align="center")
    ax_r.set_yticks(y)
    ax_r.set_yticklabels(names)
    ax_r.set_xlabel("|d(radius logit) / d(room_param)|")
    ax_r.set_title("Radius head")
    ax_r.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    experiment_dir: Path,
    checkpoint: str,
    config_override: Optional[str],
    num_samples: Optional[int],
    device_str: str,
    out_subdir_name: str,
    seed: int,
    skip_gradient: bool = False,
):
    # ---- setup -------------------------------------------------------------
    config_path = _find_config(experiment_dir, config_override)
    print(f"Config:     {config_path}")
    config = toml.load(config_path.as_posix())

    checkpoint_path = _resolve_checkpoint(experiment_dir, checkpoint)
    print(f"Checkpoint: {checkpoint_path}")

    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device:     {device}")

    transformfunc = initialize_module(
        config["acoustic"]["path"], config["acoustic"]["args"]
    )
    model = _load_model(config, checkpoint_path, device)

    valid_dataloader = _build_valid_dataloader(config, snr_override=None)

    model_args = config["model"]["args"]
    dl_args = config["dataloader"]["args"]

    num_angles = int(model_args.get("num_angles", 181))
    angle_unit_scale = 1.0      # one class == one degree

    max_rad_value = float(model_args.get("max_rad_value", dl_args.get("max_rad_value", 6.0)))
    rad_resolution = float(model_args.get("rad_resolution", dl_args.get("rad_resolution", 0.1)))
    num_radius = int(max_rad_value / rad_resolution) + 1
    radius_unit_scale = rad_resolution

    use_room_conditioning = bool(dl_args.get("use_room_conditioning", False))
    print(
        f"num_angle_classes={num_angles}, num_radius_classes={num_radius}, "
        f"use_room_conditioning={use_room_conditioning}"
    )

    if not use_room_conditioning:
        print(
            "[warn] use_room_conditioning=False — the dataloader does not emit "
            "room_params, so both analyses will return zeros.  "
            "Set use_room_conditioning=true in the config to get meaningful results."
        )

    out_dir = experiment_dir / out_subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass 0: baseline inference (no_grad) + collect room_params --------
    print("\n[1/10] Baseline inference pass...")
    baseline_a_mat, baseline_r_mat, all_rp = _inference_pass(
        model, valid_dataloader, transformfunc, device,
        num_angles, num_radius, num_samples,
        use_room_conditioning=use_room_conditioning,
        room_params_all=None,
        desc="Baseline",
    )
    baseline_angle_mae = _global_mae(baseline_a_mat, angle_unit_scale)
    baseline_radius_mae = _global_mae(baseline_r_mat, radius_unit_scale)
    # Use the actual seen count from room_params shape for permutation limit
    n_rp = all_rp.shape[0] if all_rp is not None else 0

    print(
        f"  Baseline angle MAE : {baseline_angle_mae:.3f}°  "
        f"(acc={np.trace(baseline_a_mat)/max(baseline_a_mat.sum(),1):.4f})"
    )
    print(
        f"  Baseline radius MAE: {baseline_radius_mae:.4f} m  "
        f"(acc={np.trace(baseline_r_mat)/max(baseline_r_mat.sum(),1):.4f})"
    )

    # ---- Pass 1: gradient sensitivity (with grad) --------------------------
    if skip_gradient:
        print("\n[2/10] Gradient sensitivity pass... SKIPPED (--no-gradient)")
        angle_grad_sens = np.zeros(NUM_PARAMS, dtype=np.float64)
        radius_grad_sens = np.zeros(NUM_PARAMS, dtype=np.float64)
    else:
        print("\n[2/10] Gradient sensitivity pass (one sample at a time)...")
        angle_grad_sens, radius_grad_sens = _gradient_pass(
            model, valid_dataloader, transformfunc, device,
            num_samples=n_rp if n_rp > 0 else num_samples,
            use_room_conditioning=use_room_conditioning,
        )
        print(
            f"  Angle   grad sens: {dict(zip(PARAM_NAMES, angle_grad_sens.round(5)))}"
        )
        print(
            f"  Radius  grad sens: {dict(zip(PARAM_NAMES, radius_grad_sens.round(5)))}"
        )

    # ---- Passes 2–9: permutation importance --------------------------------
    perm_angle_delta = np.zeros(NUM_PARAMS, dtype=np.float64)
    perm_radius_delta = np.zeros(NUM_PARAMS, dtype=np.float64)

    rng = np.random.default_rng(seed=seed)

    if all_rp is not None and n_rp > 0:
        for j in range(NUM_PARAMS):
            print(f"\n[{j + 3}/10] Permutation pass: {PARAM_NAMES[j]} ...")
            perm_idx = rng.permutation(n_rp)
            shuffled_rp = all_rp.clone()
            shuffled_rp[:, j] = all_rp[perm_idx, j]

            a_mat, r_mat, _ = _inference_pass(
                model, valid_dataloader, transformfunc, device,
                num_angles, num_radius,
                num_samples=n_rp,           # same number of samples as baseline
                use_room_conditioning=True,
                room_params_all=shuffled_rp,
                desc=f"Perm {PARAM_NAMES[j]}",
            )
            perm_angle_delta[j] = _global_mae(a_mat, angle_unit_scale) - baseline_angle_mae
            perm_radius_delta[j] = _global_mae(r_mat, radius_unit_scale) - baseline_radius_mae
            print(
                f"  {PARAM_NAMES[j]:12s}  angle Δ={perm_angle_delta[j]:+.3f}°  "
                f"radius Δ={perm_radius_delta[j]:+.4f} m"
            )
    else:
        print(
            "[warn] No room_params collected (use_room_conditioning=False). "
            "Permutation deltas will all be zero."
        )

    # ---- Plots -------------------------------------------------------------
    print("\nPlotting...")
    ckpt_name = checkpoint_path.name

    _plot_permutation_importance(
        perm_angle_delta, perm_radius_delta,
        baseline_angle_mae, baseline_radius_mae,
        out_dir / "permutation_importance.png",
        checkpoint_name=ckpt_name,
        n_samples=n_rp,
    )
    _plot_gradient_sensitivity(
        angle_grad_sens, radius_grad_sens,
        out_dir / "gradient_sensitivity.png",
        checkpoint_name=ckpt_name,
        n_samples=n_rp,
    )

    # ---- Summary -----------------------------------------------------------
    def _fmt_table(arr: np.ndarray, fmt: str = ".6f") -> str:
        lines = []
        for name, val in zip(PARAM_NAMES, arr):
            lines.append(f"    {name:14s}: {val:{fmt}}")
        return "\n".join(lines)

    # Sort by combined normalised permutation importance
    norm_a = perm_angle_delta / (baseline_angle_mae + 1e-12)
    norm_r = perm_radius_delta / (baseline_radius_mae + 1e-12)
    combined_perm = (norm_a + norm_r) / 2.0
    rank_perm = np.argsort(-combined_perm) + 1   # rank 1 = most important

    summary = (
        f"experiment_dir : {experiment_dir}\n"
        f"config         : {config_path}\n"
        f"checkpoint     : {checkpoint_path}\n"
        f"n_samples      : {n_rp}\n"
        f"rng_seed       : {seed}\n"
        f"\n"
        f"[baseline]\n"
        f"  angle_mae_deg  : {baseline_angle_mae:.6f}\n"
        f"  angle_acc      : {np.trace(baseline_a_mat)/max(baseline_a_mat.sum(),1):.6f}\n"
        f"  radius_mae_m   : {baseline_radius_mae:.6f}\n"
        f"  radius_acc     : {np.trace(baseline_r_mat)/max(baseline_r_mat.sum(),1):.6f}\n"
        f"\n"
        f"[permutation_importance]\n"
        f"  angle_mae_delta_deg (higher = more important):\n"
        f"{_fmt_table(perm_angle_delta)}\n"
        f"  radius_mae_delta_m:\n"
        f"{_fmt_table(perm_radius_delta)}\n"
        f"  combined_normalised_rank (1=most important):\n"
    )
    for i in range(NUM_PARAMS):
        summary += f"    {PARAM_NAMES[i]:14s}: rank {int(rank_perm[i])}\n"

    summary += (
        f"\n"
        f"[gradient_sensitivity]\n"
        f"  angle_head  |d(logit)/d(param)|:\n"
        f"{_fmt_table(angle_grad_sens)}\n"
        f"  radius_head |d(logit)/d(param)|:\n"
        f"{_fmt_table(radius_grad_sens)}\n"
    )

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary)
    print(f"\nSaved summary: {summary_path}")
    print(f"All outputs in: {out_dir}")

    # Also print a quick ranked table to stdout
    print("\n--- Permutation importance ranking (combined, normalised) ---")
    order = np.argsort(-combined_perm)
    for rank, idx in enumerate(order, 1):
        print(
            f"  {rank}. {PARAM_NAMES[idx]:14s}  "
            f"angle Δ={perm_angle_delta[idx]:+.3f}°  "
            f"radius Δ={perm_radius_delta[idx]:+.4f} m"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Inference-only FiLM parameter importance: "
            "permutation importance + gradient sensitivity."
        )
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
        help="Limit samples for analysis (default: full validation set)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cpu, cuda, or cuda:N (default: cuda if available)",
    )
    parser.add_argument(
        "--out-subdir",
        type=str, default="film_importance",
        help="Subdir under experiment dir to write outputs (default: film_importance)",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="RNG seed for global permutation shuffle (default: 42)",
    )
    parser.add_argument(
        "--no-gradient",
        action="store_true",
        default=False,
        help=(
            "Skip the gradient sensitivity pass (saves GPU memory). "
            "The permutation importance results are still computed."
        ),
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
        out_subdir_name=args.out_subdir,
        seed=args.seed,
        skip_gradient=args.no_gradient,
    )
