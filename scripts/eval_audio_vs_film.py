#!/usr/bin/env python3
"""A3: audio / metadata ablations on the locked FiLM model.

Caches the fixed 1024-clip @20 dB set once, then evaluates:
  normal, meta_zero, meta_permute, audio_silence, audio_noise, audio_permute

Optionally evaluates separately trained audio-only / metadata-only controls
(and mirror-folded angle MAE for audio-only).

Usage (from repo root, venv activated):
  python scripts/eval_audio_vs_film.py \\
    --config config/Rec-RIR-heads-only-free-mic-z-h100-miczfree-s2033-2gpu.toml \\
    --checkpoint /storage/reie/experiments/free-mic-z-h100-miczfree-s2033/ckpt/best.tar \\
    --out-dir /storage/reie/experiments/free-mic-z-h100-miczfree-s2033/a3_eval \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import toml
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trainer_inferencer.utils import initialize_module  # noqa: E402

import warnings

warnings.filterwarnings("ignore")


INPUT_CONDITIONS = (
    "normal",
    "meta_zero",
    "meta_permute",
    "audio_silence",
    "audio_noise",
    "audio_permute",
)


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
        shuffle=False,
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
    return loader, {"n_rir": n, "snr_range": list(snr_range), "seed": int(seed)}


@torch.no_grad()
def _cache_clips(
    dataloader,
    transformfunc,
    device: torch.device,
    *,
    num_samples: int,
    use_room_conditioning: bool,
) -> Dict[str, torch.Tensor]:
    """Materialise STFT inputs + labels (+ room params) for the fixed clip set."""
    inputs: List[torch.Tensor] = []
    angles: List[torch.Tensor] = []
    radii: List[torch.Tensor] = []
    rooms: List[torch.Tensor] = []
    seen = 0
    pbar = tqdm(total=num_samples, desc="cache", unit="sample")
    for batch in dataloader:
        noisy_wav = batch[0]
        angle_class = batch[5]
        radius_class = batch[6]
        room_params = batch[7] if use_room_conditioning else None

        if seen + noisy_wav.shape[0] > num_samples:
            keep = num_samples - seen
            noisy_wav = noisy_wav[:keep]
            angle_class = angle_class[:keep]
            radius_class = radius_class[:keep]
            if room_params is not None:
                room_params = room_params[:keep]

        input_complex = transformfunc.stft(noisy_wav.to(device), output_type="complex")
        input_ft = transformfunc.preprocess(input_complex).cpu()
        inputs.append(input_ft)
        angles.append(angle_class.cpu())
        radii.append(radius_class.cpu())
        if room_params is not None:
            rooms.append(room_params.cpu())

        bs = int(noisy_wav.shape[0])
        seen += bs
        pbar.update(bs)
        if seen >= num_samples:
            break
    pbar.close()

    out = {
        "input_ft": torch.cat(inputs, dim=0),
        "angle_class": torch.cat(angles, dim=0),
        "radius_class": torch.cat(radii, dim=0),
    }
    if rooms:
        out["room_params"] = torch.cat(rooms, dim=0)
    print(
        f"cached: {out['input_ft'].shape[0]} clips  "
        f"input={tuple(out['input_ft'].shape)}  "
        f"room={'yes' if 'room_params' in out else 'no'}"
    )
    return out


def _mirror_fold_err(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> torch.Tensor:
    """min(|pred-θ|, |pred-(180-θ)|) per sample, degrees (class indices = degrees)."""
    mirror = 180.0 - true_deg.float()
    e1 = torch.abs(pred_deg.float() - true_deg.float())
    e2 = torch.abs(pred_deg.float() - mirror)
    return torch.minimum(e1, e2)


@torch.no_grad()
def _eval_cached(
    model: torch.nn.Module,
    cache: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    condition: str,
    rad_resolution: float,
    batchsize: int,
    permute_seed: int,
    noise_seed: int,
    report_mirror: bool = False,
) -> Dict[str, Any]:
    n = int(cache["input_ft"].shape[0])
    angle_err_sum = 0.0
    radius_err_sum = 0.0
    mirror_err_sum = 0.0
    n_angle = 0
    n_radius = 0
    n_mirror = 0

    # Global permutations (deterministic): one shuffle of the full set, applied
    # so each sample gets another sample's audio or metadata.
    g = torch.Generator()
    g.manual_seed(int(permute_seed))
    perm = torch.randperm(n, generator=g)

    noise_g = torch.Generator()
    noise_g.manual_seed(int(noise_seed))

    pbar = tqdm(total=n, desc=condition, unit="sample")
    for start in range(0, n, batchsize):
        end = min(start + batchsize, n)
        idx = torch.arange(start, end)

        x = cache["input_ft"][idx].to(device)
        angle_class = cache["angle_class"][idx].to(device)
        radius_class = cache["radius_class"][idx].to(device)
        room = cache["room_params"][idx].to(device) if "room_params" in cache else None

        if condition == "normal":
            pass
        elif condition == "meta_zero":
            if room is None:
                raise RuntimeError("meta_zero requires room_params in cache")
            room = torch.zeros_like(room)
        elif condition == "meta_permute":
            if room is None:
                raise RuntimeError("meta_permute requires room_params in cache")
            room = cache["room_params"][perm[idx]].to(device)
        elif condition == "audio_silence":
            x = torch.zeros_like(x)
        elif condition == "audio_noise":
            # Complex STFT features: replace with Gaussian noise of matching shape.
            x = torch.randn(x.shape, generator=noise_g, dtype=x.dtype, device="cpu").to(device)
        elif condition == "audio_permute":
            x = cache["input_ft"][perm[idx]].to(device)
        else:
            raise ValueError(f"unknown condition: {condition}")

        outputs = model(x, room_params=room)
        angle_preds = torch.argmax(outputs[3], dim=1)
        radius_preds = torch.argmax(outputs[4], dim=1)

        valid_a = angle_class >= 0
        if valid_a.any():
            angle_err_sum += torch.abs(
                angle_preds[valid_a] - angle_class[valid_a]
            ).float().sum().item()
            n_angle += int(valid_a.sum().item())
            if report_mirror:
                mirror_err_sum += _mirror_fold_err(
                    angle_preds[valid_a], angle_class[valid_a]
                ).sum().item()
                n_mirror += int(valid_a.sum().item())

        valid_r = radius_class >= 0
        if valid_r.any():
            radius_err_sum += torch.abs(
                radius_preds[valid_r] - radius_class[valid_r]
            ).float().sum().item()
            n_radius += int(valid_r.sum().item())

        pbar.update(end - start)
    pbar.close()

    out = {
        "condition": condition,
        "n_clips": n,
        "n_angle_valid": n_angle,
        "n_radius_valid": n_radius,
        "angle_mae_deg": float(angle_err_sum / n_angle) if n_angle else float("nan"),
        "distance_mae_m": float((radius_err_sum / n_radius) * rad_resolution)
        if n_radius
        else float("nan"),
        "permute_seed": int(permute_seed),
        "noise_seed": int(noise_seed) if condition == "audio_noise" else None,
    }
    if report_mirror:
        out["mirror_folded_angle_mae_deg"] = (
            float(mirror_err_sum / n_mirror) if n_mirror else float("nan")
        )
    print(
        f"  {condition:16s}  ang {out['angle_mae_deg']:.2f}°  "
        f"rad {out['distance_mae_m']:.3f}m"
        + (
            f"  mirror {out['mirror_folded_angle_mae_deg']:.2f}°"
            if report_mirror
            else ""
        )
    )
    return out


@torch.no_grad()
def _eval_metadata_only_mlp(
    mlp: torch.nn.Module,
    cache: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    rad_resolution: float,
    batchsize: int,
) -> Dict[str, Any]:
    """Evaluate a metadata→(angle,radius) MLP on cached room_params."""
    if "room_params" not in cache:
        raise RuntimeError("metadata-only eval needs room_params")
    n = int(cache["room_params"].shape[0])
    angle_err_sum = 0.0
    radius_err_sum = 0.0
    n_angle = 0
    n_radius = 0
    mlp.eval()
    for start in range(0, n, batchsize):
        end = min(start + batchsize, n)
        room = cache["room_params"][start:end].to(device)
        angle_class = cache["angle_class"][start:end].to(device)
        radius_class = cache["radius_class"][start:end].to(device)
        angle_logits, radius_logits = mlp(room)
        angle_preds = torch.argmax(angle_logits, dim=1)
        radius_preds = torch.argmax(radius_logits, dim=1)
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
    return {
        "condition": "metadata_only_mlp",
        "n_clips": n,
        "n_angle_valid": n_angle,
        "n_radius_valid": n_radius,
        "angle_mae_deg": float(angle_err_sum / n_angle) if n_angle else float("nan"),
        "distance_mae_m": float((radius_err_sum / n_radius) * rad_resolution)
        if n_radius
        else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="A3 audio vs FiLM ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument(
        "--fixed1024-list",
        default=(
            "config/rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w09-miczfree-s2033/"
            "reie-test-rir-fixed1024.txt"
        ),
    )
    parser.add_argument("--fixed1024-seed", type=int, default=2024)
    parser.add_argument("--permute-seed", type=int, default=2024)
    parser.add_argument("--noise-seed", type=int, default=2024)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--cache-path", default=None, help="Optional .pt path to save/load cache")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(INPUT_CONDITIONS),
        help="Subset of input ablations to run",
    )
    parser.add_argument(
        "--audio-only-checkpoint",
        default=None,
        help="Optional separately trained audio-only (use_film=false) ckpt",
    )
    parser.add_argument(
        "--audio-only-config",
        default=None,
        help="Config for audio-only model (defaults to --config with use_film off)",
    )
    parser.add_argument(
        "--metadata-only-checkpoint",
        default=None,
        help="Optional metadata-only MLP .pt (state_dict or {'model': ...})",
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
    meta_seed = int(config["meta"]["seed"])
    test_speech = dl_args["src_pathlist"][2]
    test_noise = dl_args["noise_pathlist"][2]
    test_seq_len = float(dl_args["seq_lenlist"][2])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} config={config_path} ckpt={ckpt_path}")

    transformfunc = initialize_module(
        config["acoustic"]["path"], config["acoustic"]["args"]
    )
    model, ckpt_epoch = _load_model(config, ckpt_path, device)
    print(f"checkpoint epoch={ckpt_epoch}")

    fixed_list_path = Path(args.fixed1024_list)
    if not fixed_list_path.is_absolute():
        fixed_list_path = _REPO_ROOT / fixed_list_path

    cache_path = (
        Path(args.cache_path).expanduser().absolute()
        if args.cache_path
        else out_dir / "fixed1024_cache.pt"
    )
    if cache_path.exists():
        print(f"loading cache from {cache_path}")
        cache = torch.load(cache_path.as_posix(), map_location="cpu")
    else:
        loader, meta = _build_eval_dataloader(
            config,
            snr_range=[20.0, 20.0],
            rir_pathlist=str(fixed_list_path),
            src_pathlist=test_speech,
            noise_pathlist=test_noise,
            batchsize=args.batchsize,
            seed=int(args.fixed1024_seed),
            seq_len=test_seq_len,
        )
        cache = _cache_clips(
            loader,
            transformfunc,
            device,
            num_samples=int(args.num_samples),
            use_room_conditioning=use_room,
        )
        cache["meta"] = {
            "n_rir": meta["n_rir"],
            "snr_range": [20.0, 20.0],
            "seed": int(args.fixed1024_seed),
            "rir_pathlist": str(fixed_list_path),
            "src_pathlist": test_speech,
            "noise_pathlist": test_noise,
        }
        torch.save(cache, cache_path.as_posix())
        print(f"wrote cache {cache_path}")

    results: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt_epoch,
        "config": str(config_path),
        "meta_seed": meta_seed,
        "num_room_params": int(model_args.get("num_room_params", 7)),
        "volume_mode": dl_args.get("volume_mode"),
        "rad_resolution_m": rad_resolution,
        "split": "test-subset-fixed1024",
        "n_clips": int(cache["input_ft"].shape[0]),
        "snr_db": 20.0,
        "seed": int(args.fixed1024_seed),
        "permute_seed": int(args.permute_seed),
        "noise_seed": int(args.noise_seed),
        "rir_pathlist": str(fixed_list_path),
        "src_pathlist": test_speech,
        "noise_pathlist": test_noise,
        "batchsize": args.batchsize,
        "device": str(device),
        "input_ablations": {},
    }

    for cond in args.conditions:
        results["input_ablations"][cond] = _eval_cached(
            model,
            cache,
            device,
            condition=cond,
            rad_resolution=rad_resolution,
            batchsize=args.batchsize,
            permute_seed=int(args.permute_seed),
            noise_seed=int(args.noise_seed),
            report_mirror=False,
        )

    # ---- optional audio-only control ----
    if args.audio_only_checkpoint:
        ao_cfg_path = (
            Path(args.audio_only_config).expanduser().absolute()
            if args.audio_only_config
            else config_path
        )
        ao_cfg = toml.load(ao_cfg_path.as_posix())
        # Force no FiLM even if config still has use_film=true
        ao_cfg["model"]["args"]["use_film"] = False
        ao_cfg["model"]["args"]["use_film_ctf_layers"] = False
        ao_model, ao_epoch = _load_model(
            ao_cfg, Path(args.audio_only_checkpoint).expanduser().absolute(), device
        )
        ao = _eval_cached(
            ao_model,
            cache,
            device,
            condition="normal",  # audio+no-room (room ignored when use_film false)
            rad_resolution=rad_resolution,
            batchsize=args.batchsize,
            permute_seed=int(args.permute_seed),
            noise_seed=int(args.noise_seed),
            report_mirror=True,
        )
        # Re-run with room_params stripped to be explicit
        cache_no_room = {k: v for k, v in cache.items() if k != "room_params"}
        ao2 = _eval_cached(
            ao_model,
            cache_no_room,
            device,
            condition="normal",
            rad_resolution=rad_resolution,
            batchsize=args.batchsize,
            permute_seed=int(args.permute_seed),
            noise_seed=int(args.noise_seed),
            report_mirror=True,
        )
        results["audio_only"] = {
            **ao2,
            "condition": "audio_only_network",
            "checkpoint": str(Path(args.audio_only_checkpoint).expanduser().absolute()),
            "checkpoint_epoch": ao_epoch,
            "config": str(ao_cfg_path),
            "with_room_cache_row": ao,  # sanity; should match ao2 if model ignores room
        }
        print(
            f"  audio_only        ang {ao2['angle_mae_deg']:.2f}°  "
            f"rad {ao2['distance_mae_m']:.3f}m  "
            f"mirror {ao2['mirror_folded_angle_mae_deg']:.2f}°"
        )

    # ---- optional metadata-only MLP ----
    if args.metadata_only_checkpoint:
        mo_path = Path(args.metadata_only_checkpoint).expanduser().absolute()
        blob = torch.load(mo_path.as_posix(), map_location="cpu")
        # Prefer a full module; else rebuild a simple MLP from state_dict shapes.
        if hasattr(blob, "forward"):
            mlp = blob.to(device).eval()
            mo_epoch = None
        elif isinstance(blob, dict) and "model" in blob and hasattr(blob["model"], "forward"):
            mlp = blob["model"].to(device).eval()
            mo_epoch = blob.get("epoch")
        else:
            # Reconstruct MetadataOnlyMLP from state_dict
            state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
            if isinstance(state, dict) and any(k.startswith("module.") for k in state):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            in_dim = int(state["net.0.weight"].shape[1])
            n_ang = int(state["angle_head.weight"].shape[0])
            n_rad = int(state["radius_head.weight"].shape[0])
            hidden = int(state["net.0.weight"].shape[0])

            class _MetadataOnlyMLP(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(in_dim, hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden, hidden),
                        torch.nn.ReLU(),
                    )
                    self.angle_head = torch.nn.Linear(hidden, n_ang)
                    self.radius_head = torch.nn.Linear(hidden, n_rad)

                def forward(self, room_params):
                    h = self.net(room_params)
                    return self.angle_head(h), self.radius_head(h)

            mlp = _MetadataOnlyMLP().to(device)
            mlp.load_state_dict(state, strict=True)
            mlp.eval()
            mo_epoch = blob.get("epoch") if isinstance(blob, dict) else None
        mo = _eval_metadata_only_mlp(
            mlp, cache, device, rad_resolution=rad_resolution, batchsize=args.batchsize
        )
        results["metadata_only"] = {
            **mo,
            "checkpoint": str(mo_path),
            "checkpoint_epoch": mo_epoch,
        }
        print(
            f"  metadata_only     ang {mo['angle_mae_deg']:.2f}°  "
            f"rad {mo['distance_mae_m']:.3f}m"
        )

    summary_path = out_dir / "a3_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {summary_path}")

    lines = [
        "A3 audio / metadata ablations (input rows)",
        f"checkpoint: {ckpt_path} (epoch={ckpt_epoch})",
        f"config: {config_path}",
        f"fixed1024 @20dB  n={results['n_clips']}  seed={results['seed']}  "
        f"permute_seed={results['permute_seed']}",
        "",
        f"{'condition':16s}  {'ang MAE':>8s}  {'rad MAE':>8s}",
        "-" * 40,
    ]
    for cond, row in results["input_ablations"].items():
        lines.append(
            f"{cond:16s}  {row['angle_mae_deg']:7.2f}°  {row['distance_mae_m']:7.3f}m"
        )
    if "audio_only" in results:
        ao = results["audio_only"]
        lines.append(
            f"{'audio_only':16s}  {ao['angle_mae_deg']:7.2f}°  {ao['distance_mae_m']:7.3f}m  "
            f"mirror={ao['mirror_folded_angle_mae_deg']:.2f}°"
        )
    if "metadata_only" in results:
        mo = results["metadata_only"]
        lines.append(
            f"{'metadata_only':16s}  {mo['angle_mae_deg']:7.2f}°  {mo['distance_mae_m']:7.3f}m"
        )
    (out_dir / "a3_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
