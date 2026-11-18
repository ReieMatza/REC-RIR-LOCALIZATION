#!/usr/bin/env python3
"""D1 (STRL-25): frozen-trunk localization-head bottleneck checks.

Freeze encoder / speech / noise / CTF (+ their FiLM) / decoders from the locked
ep40 checkpoint. Train only the localization readout
(embedding_1dconv, angle_fc*, radius_head, readout FiLM) for a short shared
budget on cached trunk embeddings, then compare variants to the control.

Variants
  0 control     — unchanged architecture + same extra training
  1 norm        — Embedding1DConv BatchNorm1d → GroupNorm(8)
  2 dist_mlp    — radius_head: Linear → MLP 96→256→classes + ReLU
  3 film_wide   — readout FiLM hidden 14 (7×2) → 64
  4 soft_decode — eval-only: weighted mean over argmax ± 2 bins (locked ckpt)

Also reports B1 (dEchorate A0/A1, no fine-tune) for the norm variant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.RecRIR import FiLMConditioner  # noqa: E402
from model.lossF import GaussianLabelSmoothingLoss  # noqa: E402
from trainer_inferencer.utils import initialize_module  # noqa: E402

import warnings

warnings.filterwarnings("ignore")

READOUT_KEYS = (
    "embedding_1dconv",
    "film_embedding_conv",
    "film_conditioner",
    "film_angle",
    "film_radius",
    "angle_fc",
    "angle_act",
    "angle_drop",
    "radius_head",
)

VARIANTS = ("control", "norm", "dist_mlp", "film_wide")


def _is_readout(name: str) -> bool:
    return any(k in name for k in READOUT_KEYS)


def _load_config(path: Path) -> dict:
    return toml.load(path.as_posix())


def _load_base_model(config: dict, ckpt_path: Path, device: torch.device):
    model = initialize_module(config["model"]["path"], args=config["model"]["args"])
    ckpt = torch.load(ckpt_path.as_posix(), map_location="cpu")
    epoch = ckpt.get("epoch", None)
    state = {
        k.replace("module.", ""): v
        for k, v in ckpt["model"].items()
        if not any(x in k for x in ["ops", "params"])
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing {len(missing)} keys e.g. {missing[:3]}")
    if unexpected:
        print(f"[warn] unexpected {len(unexpected)} keys e.g. {unexpected[:3]}")
    model.to(device).eval()
    return model, epoch, state


def _freeze_trunk(model: nn.Module) -> Tuple[List[nn.Parameter], int, int]:
    trainable: List[nn.Parameter] = []
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        if _is_readout(name):
            p.requires_grad = True
            trainable.append(p)
            n_train += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    return trainable, n_train, n_frozen


def _set_readout_train_mode(model: nn.Module) -> None:
    model.eval()
    for name, module in model.named_modules():
        if name and _is_readout(name):
            module.train()


def loc_forward(
    model: nn.Module, x_CTF: torch.Tensor, room_params: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Localization readout only (mirrors BiSpatialNet after return_embedding)."""
    if model.embedding_type == "1dconv":
        ctf_embedding = model.embedding_1dconv(x_CTF, room_params=room_params)
    elif model.embedding_type == "variance":
        ctf_mean = x_CTF.mean(dim=1).squeeze(1)
        ctf_std = x_CTF.std(dim=1).squeeze(1)
        ctf_embedding = torch.cat([ctf_mean, ctf_std], dim=-1).squeeze(1)
    else:
        ctf_embedding = x_CTF.mean(dim=1).squeeze(1)

    if model.use_film and model.film_conditioner is not None and room_params is not None:
        gamma, beta = model.film_conditioner(room_params)
        ctf_embedding = gamma * ctf_embedding + beta

    angle_x = model.angle_fc1(ctf_embedding)
    angle_x = model.angle_act1(angle_x)
    if model.use_film and model.film_angle_1 is not None and room_params is not None:
        gamma, beta = model.film_angle_1(room_params)
        angle_x = gamma * angle_x + beta
    angle_x = model.angle_drop1(angle_x)
    angle_x = model.angle_fc2(angle_x)
    angle_x = model.angle_act2(angle_x)
    if model.use_film and model.film_angle_2 is not None and room_params is not None:
        gamma, beta = model.film_angle_2(room_params)
        angle_x = gamma * angle_x + beta
    angle_x = model.angle_drop2(angle_x)
    angle_logits = model.angle_fc3(angle_x)

    radius_in = ctf_embedding
    if model.use_film and model.film_radius is not None and room_params is not None:
        gamma, beta = model.film_radius(room_params)
        radius_in = gamma * radius_in + beta
    radius_logits = model.radius_head(radius_in)
    return angle_logits, radius_logits


def apply_variant(model: nn.Module, variant: str, *, num_room_params: int, dim_hidden: int, num_rad: int) -> None:
    """In-place architecture tweak for a D1 variant (after loading locked weights)."""
    if variant == "control":
        return
    if variant == "norm":
        # Only BatchNorm in the model lives here; GroupNorm avoids train/eval BN mismatch.
        old = model.embedding_1dconv.bn1
        gn = nn.GroupNorm(num_groups=8, num_channels=dim_hidden)
        with torch.no_grad():
            if hasattr(old, "weight") and old.weight is not None:
                gn.weight.copy_(old.weight)
                gn.bias.copy_(old.bias)
        model.embedding_1dconv.bn1 = gn.to(next(model.parameters()).device)
        return
    if variant == "dist_mlp":
        device = next(model.parameters()).device
        model.radius_head = nn.Sequential(
            nn.Linear(dim_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, num_rad),
        ).to(device)
        return
    if variant == "film_wide":
        device = next(model.parameters()).device
        # hidden layer 14 → 64 for readout FiLM conditioners only
        def _wide(dim: int) -> FiLMConditioner:
            # FiLMConditioner uses hidden = num_room_params * hidden_mult.
            # Choose hidden_mult so hidden == 64 (ceil).
            # 7 * 9 = 63 ≈ 64; build custom Sequential instead for exact 64.
            cond = FiLMConditioner(num_room_params, dim, hidden_mult=2, identity_init=False)
            last = nn.Linear(64, dim * 2)
            cond.mlp = nn.Sequential(
                nn.Linear(num_room_params, 64),
                nn.ReLU(),
                last,
            ).to(device)
            return cond.to(device)

        model.film_conditioner = _wide(dim_hidden)
        model.film_angle_1 = _wide(512)
        model.film_angle_2 = _wide(256)
        model.film_radius = _wide(dim_hidden)
        return
    raise ValueError(f"unknown variant {variant}")


def _expand_dataset_to_all_rirs(dataset) -> int:
    n_rir = len(dataset.rir_pathlist)
    speech = list(dataset.source_pathlist)
    reps = (n_rir + len(speech) - 1) // len(speech)
    dataset.source_pathlist = (speech * reps)[:n_rir]
    dataset.length = n_rir
    return n_rir


def _build_loader(
    config: dict,
    *,
    snr_range: List[float],
    rir_pathlist: str,
    src_pathlist: str,
    noise_pathlist: str,
    batchsize: int,
    seed: int,
    seq_len: float,
    num_workers: int,
    shard_index: int = 0,
    shard_count: int = 1,
    max_samples: Optional[int] = None,
):
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
    # Manual shard by index range for multi-GPU cache extraction.
    # Optional cap (deterministic prefix) keeps D1 variants comparable on a
    # short frozen-trunk budget without a multi-hour full-corpus cache.
    n_use = n if max_samples is None else min(n, int(max_samples))
    indices = list(range(shard_index, n_use, shard_count))
    sampler = MyDistributedSampler(
        ds, seed=int(seed), shuffle=False, rank=0, num_replicas=1, drop_last=False
    )
    # Wrap: only yield sharded indices while preserving (idx, seed) protocol.
    class _Shard:
        def __iter__(self):
            # MyDistributedSampler yields (idx, seed) pairs
            pairs = list(sampler)
            for i in indices:
                yield pairs[i]

        def __len__(self):
            return len(indices)

    loader = DataLoader(
        ds,
        sampler=_Shard(),
        batch_size=batchsize,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )
    meta = {
        "n_rir": n,
        "n_use": n_use,
        "max_samples": max_samples,
        "n_shard": len(indices),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "snr_range": list(snr_range),
        "seed": int(seed),
        "seq_len": float(seq_len),
        "rir_pathlist": rir_pathlist,
        "src_pathlist": src_pathlist,
        "noise_pathlist": noise_pathlist,
        "batchsize": batchsize,
        "indices": indices,
    }
    return loader, meta


@torch.no_grad()
def extract_cache_shard(
    model: nn.Module,
    transformfunc,
    loader,
    device: torch.device,
    *,
    n_expected: int,
    desc: str,
    store_locked_logits: bool = True,
) -> Dict[str, torch.Tensor]:
    xs: List[torch.Tensor] = []
    angs: List[torch.Tensor] = []
    rads: List[torch.Tensor] = []
    rps: List[torch.Tensor] = []
    a_logits: List[torch.Tensor] = []
    r_logits: List[torch.Tensor] = []
    use_amp = device.type == "cuda"
    for batch in tqdm(loader, desc=desc, total=(n_expected + loader.batch_size - 1) // loader.batch_size):
        noisy_wav = batch[0].to(device, non_blocking=True)
        angle_class = batch[5]
        radius_class = batch[6]
        room_params = batch[7].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            input_ft = transformfunc.preprocess(transformfunc.stft(noisy_wav, output_type="complex"))
            x_CTF = model(input_ft, room_params=room_params, return_embedding=True)
            if store_locked_logits:
                ang_log, rad_log = loc_forward(model, x_CTF.float(), room_params)
        xs.append(x_CTF.float().half().cpu())
        angs.append(angle_class.cpu())
        rads.append(radius_class.cpu())
        rps.append(room_params.float().cpu())
        if store_locked_logits:
            a_logits.append(ang_log.float().cpu())
            r_logits.append(rad_log.float().cpu())
    out = {
        "x_CTF": torch.cat(xs, dim=0),
        "angle_class": torch.cat(angs, dim=0),
        "radius_class": torch.cat(rads, dim=0),
        "room_params": torch.cat(rps, dim=0),
    }
    if store_locked_logits:
        out["locked_angle_logits"] = torch.cat(a_logits, dim=0)
        out["locked_radius_logits"] = torch.cat(r_logits, dim=0)
    return out


def merge_shards(shard_paths: Sequence[Path], indices_list: Sequence[List[int]], out_path: Path) -> Dict[str, Any]:
    shards = [torch.load(p, map_location="cpu") for p in shard_paths]
    n = max(max(ix) for ix in indices_list) + 1
    keys = [k for k in shards[0].keys() if torch.is_tensor(shards[0][k])]
    shapes = {k: list(shards[0][k].shape) for k in keys}
    for k in keys:
        shapes[k][0] = n
    merged = {k: torch.empty(shapes[k], dtype=shards[0][k].dtype) for k in keys}
    for shard, indices in zip(shards, indices_list):
        for j, idx in enumerate(indices):
            for k in keys:
                merged[k][idx] = shard[k][j]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)
    return {"n": n, "path": str(out_path), "bytes": out_path.stat().st_size, "keys": keys}


def metrics_from_logits(
    angle_logits: torch.Tensor,
    radius_logits: torch.Tensor,
    angle_class: torch.Tensor,
    radius_class: torch.Tensor,
    rad_resolution: float,
    *,
    soft: bool = False,
    soft_window: int = 2,
) -> Dict[str, Any]:
    if soft:
        angle_preds = soft_class_expectation(angle_logits, soft_window)
        radius_preds = soft_class_expectation(radius_logits, soft_window)
        # continuous class expectations → compare to integer labels in class units
        angle_err = (angle_preds - angle_class.float()).abs()
        radius_err = (radius_preds - radius_class.float()).abs()
    else:
        angle_preds = angle_logits.argmax(dim=1)
        radius_preds = radius_logits.argmax(dim=1)
        angle_err = (angle_preds - angle_class).abs().float()
        radius_err = (radius_preds - radius_class).abs().float()

    valid_a = angle_class >= 0
    valid_r = radius_class >= 0
    angle_mae = float(angle_err[valid_a].mean()) if valid_a.any() else float("nan")
    dist_mae = float(radius_err[valid_r].mean() * rad_resolution) if valid_r.any() else float("nan")
    out = {
        "n": int(angle_class.numel()),
        "angle_mae_deg": angle_mae,
        "distance_mae_m": dist_mae,
        "soft": soft,
        "soft_window": soft_window if soft else None,
    }
    if soft:
        # also provide argmax baseline on same logits for paired Δ
        a_arg = angle_logits.argmax(dim=1)
        r_arg = radius_logits.argmax(dim=1)
        out["argmax_angle_mae_deg"] = float((a_arg - angle_class).abs()[valid_a].float().mean())
        out["argmax_distance_mae_m"] = float(
            (r_arg - radius_class).abs()[valid_r].float().mean() * rad_resolution
        )
    return out


def soft_class_expectation(logits: torch.Tensor, window: int) -> torch.Tensor:
    """Probability-weighted mean over argmax ± window bins (class-index units)."""
    B, C = logits.shape
    k = logits.argmax(dim=1)
    idx = torch.arange(C, device=logits.device).unsqueeze(0).expand(B, -1)
    lo = (k - window).unsqueeze(1)
    hi = (k + window).unsqueeze(1)
    mask = (idx >= lo) & (idx <= hi)
    masked = logits.masked_fill(~mask, -1e9)
    probs = F.softmax(masked, dim=1)
    return (probs * idx.float()).sum(dim=1)


def distance_bin_report(
    radius_logits: torch.Tensor,
    radius_class: torch.Tensor,
    rad_resolution: float,
    *,
    soft: bool,
    soft_window: int = 2,
    n_bins: int = 5,
) -> List[Dict[str, Any]]:
    true_m = radius_class.float() * rad_resolution
    if soft:
        pred = soft_class_expectation(radius_logits, soft_window) * rad_resolution
        pred_arg = radius_logits.argmax(dim=1).float() * rad_resolution
    else:
        pred = radius_logits.argmax(dim=1).float() * rad_resolution
        pred_arg = pred
    qs = np.quantile(true_m.numpy(), np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = float(qs[i]), float(qs[i + 1])
        if i < n_bins - 1:
            m = (true_m >= lo) & (true_m < hi)
        else:
            m = (true_m >= lo) & (true_m <= hi)
        if not m.any():
            continue
        err = (pred[m] - true_m[m]).abs()
        err_arg = (pred_arg[m] - true_m[m]).abs()
        rows.append(
            {
                "bin": i,
                "true_dist_lo_m": lo,
                "true_dist_hi_m": hi,
                "n": int(m.sum()),
                "distance_mae_m": float(err.mean()),
                "argmax_distance_mae_m": float(err_arg.mean()),
            }
        )
    return rows


def train_variant(
    model: nn.Module,
    train_cache: Dict[str, torch.Tensor],
    val_cache: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    rad_resolution: float,
    angle_sigma: float,
    radius_sigma: float,
    num_angles: int,
    num_rad: int,
) -> Dict[str, Any]:
    trainable, n_train, n_frozen = _freeze_trunk(model)
    opt = torch.optim.AdamW(trainable, lr=lr)
    angle_loss = GaussianLabelSmoothingLoss(num_angles, sigma=angle_sigma).to(device)
    radius_loss = GaussianLabelSmoothingLoss(num_rad, sigma=radius_sigma).to(device)

    x = train_cache["x_CTF"]
    ang = train_cache["angle_class"]
    rad = train_cache["radius_class"]
    rp = train_cache["room_params"]
    ds = TensorDataset(x, ang, rad, rp)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    history = []
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    for ep in range(1, epochs + 1):
        _set_readout_train_mode(model)
        t0 = time.time()
        loss_sum = 0.0
        n_seen = 0
        for xb, ab, rb, rpb in loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            ab = ab.to(device)
            rb = rb.to(device)
            rpb = rpb.to(device=device, dtype=torch.float32, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            a_log, r_log = loc_forward(model, xb, rpb)
            loss = angle_loss(a_log, ab) + radius_loss(r_log, rb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            bs = xb.shape[0]
            loss_sum += float(loss.item()) * bs
            n_seen += bs
        model.eval()
        with torch.no_grad():
            val_m = eval_cache(model, val_cache, device, rad_resolution)
        score = val_m["angle_mae_deg"] + 10.0 * val_m["distance_mae_m"]
        row = {
            "epoch": ep,
            "train_loss": loss_sum / max(n_seen, 1),
            "val_angle_mae_deg": val_m["angle_mae_deg"],
            "val_distance_mae_m": val_m["distance_mae_m"],
            "val_score": score,
            "sec": time.time() - t0,
        }
        history.append(row)
        if score < best_score:
            best_score = score
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"  ep{ep}: loss={row['train_loss']:.4f}  "
            f"val={row['val_angle_mae_deg']:.3f}°/{row['val_distance_mae_m']:.3f}m  "
            f"({row['sec']:.1f}s)",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return {
        "n_trainable_params": n_train,
        "n_frozen_params": n_frozen,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
    }


@torch.no_grad()
def eval_cache(
    model: nn.Module,
    cache: Dict[str, torch.Tensor],
    device: torch.device,
    rad_resolution: float,
    *,
    batch_size: int = 512,
) -> Dict[str, Any]:
    model.eval()
    n = cache["x_CTF"].shape[0]
    a_logs, r_logs = [], []
    for i in range(0, n, batch_size):
        xb = cache["x_CTF"][i : i + batch_size].to(device=device, dtype=torch.float32)
        rpb = cache["room_params"][i : i + batch_size].to(device=device, dtype=torch.float32)
        a_log, r_log = loc_forward(model, xb, rpb)
        a_logs.append(a_log.cpu())
        r_logs.append(r_log.cpu())
    return metrics_from_logits(
        torch.cat(a_logs),
        torch.cat(r_logs),
        cache["angle_class"],
        cache["radius_class"],
        rad_resolution,
        soft=False,
    )


def save_readout_state(model: nn.Module, path: Path) -> None:
    state = {k: v.cpu() for k, v in model.state_dict().items() if _is_readout(k)}
    # Also keep non-readout keys that variants may have replaced under readout names;
    # for GroupNorm / MLP / wide FiLM the new modules are covered by READOUT_KEYS.
    # radius_head.* and embedding_1dconv.bn1.* match.
    # Include any Sequential radius_head params.
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_readout_into_model(model: nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu")
    cur = model.state_dict()
    cur.update(state)
    model.load_state_dict(cur, strict=False)


def stage_cache(args, config) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transformfunc = initialize_module(config["acoustic"]["path"], config["acoustic"]["args"])
    model, epoch, _ = _load_base_model(config, Path(args.checkpoint), device)
    print(f"cache: locked epoch={epoch} device={device} shard={args.shard_index}/{args.shard_count}")

    dl = config["dataloader"]["args"]
    cache_dir = Path(args.out_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "train": dict(
            snr_range=list(dl["snr_range"]),
            rir_pathlist=dl["rir_pathlist"][0],
            src_pathlist=dl["src_pathlist"][0],
            noise_pathlist=dl["noise_pathlist"][0],
            seq_len=float(dl["seq_lenlist"][0]),
            seed=int(dl["seeds"][0]),
        ),
        "val": dict(
            snr_range=list(dl["snr_range"]),
            rir_pathlist=dl["rir_pathlist"][1],
            src_pathlist=dl["src_pathlist"][1],
            noise_pathlist=dl["noise_pathlist"][1],
            seq_len=float(dl["seq_lenlist"][1]),
            seed=int(dl["seeds"][1]),
        ),
        "fixed1024": dict(
            snr_range=[20.0, 20.0],
            rir_pathlist=str(
                _REPO_ROOT
                / "config/rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w09-miczfree-s2033/reie-test-rir-fixed1024.txt"
            ),
            src_pathlist=dl["src_pathlist"][2],
            noise_pathlist=dl["noise_pathlist"][2],
            seq_len=float(dl["seq_lenlist"][2]),
            seed=2024,
        ),
    }
    for name, spec in specs.items():
        if args.split not in ("all", name):
            continue
        max_samples = None
        if name == "train" and args.max_train_samples is not None:
            max_samples = int(args.max_train_samples)
        loader, meta = _build_loader(
            config,
            batchsize=args.batchsize,
            num_workers=args.num_workers,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            max_samples=max_samples,
            **spec,
        )
        shard = extract_cache_shard(
            model,
            transformfunc,
            loader,
            device,
            n_expected=meta["n_shard"],
            desc=f"cache-{name}-s{args.shard_index}",
            # Train cache is only for readout training; skip locked logits I/O.
            store_locked_logits=(name != "train"),
        )
        shard_path = cache_dir / f"{name}.shard{args.shard_index}.pt"
        meta_path = cache_dir / f"{name}.shard{args.shard_index}.meta.json"
        torch.save(shard, shard_path)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"wrote {shard_path} n={shard['x_CTF'].shape[0]}", flush=True)


def stage_merge(args) -> None:
    cache_dir = Path(args.out_dir) / "cache"
    for name in ("train", "val", "fixed1024"):
        if args.split not in ("all", name):
            continue
        metas = []
        paths = []
        for s in range(args.shard_count):
            p = cache_dir / f"{name}.shard{s}.pt"
            m = json.loads((cache_dir / f"{name}.shard{s}.meta.json").read_text())
            paths.append(p)
            metas.append(m["indices"])
        out = cache_dir / f"{name}.pt"
        info = merge_shards(paths, metas, out)
        print(f"merged {name}: {info}", flush=True)
        blob = torch.load(out, map_location="cpu")
        if "locked_angle_logits" not in blob:
            continue
        rad_res = 0.25
        locked = metrics_from_logits(
            blob["locked_angle_logits"],
            blob["locked_radius_logits"],
            blob["angle_class"],
            blob["radius_class"],
            rad_res,
            soft=False,
        )
        soft = metrics_from_logits(
            blob["locked_angle_logits"],
            blob["locked_radius_logits"],
            blob["angle_class"],
            blob["radius_class"],
            rad_res,
            soft=True,
            soft_window=2,
        )
        bins = distance_bin_report(
            blob["locked_radius_logits"],
            blob["radius_class"],
            rad_res,
            soft=True,
            soft_window=2,
        )
        summary = {"locked_argmax": locked, "locked_soft_w2": soft, "soft_distance_bins": bins}
        (cache_dir / f"{name}.locked_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  locked argmax {locked['angle_mae_deg']:.3f}° / {locked['distance_mae_m']:.3f}m", flush=True)
        print(f"  locked soft   {soft['angle_mae_deg']:.3f}° / {soft['distance_mae_m']:.3f}m", flush=True)


def stage_train(args, config) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.out_dir) / "cache"
    train_cache = torch.load(cache_dir / "train.pt", map_location="cpu")
    val_cache = torch.load(cache_dir / "val.pt", map_location="cpu")
    fixed_cache = torch.load(cache_dir / "fixed1024.pt", map_location="cpu")

    ma = config["model"]["args"]
    dl = config["dataloader"]["args"]
    rad_res = float(ma.get("rad_resolution", 0.25))
    num_angles = int(ma.get("num_angles", 181))
    max_rad = float(ma.get("max_rad_value", 13.0))
    num_rad = int(max_rad / rad_res) + 1
    dim_hidden = int(ma["dim_hidden"])
    num_room = int(ma.get("num_room_params", 7))
    angle_sigma = float(config["trainer"]["train"].get("angle_label_smoothing_sigma", 10.0))
    radius_sigma = float(config["trainer"]["train"].get("radius_label_smoothing_sigma", 2.0))
    lr = float(config["optimizer"]["args"]["lr"])

    variants = VARIANTS if args.variant == "all" else (args.variant,)
    results: Dict[str, Any] = {
        "epochs": args.epochs,
        "lr": lr,
        "train_batch_size": args.train_batch_size,
        "protocol": "frozen_trunk_cached_xCTF_train_readout",
    }

    for variant in variants:
        print(f"\n===== TRAIN variant={variant} =====", flush=True)
        model, epoch, _ = _load_base_model(config, Path(args.checkpoint), device)
        apply_variant(
            model,
            variant,
            num_room_params=num_room,
            dim_hidden=dim_hidden,
            num_rad=num_rad,
        )
        train_info = train_variant(
            model,
            train_cache,
            val_cache,
            device=device,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            lr=lr,
            rad_resolution=rad_res,
            angle_sigma=angle_sigma,
            radius_sigma=radius_sigma,
            num_angles=num_angles,
            num_rad=num_rad,
        )
        model.eval()
        val_m = eval_cache(model, val_cache, device, rad_res)
        fix_m = eval_cache(model, fixed_cache, device, rad_res)
        ckpt_out = Path(args.out_dir) / "ckpts" / f"{variant}.pt"
        # Save full state_dict so GroupNorm / MLP / wide FiLM reload cleanly
        torch.save(
            {
                "variant": variant,
                "base_epoch": epoch,
                "model": model.state_dict(),
                "train_info": train_info,
                "val": val_m,
                "fixed1024": fix_m,
            },
            ckpt_out,
        )
        results[variant] = {
            "train": train_info,
            "val": val_m,
            "fixed1024": fix_m,
            "ckpt": str(ckpt_out),
        }
        print(
            f"  DONE {variant}: val {val_m['angle_mae_deg']:.3f}°/{val_m['distance_mae_m']:.3f}m  "
            f"fixed1024 {fix_m['angle_mae_deg']:.3f}°/{fix_m['distance_mae_m']:.3f}m",
            flush=True,
        )

    out = Path(args.out_dir) / "d1_train_results.json"
    # locked baselines from cache metrics
    locked = {}
    for name in ("val", "fixed1024"):
        p = cache_dir / f"{name}.locked_metrics.json"
        if p.exists():
            locked[name] = json.loads(p.read_text())
    results["locked_no_extra_train"] = locked
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


def stage_soft(args) -> None:
    cache_dir = Path(args.out_dir) / "cache"
    out = {"soft_window": 2, "splits": {}}
    for name in ("val", "fixed1024"):
        blob = torch.load(cache_dir / f"{name}.pt", map_location="cpu")
        rad_res = 0.25
        argmax = metrics_from_logits(
            blob["locked_angle_logits"],
            blob["locked_radius_logits"],
            blob["angle_class"],
            blob["radius_class"],
            rad_res,
            soft=False,
        )
        soft = metrics_from_logits(
            blob["locked_angle_logits"],
            blob["locked_radius_logits"],
            blob["angle_class"],
            blob["radius_class"],
            rad_res,
            soft=True,
            soft_window=2,
        )
        bins = distance_bin_report(
            blob["locked_radius_logits"],
            blob["radius_class"],
            rad_res,
            soft=True,
            soft_window=2,
        )
        out["splits"][name] = {
            "argmax": argmax,
            "soft_w2": soft,
            "delta_soft_minus_argmax": {
                "angle_mae_deg": soft["angle_mae_deg"] - soft["argmax_angle_mae_deg"],
                "distance_mae_m": soft["distance_mae_m"] - soft["argmax_distance_mae_m"],
            },
            "distance_bins": bins,
        }
        print(
            f"[soft/{name}] argmax {argmax['angle_mae_deg']:.3f}°/{argmax['distance_mae_m']:.3f}m  "
            f"soft {soft['angle_mae_deg']:.3f}°/{soft['distance_mae_m']:.3f}m  "
            f"Δdist={soft['distance_mae_m']-soft['argmax_distance_mae_m']:+.4f}m",
            flush=True,
        )
    path = Path(args.out_dir) / "d1_soft_decode.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def stage_b1_norm(args, config) -> None:
    """Zero-shot dEchorate B1 for the trained norm variant (sim-to-real concern)."""
    import csv

    import model.RecRIR as rec_rir
    from scripts.eval_but import run as but_run

    ckpt_blob = torch.load(Path(args.out_dir) / "ckpts" / "norm.pt", map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dim_hidden = int(config["model"]["args"]["dim_hidden"])

    # eval_but builds a stock BiSpatialNet (BatchNorm). Patch Embedding1DConv so
    # freshly constructed models use GroupNorm before the norm-variant state loads.
    _orig_init = rec_rir.Embedding1DConv.__init__

    def _gn_init(self, dim_hidden, kernel_size=5, film_post_conv1=None, film_post_conv2=None):
        _orig_init(
            self,
            dim_hidden,
            kernel_size=kernel_size,
            film_post_conv1=film_post_conv1,
            film_post_conv2=film_post_conv2,
        )
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=dim_hidden)

    rec_rir.Embedding1DConv.__init__ = _gn_init  # type: ignore[method-assign]

    model, base_epoch, _ = _load_base_model(config, Path(args.checkpoint), device)
    # base load put BN weights into a GroupNorm module (shape match on weight/bias);
    # overwrite with trained norm-variant state.
    missing, unexpected = model.load_state_dict(ckpt_blob["model"], strict=False)
    print(f"[b1] load norm state missing={len(missing)} unexpected={len(unexpected)}")
    full_ckpt = Path(args.out_dir) / "ckpts" / "norm_full.tar"
    torch.save({"epoch": base_epoch, "model": model.state_dict()}, full_ckpt)

    speech = _REPO_ROOT / "config/reie_validation_speech_quick.txt"
    noise = _REPO_ROOT / "config/reie_train_noise.txt"
    datasets = {
        "measured": Path("/storage/reie/data/dechorate-arr01"),
        "twin-exact": Path("/storage/reie/data/dechorate-arr01-twin/exact"),
    }
    results = {}
    try:
        for name, root in datasets.items():
            out_dir = Path(args.out_dir) / "b1_norm" / name
            print(f"\n===== B1 norm variant dataset={name} =====", flush=True)
            but_run(
                config_path=Path(args.config),
                checkpoint_path=full_ckpt,
                but_csv=None,
                dataset_root=root,
                speech_txt=speech,
                noise_txt=noise,
                clips_per_rir=20,
                seq_len=4.0,
                snr_range=(20.0, 20.0),
                noisy_proportion=1.0,
                out_dir=out_dir,
                device_str=args.device,
                no_room_conditioning=False,
                estimate_rt60=False,
                batch_size=8,
                num_workers=0,
                seed=42,
                filter_to_film_volume_range=True,
                film_mode="normal",
                volume_mode="drop",
            )
            pred_csv = out_dir / "per_entry_preds.csv"
            ang, rad = [], []
            with open(pred_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ang.append(float(row["angle_error_deg"]))
                    rad.append(float(row["radius_error_m"]))
            results[name] = {
                "n_clips": len(ang),
                "angle_mae_deg": float(np.mean(ang)) if ang else float("nan"),
                "radius_mae_m": float(np.mean(rad)) if rad else float("nan"),
            }
            print(json.dumps({name: results[name]}, indent=2), flush=True)
    finally:
        rec_rir.Embedding1DConv.__init__ = _orig_init  # type: ignore[method-assign]

    orig = Path("/storage/reie/experiments/b1-loudspeaker-type/b1_source_type_summary.json")
    payload = {
        "norm_variant_ckpt": str(full_ckpt),
        "results": results,
        "original_b1_summary": str(orig) if orig.exists() else None,
    }
    if orig.exists():
        payload["original_b1"] = json.loads(orig.read_text())
    out = Path(args.out_dir) / "d1_b1_norm.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="D1 frozen-trunk head bottleneck")
    p.add_argument(
        "--config",
        default="config/Rec-RIR-heads-only-free-mic-z-h100-miczfree-s2033-2gpu.toml",
    )
    p.add_argument(
        "--checkpoint",
        default="/storage/reie/experiments/free-mic-z-h100-miczfree-s2033/ckpt/best.tar",
    )
    p.add_argument(
        "--out-dir",
        default="/storage/reie/experiments/d1-frozen-trunk",
    )
    p.add_argument(
        "--stage",
        choices=("cache", "merge", "train", "soft", "b1", "all"),
        default="all",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batchsize", type=int, default=8, help="audio forward batch for cache")
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--split", choices=("all", "train", "val", "fixed1024"), default="all")
    p.add_argument("--variant", default="all", choices=("all",) + VARIANTS)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
        help="Deterministic train-RIR prefix for frozen-trunk readout training "
        "(None/-1 = full corpus). Default 10k keeps D1 wall-time practical on Ada.",
    )
    args = p.parse_args()
    if args.max_train_samples is not None and args.max_train_samples < 0:
        args.max_train_samples = None

    os.chdir(_REPO_ROOT)
    config = _load_config(Path(args.config).expanduser().absolute())
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.stage in ("cache", "all") and args.stage != "merge":
        if args.stage == "all" and args.shard_count > 1:
            raise SystemExit("stage=all expects shard_count=1; use cache/merge separately for multi-GPU")
        stage_cache(args, config)
    if args.stage == "merge":
        stage_merge(args)
    if args.stage in ("train", "all"):
        if args.stage == "all":
            # single-process cache already wrote shard0; promote to merged name if needed
            cache_dir = Path(args.out_dir) / "cache"
            for name in ("train", "val", "fixed1024"):
                merged = cache_dir / f"{name}.pt"
                shard = cache_dir / f"{name}.shard0.pt"
                if not merged.exists() and shard.exists():
                    args.shard_count = 1
                    # synthesize meta if missing
                    meta_p = cache_dir / f"{name}.shard0.meta.json"
                    if meta_p.exists():
                        stage_merge(args)
                    else:
                        torch.save(torch.load(shard, map_location="cpu"), merged)
        stage_train(args, config)
    if args.stage in ("soft", "all"):
        stage_soft(args)
    if args.stage in ("b1", "all"):
        stage_b1_norm(args, config)


if __name__ == "__main__":
    main()
