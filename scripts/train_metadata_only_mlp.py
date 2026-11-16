#!/usr/bin/env python3
"""Train a metadata-only MLP (no audio) for A3.

Input: 7-D room vector matching volume_mode=drop (RT60, L/W/H, mic x/y/z),
extracted from RIR npz files with the same normalisation as MyDataset.
Targets: angle / radius class indices from rir_localization.csv.

Architecture mirrors the Oct-2025 film-only control (256-wide backbone +
angle/radius heads), adapted to 7-D input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


RT60_MIN, RT60_MAX = 0.15, 1.5
ROOM_SZ_MAX = 15.0


def _normalize_room_params(npz: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    """7-D vector: RT60, room_sz(3), mic_center(3). Matches volume_mode=drop."""
    try:
        rt60 = float(np.asarray(npz["RT60"]).ravel()[0])
        room_sz = np.asarray(npz["room_sz"]).ravel()[:3].astype(np.float64)
        pos_rcv = np.asarray(npz["pos_rcv"])
        if pos_rcv.ndim == 1:
            pos_rcv = pos_rcv.reshape(-1, 3)
        mic = pos_rcv.mean(axis=0).astype(np.float64)
        rt60_n = np.clip((rt60 - RT60_MIN) / (RT60_MAX - RT60_MIN), 0.0, 1.0)
        room_n = np.clip(room_sz / ROOM_SZ_MAX, 0.0, 1.0)
        mic_n = np.clip(mic / (room_sz + 1e-8), 0.0, 1.0)
        return np.concatenate([[rt60_n], room_n, mic_n]).astype(np.float32)
    except (KeyError, ValueError, IndexError):
        return None


def _read_pathlist(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _load_loc_map(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["rir_path"]] = (float(row["angle_deg"]), float(row["radius_m"]))
    return out


class RoomParamDataset(Dataset):
    def __init__(
        self,
        rir_paths: List[str],
        loc_map: Dict[str, Tuple[float, float]],
        *,
        num_angles: int = 181,
        max_rad_value: float = 13.0,
        rad_resolution: float = 0.25,
        cache_path: Optional[Path] = None,
    ):
        self.num_angles = num_angles
        self.rad_resolution = rad_resolution
        self.max_rad_value = max_rad_value
        n_rad = int(max_rad_value / rad_resolution) + 1

        if cache_path is not None and cache_path.exists():
            blob = torch.load(cache_path.as_posix(), map_location="cpu")
            self.room = blob["room"]
            self.angle = blob["angle"]
            self.radius = blob["radius"]
            print(f"loaded cache {cache_path} n={len(self.angle)}")
            return

        rooms, angles, radii = [], [], []
        skipped = 0
        for p in tqdm(rir_paths, desc="extract"):
            if p not in loc_map:
                skipped += 1
                continue
            try:
                npz = np.load(p)
            except Exception:
                skipped += 1
                continue
            rp = _normalize_room_params(npz)
            if rp is None:
                skipped += 1
                continue
            ang, rad = loc_map[p]
            a_cls = int(round(ang))
            a_cls = max(0, min(num_angles - 1, a_cls))
            r_cls = int(round(rad / rad_resolution))
            r_cls = max(0, min(n_rad - 1, r_cls))
            rooms.append(rp)
            angles.append(a_cls)
            radii.append(r_cls)
        if not rooms:
            raise RuntimeError("no usable room_params extracted")
        self.room = torch.from_numpy(np.stack(rooms))
        self.angle = torch.tensor(angles, dtype=torch.long)
        self.radius = torch.tensor(radii, dtype=torch.long)
        print(f"extracted n={len(self.angle)} skipped={skipped}")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"room": self.room, "angle": self.angle, "radius": self.radius},
                cache_path.as_posix(),
            )
            print(f"wrote {cache_path}")

    def __len__(self):
        return int(self.angle.shape[0])

    def __getitem__(self, idx):
        return self.room[idx], self.angle[idx], self.radius[idx]


class MetadataOnlyMLP(nn.Module):
    """Same layout as Oct-2025 film_mlp, with configurable input dim (7 or 8)."""

    def __init__(self, in_dim: int = 7, num_angles: int = 181, num_radii: int = 53):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_angles),
        )
        self.radius_head = nn.Linear(256, num_radii)

    def forward(self, room_params: torch.Tensor):
        h = self.backbone(room_params)
        return self.angle_head(h), self.radius_head(h)


@torch.no_grad()
def _eval(model, loader, device, rad_resolution: float) -> Dict[str, float]:
    model.eval()
    ang_err = rad_err = n_a = n_r = 0.0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for room, ang, rad in loader:
        room = room.to(device)
        ang = ang.to(device)
        rad = rad.to(device)
        a_logits, r_logits = model(room)
        loss = ce(a_logits, ang) + ce(r_logits, rad)
        loss_sum += float(loss.item()) * room.shape[0]
        a_pred = a_logits.argmax(1)
        r_pred = r_logits.argmax(1)
        ang_err += torch.abs(a_pred - ang).float().sum().item()
        rad_err += torch.abs(r_pred - rad).float().sum().item()
        n_a += room.shape[0]
        n_r += room.shape[0]
    return {
        "loss": loss_sum / max(n_a, 1),
        "angle_mae_deg": ang_err / max(n_a, 1),
        "distance_mae_m": (rad_err / max(n_r, 1)) * rad_resolution,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-list",
        default=(
            "config/rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w09-miczfree-s2033/"
            "reie-train-rir.txt"
        ),
    )
    parser.add_argument(
        "--val-list",
        default=(
            "config/rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w09-miczfree-s2033/"
            "reie-validation-rir.txt"
        ),
    )
    parser.add_argument(
        "--localization-csv",
        default=(
            "/storage/reie/data/rec-rir/"
            "rir-uniform-angle-fullroom-edge-y0-band-min03-ow02-w09-miczfree-s2033/"
            "rir_localization.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="/storage/reie/experiments/a3-metadata-only-mlp-miczfree-s2033",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batchsize", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-angles", type=int, default=181)
    parser.add_argument("--max-rad-value", type=float, default=13.0)
    parser.add_argument("--rad-resolution", type=float, default=0.25)
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_list = Path(args.train_list)
    val_list = Path(args.val_list)
    if not train_list.is_absolute():
        train_list = _REPO_ROOT / train_list
    if not val_list.is_absolute():
        val_list = _REPO_ROOT / val_list

    loc_map = _load_loc_map(Path(args.localization_csv))
    train_ds = RoomParamDataset(
        _read_pathlist(train_list),
        loc_map,
        num_angles=args.num_angles,
        max_rad_value=args.max_rad_value,
        rad_resolution=args.rad_resolution,
        cache_path=out_dir / "cache_train.pt",
    )
    val_ds = RoomParamDataset(
        _read_pathlist(val_list),
        loc_map,
        num_angles=args.num_angles,
        max_rad_value=args.max_rad_value,
        rad_resolution=args.rad_resolution,
        cache_path=out_dir / "cache_val.pt",
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batchsize, shuffle=False, num_workers=0, drop_last=False
    )

    # Chance: train-mean constant on val labels
    train_ang = train_ds.angle.float()
    train_rad = train_ds.radius.float() * args.rad_resolution
    mean_ang = float(train_ang.mean())
    mean_rad = float(train_rad.mean())
    val_ang = val_ds.angle.float()
    val_rad = val_ds.radius.float() * args.rad_resolution
    chance_ang = float(torch.abs(val_ang - mean_ang).mean())
    chance_rad = float(torch.abs(val_rad - mean_rad).mean())
    print(
        f"train n={len(train_ds)} val n={len(val_ds)}  "
        f"chance(train-mean on val)={chance_ang:.2f}° / {chance_rad:.3f}m"
    )

    n_rad = int(args.max_rad_value / args.rad_resolution) + 1
    model = MetadataOnlyMLP(7, args.num_angles, n_rad).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    best = {"angle_mae_deg": 1e9, "distance_mae_m": 1e9, "epoch": -1, "loss": 1e9}
    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write(
            "epoch,lr,train_loss,train_angle_mae,train_distance_mae,"
            "val_loss,val_angle_mae,val_distance_mae\n"
        )

    for ep in range(1, args.epochs + 1):
        model.train()
        for room, ang, rad in train_loader:
            room = room.to(device)
            ang = ang.to(device)
            rad = rad.to(device)
            opt.zero_grad(set_to_none=True)
            a_logits, r_logits = model(room)
            loss = ce(a_logits, ang) + ce(r_logits, rad)
            loss.backward()
            opt.step()
        sched.step()
        tr = _eval(model, train_loader, device, args.rad_resolution)
        va = _eval(model, val_loader, device, args.rad_resolution)
        lr = float(opt.param_groups[0]["lr"])
        with open(metrics_path, "a") as f:
            f.write(
                f"{ep},{lr},{tr['loss']},{tr['angle_mae_deg']},{tr['distance_mae_m']},"
                f"{va['loss']},{va['angle_mae_deg']},{va['distance_mae_m']}\n"
            )
        star = ""
        # Prefer lower angle MAE; tie-break on distance
        if (va["angle_mae_deg"] < best["angle_mae_deg"] - 1e-6) or (
            abs(va["angle_mae_deg"] - best["angle_mae_deg"]) < 1e-6
            and va["distance_mae_m"] < best["distance_mae_m"]
        ):
            best = {
                "angle_mae_deg": va["angle_mae_deg"],
                "distance_mae_m": va["distance_mae_m"],
                "epoch": ep,
                "loss": va["loss"],
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "best": best,
                    "name": "metadata_only_mlp",
                    "in_dim": 7,
                    "num_angles": args.num_angles,
                    "num_radii": n_rad,
                    "seed": args.seed,
                },
                (out_dir / "best_metadata_mlp.pt").as_posix(),
            )
            star = " *"
        if ep == 1 or ep % 5 == 0 or star:
            print(
                f"ep {ep:3d}  val {va['angle_mae_deg']:.2f}°  "
                f"{va['distance_mae_m']:.3f}m{star}"
            )

    summary = {
        "best": best,
        "chance_train_mean_on_val": {
            "angle_mae_deg": chance_ang,
            "distance_mae_m": chance_rad,
            "train_mean_angle_deg": mean_ang,
            "train_mean_distance_m": mean_rad,
        },
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "in_dim": 7,
        "volume_mode": "drop",
        "seed": args.seed,
        "epochs": args.epochs,
        "device": str(device),
        "train_list": str(train_list),
        "val_list": str(val_list),
        "localization_csv": args.localization_csv,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "Metadata-only MLP (no audio) on miczfree-s2033",
        f"train n={len(train_ds)}  val n={len(val_ds)}  epochs={args.epochs}",
        f"best epoch {best['epoch']}: {best['angle_mae_deg']:.2f}° / {best['distance_mae_m']:.3f}m",
        f"chance (train-mean on val): {chance_ang:.2f}° / {chance_rad:.3f}m",
        "input = 7-D volume_mode=drop (RT60, L/W/H, mic x/y/z)",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out_dir / 'best_metadata_mlp.pt'}")


if __name__ == "__main__":
    main()
