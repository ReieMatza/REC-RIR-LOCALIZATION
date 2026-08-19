# @description: Render 2D top-down room layouts for RIR dataset samples with FiLM params.
#
# Loads RIR .npz files from either:
#   - a dataset config folder with reie-{train,validation,test}-rir.txt lists, or
#   - a raw generated dataset under data/rec-rir/... with train|validation|test/
# draws room outline, mic and source positions, and annotates FiLM room parameters.
#
# Usage (from REC-RIR-LOCALIZATION):
#   python visualize_dataset.py config/rir-uniform-angle-fullroom-edge -n 12 -o out/viz
#   python visualize_dataset.py config/rir-uniform-angle-fullroom-edge --split test -n 5 --seed 0
#   python visualize_dataset.py /storage/reie/data/rec-rir/rir-uniform-angle-fullroom-edge-y0-test -n 8

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# FiLM channel names (matches film_importance.py / dataset_RecRIR)
FILM_NAMES = [
    "RT60",
    "room_sz_x",
    "room_sz_y",
    "room_sz_z",
    "mic_x",
    "mic_y",
    "mic_z",
    "volume",
]

# Normalization constants (dataset_RecRIR.MyDataset)
_RT60_MIN, _RT60_MAX = 0.1, 1.5
_ROOM_SZ_MAX = 15.0
_LOG_VOL_MIN = float(np.log(10.0))
_LOG_VOL_MAX = float(np.log(1000.0))

SPLIT_TO_LIST = {
    "train": "reie-train-rir.txt",
    "validation": "reie-validation-rir.txt",
    "val": "reie-validation-rir.txt",
    "test": "reie-test-rir.txt",
}

SPLIT_TO_SUBDIR = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "test": "test",
}


def _read_pathlist(path: Path) -> List[str]:
    with path.open("r") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_localization_map(
    csv_path: Path,
    rir_col: str = "rir_path",
    angle_col: str = "angle_deg",
    radius_col: str = "radius_m",
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    with csv_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            out[Path(row[rir_col]).as_posix()] = (
                float(row[angle_col]),
                float(row[radius_col]),
            )
    return out


def _lookup_localization(
    loc_map: Dict[str, Tuple[float, float]], rir_path: str
) -> Optional[Tuple[float, float]]:
    key = Path(rir_path).as_posix()
    if key in loc_map:
        return loc_map[key]
    # CSV may use paths relative to a different prefix
    for k, v in loc_map.items():
        if k.endswith(Path(rir_path).name) or rir_path.endswith(Path(k).name):
            return v
    return None


def extract_room_params(rir_dict: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return (film_norm [8], film_raw dict) using dataset_RecRIR normalization."""
    RT60 = float(np.asarray(rir_dict["RT60"]).ravel()[0])
    room_sz = np.asarray(rir_dict["room_sz"]).ravel()[:3].astype(np.float64)
    pos_rcv = np.asarray(rir_dict["pos_rcv"])
    if pos_rcv.ndim == 1:
        pos_rcv = pos_rcv.reshape(-1, 3)
    mic_center = pos_rcv.mean(axis=0)

    rt60_norm = float(np.clip((RT60 - _RT60_MIN) / (_RT60_MAX - _RT60_MIN), 0.0, 1.0))
    room_sz_norm = np.clip(room_sz / _ROOM_SZ_MAX, 0.0, 1.0)
    eps = 1e-8
    mic_center_norm = np.clip(mic_center / (room_sz + eps), 0.0, 1.0)
    volume_m3 = float(room_sz[0] * room_sz[1] * room_sz[2])
    log_vol = np.log(max(volume_m3, eps))
    volume_norm = float(
        np.clip(
            (log_vol - _LOG_VOL_MIN) / (_LOG_VOL_MAX - _LOG_VOL_MIN),
            0.0,
            1.0,
        )
    )

    film_norm = np.array(
        [
            rt60_norm,
            room_sz_norm[0],
            room_sz_norm[1],
            room_sz_norm[2],
            mic_center_norm[0],
            mic_center_norm[1],
            mic_center_norm[2],
            volume_norm,
        ],
        dtype=np.float64,
    )
    film_raw = {
        "RT60_s": RT60,
        "room_sz_m": tuple(room_sz.tolist()),
        "mic_center_m": tuple(mic_center.tolist()),
        "volume_m3": volume_m3,
    }
    return film_norm, film_raw


def _nearest_wall_label(x: float, y: float, Lx: float, Ly: float) -> str:
    dists = {"x0 (y-wall low)": x, "xL (y-wall high)": Lx - x, "y0 (x-wall low)": y, "yL (x-wall high)": Ly - y}
    return min(dists, key=dists.get)


def _split_subdir(split: str) -> str:
    subdir = SPLIT_TO_SUBDIR.get(split.lower())
    if subdir is None:
        raise ValueError(f"Unknown split {split!r}. Use train, validation, or test.")
    return subdir


def _collect_raw_rir_paths(dataset_dir: Path, split: str) -> List[str]:
    """List .npz paths from a raw data/rec-rir/<name>/{train,validation,test}/ folder."""
    subdir = _split_subdir(split)
    split_dir = dataset_dir / subdir
    if split_dir.is_dir():
        paths = sorted(str(p.resolve()) for p in split_dir.glob("*.npz"))
        if paths:
            return paths

    loc_csv = dataset_dir / "rir_localization.csv"
    if loc_csv.is_file():
        needle = f"/{subdir}/"
        paths = []
        with loc_csv.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                rir_path = row.get("rir_path", "").strip()
                if rir_path and needle in Path(rir_path).as_posix():
                    paths.append(rir_path)
        if paths:
            return sorted(paths)

    raise FileNotFoundError(
        f"No RIR .npz files for split {split!r} under {dataset_dir} "
        f"(expected {split_dir}/{{*.npz}} or paths in {loc_csv})"
    )


def _is_raw_dataset_dir(dataset_dir: Path, split: str) -> bool:
    """True when dataset_dir is a generated rec-rir folder, not a config list folder."""
    list_name = SPLIT_TO_LIST.get(split.lower())
    if list_name and (dataset_dir / list_name).is_file():
        return False
    subdir = _split_subdir(split)
    if (dataset_dir / subdir).is_dir() and any((dataset_dir / subdir).glob("*.npz")):
        return True
    if (dataset_dir / "rir_localization.csv").is_file():
        return True
    return False


def _resolve_dataset_paths(
    dataset_dir: Path,
    split: str,
    localization_csv: Optional[Path],
) -> Tuple[List[str], Path, Optional[Path], bool]:
    """Return (rir_paths, localization_csv, gen_config, is_raw_dataset)."""
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    _split_subdir(split)  # validate split name
    is_raw = _is_raw_dataset_dir(dataset_dir, split)

    if is_raw:
        rir_paths = _collect_raw_rir_paths(dataset_dir, split)
        if localization_csv is None:
            localization_csv = dataset_dir / "rir_localization.csv"
        else:
            localization_csv = localization_csv.expanduser().resolve()
        gen_config = dataset_dir / "config.json"
        gen_config = gen_config if gen_config.is_file() else None
    else:
        list_name = SPLIT_TO_LIST[split.lower()]
        rir_list = dataset_dir / list_name
        if not rir_list.is_file():
            raise FileNotFoundError(f"RIR list not found: {rir_list}")
        rir_paths = _read_pathlist(rir_list)

        if localization_csv is None:
            localization_csv = (
                Path("/storage/reie/data/rec-rir")
                / dataset_dir.name
                / "rir_localization.csv"
            )
        else:
            localization_csv = localization_csv.expanduser().resolve()
        gen_config = Path("/storage/reie/data/rec-rir") / dataset_dir.name / "config.json"
        gen_config = gen_config if gen_config.is_file() else None

    if not localization_csv.is_file():
        raise FileNotFoundError(f"Localization CSV not found: {localization_csv}")

    return rir_paths, localization_csv, gen_config, is_raw


def render_sample(
    rir_path: Path,
    out_path: Path,
    localization: Optional[Tuple[float, float]],
    mic_wall_offset: Optional[float],
    mic_edge_width: float = 0.0,
    title_extra: str = "",
) -> None:
    data = np.load(rir_path, allow_pickle=True)
    room_sz = np.asarray(data["room_sz"]).ravel()[:3].astype(np.float64)
    Lx, Ly, Lz = room_sz.tolist()

    pos_rcv = np.asarray(data["pos_rcv"], dtype=np.float64)
    if pos_rcv.ndim == 1:
        pos_rcv = pos_rcv.reshape(-1, 3)
    pos_src = np.asarray(data["pos_src"], dtype=np.float64)
    if pos_src.ndim == 1:
        pos_src = pos_src.reshape(1, -1)
    src = pos_src[0]
    mic_center = pos_rcv.mean(axis=0)

    film_norm, film_raw = extract_room_params(data)
    RT60 = film_raw["RT60_s"]

    fig, (ax_room, ax_text) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    # --- Room floor plan (XY) ------------------------------------------------
    margin = 0.35
    ax_room.set_xlim(-margin, Lx + margin)
    ax_room.set_ylim(-margin, Ly + margin)
    ax_room.set_aspect("equal", adjustable="box")
    ax_room.add_patch(
        Rectangle((0, 0), Lx, Ly, fill=False, edgecolor="black", linewidth=2.0)
    )

    if mic_wall_offset is not None:
        o = float(mic_wall_offset)
        w = float(mic_edge_width)
        inner = Rectangle(
            (o, o),
            max(Lx - 2 * o, 0),
            max(Ly - 2 * o, 0),
            fill=False,
            edgecolor="#888888",
            linewidth=1.0,
            linestyle="--",
            label=f"mic inset (offset={o:.2f} m)",
        )
        ax_room.add_patch(inner)
        if w > 0.0:
            band = Rectangle(
                (o, o),
                max(Lx - 2 * o, 0),
                w,
                facecolor="#4393c3",
                edgecolor="#2166ac",
                alpha=0.2,
                linewidth=1.0,
                label=f"y0 band [{o:.2f}, {o + w:.2f}] m",
            )
            ax_room.add_patch(band)

    ax_room.scatter(
        pos_rcv[:, 0],
        pos_rcv[:, 1],
        s=80,
        c="#2166ac",
        marker="s",
        zorder=4,
        label="mic(s)",
    )
    ax_room.scatter(
        mic_center[0],
        mic_center[1],
        s=120,
        c="#2166ac",
        marker="x",
        linewidths=2,
        zorder=5,
    )
    ax_room.scatter(
        src[0], src[1], s=140, c="#b2182b", marker="*", zorder=5, label="source"
    )

    arrow = FancyArrowPatch(
        (mic_center[0], mic_center[1]),
        (src[0], src[1]),
        arrowstyle="-|>",
        mutation_scale=14,
        color="#4d4d4d",
        linewidth=1.5,
        zorder=3,
    )
    ax_room.add_patch(arrow)

    wall_lbl = _nearest_wall_label(mic_center[0], mic_center[1], Lx, Ly)
    ax_room.set_xlabel("x (m)")
    ax_room.set_ylabel("y (m)")
    ax_room.grid(True, linestyle=":", alpha=0.4)
    ax_room.legend(loc="upper right", fontsize=8)

    subtitle = (
        f"{rir_path.name}  |  room {Lx:.1f}×{Ly:.1f}×{Lz:.1f} m  |  RT60={RT60:.2f} s\n"
        f"mic wall: {wall_lbl}  |  z_mic={mic_center[2]:.2f} m  z_src={src[2]:.2f} m"
    )
    if localization is not None:
        ang, rad = localization
        subtitle += f"  |  label: angle={ang:.1f}°  radius={rad:.2f} m"
    if title_extra:
        subtitle = title_extra + "\n" + subtitle
    ax_room.set_title(subtitle, fontsize=9)

    # Global +x reference for angle convention (0° = +x)
    ax_room.annotate(
        "",
        xy=(0.8, 0.15),
        xytext=(0.2, 0.15),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="green", lw=1.2),
    )
    ax_room.text(0.82, 0.12, "0° (+x)", transform=ax_room.transAxes, fontsize=7, color="green")

    # --- FiLM parameters text panel ------------------------------------------
    ax_text.axis("off")
    lines = ["FiLM room parameters", "─" * 36, "Raw (physical):"]
    lines.append(f"  RT60        = {film_raw['RT60_s']:.3f} s")
    rs = film_raw["room_sz_m"]
    lines.append(f"  room_sz     = ({rs[0]:.2f}, {rs[1]:.2f}, {rs[2]:.2f}) m")
    mc = film_raw["mic_center_m"]
    lines.append(f"  mic_center  = ({mc[0]:.2f}, {mc[1]:.2f}, {mc[2]:.2f}) m")
    lines.append(f"  volume      = {film_raw['volume_m3']:.1f} m³")
    lines.append("")
    lines.append("Normalized (model input [0,1]):")
    for name, val in zip(FILM_NAMES, film_norm):
        lines.append(f"  {name:12s} = {val:.4f}")
    lines.append("")
    lines.append(f"Positions (from npz):")
    lines.append(f"  source (x,y) = ({src[0]:.2f}, {src[1]:.2f}) m")
    lines.append(f"  mic ctr (x,y)= ({mic_center[0]:.2f}, {mic_center[1]:.2f}) m")

    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Render 2D top-down room layouts for RIR dataset samples with FiLM parameters."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help=(
            "Dataset config folder with reie-{train,validation,test}-rir.txt "
            "(e.g. config/rir-uniform-angle-fullroom-edge), or a raw generated "
            "dataset folder under data/rec-rir/... with train|validation|test/ subdirs"
        ),
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to render (default: 8)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: <dataset_dir>/renders/<split>)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Which RIR list to use: train, validation, or test (default: validation)",
    )
    parser.add_argument(
        "--localization-csv",
        type=str,
        default=None,
        help=(
            "Path to rir_localization.csv (default: <raw-dataset>/rir_localization.csv "
            "or data/rec-rir/<config-name>/rir_localization.csv)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for subsampling (default: 42)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Use the first N samples starting at this index instead of random subsampling",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Take the first N paths in list order (honors --start-index); default is random sample",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    loc_csv = Path(args.localization_csv) if args.localization_csv else None

    try:
        paths, loc_path, gen_config_path, is_raw = _resolve_dataset_paths(
            dataset_dir, args.split, loc_csv
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    mic_wall_offset: Optional[float] = None
    mic_edge_width: float = 0.0
    if gen_config_path is not None:
        with gen_config_path.open("r") as handle:
            gen_cfg = json.load(handle)
        if gen_cfg.get("mic_wall_offset") is not None:
            mic_wall_offset = float(gen_cfg["mic_wall_offset"])
        mic_edge_width = float(gen_cfg.get("mic_edge_width", 0.0) or 0.0)

    split_key = _split_subdir(args.split)
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (dataset_dir.resolve() / "renders" / split_key)
    )

    if not paths:
        print(f"No RIR paths for split {args.split!r} in {dataset_dir}", file=sys.stderr)
        return 1

    n = max(1, args.num_samples)
    if args.sequential:
        start = max(0, args.start_index)
        chosen = paths[start : start + n]
    else:
        rng = np.random.default_rng(args.seed)
        if n >= len(paths):
            chosen = paths
        else:
            idx = rng.choice(len(paths), size=n, replace=False)
            chosen = [paths[i] for i in sorted(idx)]

    loc_map = _load_localization_map(loc_path)
    dataset_kind = "raw" if is_raw else "config"
    print(f"Dataset:      {dataset_dir.resolve()}  ({dataset_kind})")
    print(f"RIR paths:    {len(paths)} for split {split_key!r}")
    print(f"Localization: {loc_path}")
    print(f"Output:       {out_dir}")
    print(f"Rendering {len(chosen)} sample(s)...")

    for i, rir_path_str in enumerate(chosen):
        rir_path = Path(rir_path_str)
        if not rir_path.is_file():
            print(f"  [skip] missing: {rir_path}", file=sys.stderr)
            continue
        loc = _lookup_localization(loc_map, rir_path_str)
        out_file = out_dir / f"{i:04d}_{rir_path.stem}.png"
        render_sample(
            rir_path,
            out_file,
            localization=loc,
            mic_wall_offset=mic_wall_offset,
            mic_edge_width=mic_edge_width,
            title_extra=f"sample {i + 1}/{len(chosen)}  |  {dataset_dir.name}",
        )
        print(f"  saved {out_file}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
