#!/usr/bin/env python3
"""
match_but_dataset.py

Walk the BUT ReverbDB and select mic–speaker pairs whose room size and
mic-wall proximity match the rir-uniform-angle-fullroom-edge-y0-band
synthetic distribution.  Outputs a CSV with the same
(rir_path, angle_deg, radius_m) label columns, extended with provenance
metadata.

Rotational symmetry
-------------------
The synthetic dataset places mics on the low-y wall (y ∈ [0.4, 0.8] m).
Here we apply the same band constraint to *any* wall by working in a
local coordinate frame where y_local points inward from the mic's nearest
wall.  This makes the selection invariant to room orientation.

Angle convention (matches gen_rir.py::compute_angle_radius)
-----------------------------------------------------------
  angle = atan2(Δy_local, Δx_local) % 360   then folded to [0, 180]:
  if angle > 180: angle = 360 – angle

  nearest wall  │  y_local    │  x_local
  ──────────────┼─────────────┼──────────
  depth=0       │  +Δdepth    │  Δwidth
  depth=D       │  −Δdepth    │  Δwidth
  width=0       │  +Δwidth    │  Δdepth
  width=W       │  −Δwidth    │  Δdepth

Usage
-----
  python match_but_dataset.py \\
      --but_root /storage/data/but \\
      --out_csv  /storage/reie/REC-RIR-LOCALIZATION/config/but_matched.csv

  # relax speaker height / wall-clearance constraints:
  python match_but_dataset.py --no_spk_height_filter --no_spk_wall_filter

  # custom limits:
  python match_but_dataset.py \\
      --mic_wall_offset 0.4 --mic_edge_width 0.4 \\
      --rt60_lim 0.2 1.5 --room_lim_xy 3 15 --room_lim_z 2.5 6

BUT metadata caveats
--------------------
- Hotel_SkalskyDvur_ConferenceRoom2 and VUT_FIT_L227 have unreliable
  position measurements (flagged in BUT read_me.txt), but both are also
  excluded by the room-size filter, so no special handling is needed.
- Mic heights in BUT vary widely (0.5–2 m); the synthetic dataset fixes
  mic_z_height = 2.0 m.  Mic height is NOT filtered by default but is
  preserved as a column (mic_height) for downstream inspection.
- Each BUT speaker setup has a single speech speaker (EnvSpk1).
- WAV version: the lexicographically last *.wav in the RIR/ subdirectory
  is selected (highest v## index = most recent measurement).
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# BUT metadata parsing
# ---------------------------------------------------------------------------

def parse_meta(path: Path) -> Dict[str, str]:
    """Parse a BUT tab-separated metadata file into a flat key→value dict.

    File format::

        $KeyName<TAB>value

    Blank lines and lines without a tab (section banners) are skipped.
    The leading '$' is removed from every key.
    """
    meta: Dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            key, _, value = line.partition("\t")
            key = key.lstrip("$").strip()
            value = value.strip()
            if key:
                meta[key] = value
    return meta


def _float(meta: Dict[str, str], key: str) -> Optional[float]:
    """Return the float value for *key*, or None if absent / non-numeric."""
    val = meta.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Room size filter
# ---------------------------------------------------------------------------

def room_passes_size(
    meta: Dict[str, str],
    depth_lim: Tuple[float, float],
    width_lim: Tuple[float, float],
    height_lim: Tuple[float, float],
) -> bool:
    """Return True when the room dimensions fall within the given limits."""
    d = _float(meta, "EnvDepth")
    w = _float(meta, "EnvWidth")
    h = _float(meta, "EnvHeight")
    if d is None or w is None or h is None:
        return False
    return (
        depth_lim[0] <= d <= depth_lim[1]
        and width_lim[0] <= w <= width_lim[1]
        and height_lim[0] <= h <= height_lim[1]
    )


# ---------------------------------------------------------------------------
# Mic band filter  (rotational symmetry)
# ---------------------------------------------------------------------------

def mic_in_band(
    mic_d: float,
    mic_w: float,
    room_d: float,
    room_w: float,
    offset: float,
    band_width: float,
) -> Tuple[bool, str, float]:
    """Check whether the mic sits in the wall band [offset, offset+band_width].

    Applies rotational symmetry: the band test uses the distance to the
    *nearest* wall in the horizontal (Depth–Width) plane, regardless of
    which wall that is.

    Returns
    -------
    (in_band, nearest_wall_label, min_wall_dist)
        nearest_wall_label is one of: 'depth0', 'depthMax', 'width0', 'widthMax'
    """
    dists = {
        "depth0":    mic_d,
        "depthMax":  room_d - mic_d,
        "width0":    mic_w,
        "widthMax":  room_w - mic_w,
    }
    nearest_wall = min(dists, key=dists.__getitem__)
    min_dist = dists[nearest_wall]
    in_band = offset <= min_dist <= offset + band_width
    return in_band, nearest_wall, min_dist


# ---------------------------------------------------------------------------
# Angle / radius  (wall-relative local frame, matching gen_rir.py)
# ---------------------------------------------------------------------------

def compute_angle_radius_local(
    mic_d: float,
    mic_w: float,
    spk_d: float,
    spk_w: float,
    nearest_wall: str,
) -> Tuple[float, float]:
    """Compute folded angle [0, 180°] and horizontal radius in a local frame.

    Builds a coordinate frame aligned to the mic's nearest wall so that the
    angle convention matches ``compute_angle_radius()`` in gen_rir.py:

    * y_local  = inward normal direction (away from the nearest wall)
    * x_local  = along-wall direction
    * angle    = atan2(Δy_local, Δx_local) % 360, then folded to [0, 180]

    Parameters
    ----------
    mic_d, mic_w  : absolute mic position (Depth, Width)
    spk_d, spk_w  : absolute speaker position (Depth, Width)
    nearest_wall  : which wall the mic is closest to
    """
    dd = spk_d - mic_d   # depth delta
    dw = spk_w - mic_w   # width delta

    if nearest_wall == "depth0":
        y_local, x_local = dd, dw
    elif nearest_wall == "depthMax":
        y_local, x_local = -dd, dw
    elif nearest_wall == "width0":
        y_local, x_local = dw, dd
    else:  # widthMax
        y_local, x_local = -dw, dd

    radius_m = math.sqrt(dd ** 2 + dw ** 2)
    angle_deg = math.degrees(math.atan2(y_local, x_local)) % 360.0
    if angle_deg > 180.0:
        angle_deg = 360.0 - angle_deg

    return angle_deg, radius_m


# ---------------------------------------------------------------------------
# Speaker wall clearance helper
# ---------------------------------------------------------------------------

def spk_min_wall_dist(
    spk_d: float,
    spk_w: float,
    room_d: float,
    room_w: float,
) -> float:
    """Return the minimum horizontal distance from the speaker to any wall."""
    return min(spk_d, room_d - spk_d, spk_w, room_w - spk_w)


# ---------------------------------------------------------------------------
# WAV file selection
# ---------------------------------------------------------------------------

def latest_rir_wav(mic_pos_dir: Path) -> Optional[Path]:
    """Return the lexicographically last *.wav in the RIR/ subdirectory.

    The naming scheme is ``IR_sweep_*_FS16kHz.v##.wav``; the last entry
    corresponds to the highest version index.
    """
    rir_dir = mic_pos_dir / "RIR"
    if not rir_dir.is_dir():
        return None
    wavs = sorted(rir_dir.glob("*.wav"))
    return wavs[-1] if wavs else None


# ---------------------------------------------------------------------------
# Main matching loop
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "rir_path",
    "angle_deg",
    "radius_m",
    "room",
    "spk_setup",
    "mic_id",
    "room_depth",
    "room_width",
    "room_height",
    "rt60",
    "mic_depth",
    "mic_width",
    "mic_height",
    "nearest_wall",
    "mic_wall_dist",
]


def match_dataset(
    but_root: Path,
    depth_lim: Tuple[float, float],
    width_lim: Tuple[float, float],
    height_lim: Tuple[float, float],
    mic_wall_offset: float,
    mic_edge_width: float,
    rt60_lim: Tuple[float, float],
    spk_height_lim: Optional[Tuple[float, float]],
    min_spk_wall_dist: Optional[float],
    min_radius: float,
    verbose: bool = True,
) -> List[Dict]:
    """Walk the BUT directory tree and return all matching records.

    Parameters
    ----------
    but_root         : root of the BUT ReverbDB (contains one dir per room)
    depth_lim        : (min, max) for room Depth in metres
    width_lim        : (min, max) for room Width in metres
    height_lim       : (min, max) for room Height in metres
    mic_wall_offset  : inner edge of the mic band (distance to nearest wall)
    mic_edge_width   : width of the mic band; outer edge = offset + edge_width
    rt60_lim         : (min, max) RT60 in seconds
    spk_height_lim   : (min, max) speaker Height, or None to skip filter
    min_spk_wall_dist: minimum horizontal distance of speaker to any wall,
                       or None to skip filter
    min_radius       : minimum horizontal mic–speaker distance in metres
    verbose          : print per-room progress to stderr
    """
    records: List[Dict] = []
    rooms_checked = 0
    rooms_passed = 0
    per_room: Dict[str, int] = defaultdict(int)

    for room_dir in sorted(but_root.iterdir()):
        if not room_dir.is_dir():
            continue
        env_meta_path = room_dir / "env_meta.txt"
        if not env_meta_path.exists():
            continue

        rooms_checked += 1
        env_meta = parse_meta(env_meta_path)

        if not room_passes_size(env_meta, depth_lim, width_lim, height_lim):
            if verbose:
                d = _float(env_meta, "EnvDepth")
                w = _float(env_meta, "EnvWidth")
                h = _float(env_meta, "EnvHeight")
                print(
                    f"  SKIP (size) {room_dir.name}  "
                    f"D={d} W={w} H={h}",
                    file=sys.stderr,
                )
            continue

        rooms_passed += 1
        room_d = float(env_meta["EnvDepth"])
        room_w = float(env_meta["EnvWidth"])
        room_h = float(env_meta["EnvHeight"])
        room_name = env_meta.get("EnvName", room_dir.name)

        if verbose:
            print(
                f"  PASS (size) {room_name}  "
                f"D={room_d} W={room_w} H={room_h}",
                file=sys.stderr,
            )

        mic_setup_dir = room_dir / "MicID01"
        if not mic_setup_dir.is_dir():
            if verbose:
                print(f"    no MicID01/ in {room_dir.name}", file=sys.stderr)
            continue

        for spk_dir in sorted(mic_setup_dir.iterdir()):
            if not spk_dir.is_dir():
                continue

            spk_matches = 0
            for mic_pos_dir in sorted(spk_dir.iterdir()):
                if not mic_pos_dir.is_dir():
                    continue
                mic_meta_path = mic_pos_dir / "mic_meta.txt"
                if not mic_meta_path.exists():
                    continue

                mic_meta = parse_meta(mic_meta_path)

                # ── resolve mic field prefix ──────────────────────────────
                # Each mic_meta.txt uses the mic's own ID in the field names:
                # directory "05" → $EnvMic5Depth, $EnvMic5RelRT60, etc.
                # Fall back to the MicID read from the file if the directory
                # name is not numeric.
                try:
                    mic_id_int = int(mic_pos_dir.name)
                except ValueError:
                    raw_id = mic_meta.get("EnvMicID")
                    if raw_id is None:
                        continue
                    try:
                        mic_id_int = int(raw_id)
                    except ValueError:
                        continue
                p = f"EnvMic{mic_id_int}"  # e.g. "EnvMic5"

                # ── retrieve positions ────────────────────────────────────
                mic_d = _float(mic_meta, f"{p}Depth")
                mic_w = _float(mic_meta, f"{p}Width")
                mic_h = _float(mic_meta, f"{p}Height")
                spk_d = _float(mic_meta, "EnvSpk1Depth")
                spk_w = _float(mic_meta, "EnvSpk1Width")
                spk_h = _float(mic_meta, "EnvSpk1Height")
                rt60  = _float(mic_meta, f"{p}RelRT60")

                if any(
                    v is None
                    for v in (mic_d, mic_w, mic_h, spk_d, spk_w, spk_h, rt60)
                ):
                    continue

                # ── RT60 ─────────────────────────────────────────────────
                if not (rt60_lim[0] <= rt60 <= rt60_lim[1]):
                    continue

                # ── mic band (rotational symmetry) ────────────────────────
                in_band, nearest_wall, mic_wall_d = mic_in_band(
                    mic_d, mic_w, room_d, room_w,
                    mic_wall_offset, mic_edge_width,
                )
                if not in_band:
                    continue

                # ── speaker height ────────────────────────────────────────
                if spk_height_lim is not None:
                    if not (spk_height_lim[0] <= spk_h <= spk_height_lim[1]):
                        continue

                # ── speaker wall clearance ────────────────────────────────
                if min_spk_wall_dist is not None:
                    if spk_min_wall_dist(spk_d, spk_w, room_d, room_w) < min_spk_wall_dist:
                        continue

                # ── angle + radius ────────────────────────────────────────
                angle_deg, radius_m = compute_angle_radius_local(
                    mic_d, mic_w, spk_d, spk_w, nearest_wall
                )
                if radius_m < min_radius:
                    continue

                # ── find WAV ──────────────────────────────────────────────
                wav_path = latest_rir_wav(mic_pos_dir)
                if wav_path is None:
                    continue

                records.append(
                    {
                        "rir_path":      str(wav_path),
                        "angle_deg":     round(angle_deg, 4),
                        "radius_m":      round(radius_m, 4),
                        "room":          room_name,
                        "spk_setup":     spk_dir.name,
                        "mic_id":        mic_pos_dir.name,
                        "room_depth":    room_d,
                        "room_width":    room_w,
                        "room_height":   room_h,
                        "rt60":          round(rt60, 5),
                        "mic_depth":     round(mic_d, 4),
                        "mic_width":     round(mic_w, 4),
                        "mic_height":    round(mic_h, 4),
                        "nearest_wall":  nearest_wall,
                        "mic_wall_dist": round(mic_wall_d, 4),
                    }
                )
                spk_matches += 1
                per_room[room_name] += 1

            if verbose and spk_matches:
                print(
                    f"    {spk_dir.name}: {spk_matches} match(es)",
                    file=sys.stderr,
                )

    if verbose:
        print(
            f"\nRooms checked: {rooms_checked}  |  "
            f"Rooms passing size filter: {rooms_passed}  |  "
            f"Total matching entries: {len(records)}",
            file=sys.stderr,
        )
        if per_room:
            print("\nPer-room breakdown:", file=sys.stderr)
            for room, cnt in sorted(per_room.items()):
                print(f"  {room}: {cnt}", file=sys.stderr)

    return records


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(records: List[Dict], out_path: Path) -> None:
    """Write *records* to a CSV file at *out_path*."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows → {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── paths ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--but_root",
        type=Path,
        default=Path("/storage/data/but"),
        help="Root directory of the BUT ReverbDB  (default: /storage/data/but)",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=Path(
            "/storage/reie/REC-RIR-LOCALIZATION/config/but_matched.csv"
        ),
        help="Output CSV path  (default: …/config/but_matched.csv)",
    )

    # ── room size limits ───────────────────────────────────────────────────
    p.add_argument(
        "--room_lim_xy",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[3.0, 15.0],
        help="Min/max room Depth AND Width in metres  (default: 3 15)",
    )
    p.add_argument(
        "--room_lim_z",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[2.5, 6.0],
        help="Min/max room Height in metres  (default: 2.5 6)",
    )

    # ── mic band parameters ────────────────────────────────────────────────
    p.add_argument(
        "--mic_wall_offset",
        type=float,
        default=0.4,
        help="Inner edge of mic band (min dist to nearest wall, metres)  "
             "(default: 0.4, matches mic_wall_offset in config.json)",
    )
    p.add_argument(
        "--mic_edge_width",
        type=float,
        default=0.4,
        help="Width of mic band in metres; outer edge = offset + edge_width  "
             "(default: 0.4, matches mic_edge_width in config.json)",
    )

    # ── RT60 ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--rt60_lim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[0.2, 1.5],
        help="RT60 range in seconds  (default: 0.2 1.5)",
    )

    # ── speaker filters ────────────────────────────────────────────────────
    p.add_argument(
        "--no_spk_height_filter",
        action="store_true",
        help="Disable the speaker height filter (default: require Height ∈ [1, 2] m)",
    )
    p.add_argument(
        "--spk_height_lim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[1.0, 2.0],
        help="Speaker height range in metres  (default: 1.0 2.0)",
    )
    p.add_argument(
        "--no_spk_wall_filter",
        action="store_true",
        help="Disable the speaker wall-clearance filter "
             "(default: require speaker ≥ 1 m from every wall)",
    )
    p.add_argument(
        "--min_spk_wall_dist",
        type=float,
        default=1.0,
        help="Minimum horizontal distance from speaker to any wall in metres  "
             "(default: 1.0)",
    )

    # ── radius ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--min_radius",
        type=float,
        default=0.3,
        help="Minimum horizontal mic–speaker distance in metres  (default: 0.3)",
    )

    # ── misc ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-room progress output",
    )

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    depth_lim  = tuple(args.room_lim_xy)
    width_lim  = tuple(args.room_lim_xy)
    height_lim = tuple(args.room_lim_z)

    spk_height_lim   = None if args.no_spk_height_filter else tuple(args.spk_height_lim)
    min_spk_wall_dist = None if args.no_spk_wall_filter  else args.min_spk_wall_dist

    if not args.quiet:
        print("BUT Dataset Matcher", file=sys.stderr)
        print(f"  but_root        : {args.but_root}", file=sys.stderr)
        print(f"  room Depth/Width: {depth_lim} m", file=sys.stderr)
        print(f"  room Height     : {height_lim} m", file=sys.stderr)
        print(f"  mic band        : [{args.mic_wall_offset}, "
              f"{args.mic_wall_offset + args.mic_edge_width}] m from nearest wall",
              file=sys.stderr)
        print(f"  RT60            : {tuple(args.rt60_lim)} s", file=sys.stderr)
        print(f"  spk height      : {spk_height_lim} m", file=sys.stderr)
        print(f"  spk wall dist   : ≥ {min_spk_wall_dist} m", file=sys.stderr)
        print(f"  min radius      : {args.min_radius} m", file=sys.stderr)
        print(f"  out_csv         : {args.out_csv}", file=sys.stderr)
        print("", file=sys.stderr)

    records = match_dataset(
        but_root          = args.but_root,
        depth_lim         = depth_lim,
        width_lim         = width_lim,
        height_lim        = height_lim,
        mic_wall_offset   = args.mic_wall_offset,
        mic_edge_width    = args.mic_edge_width,
        rt60_lim          = tuple(args.rt60_lim),
        spk_height_lim    = spk_height_lim,
        min_spk_wall_dist = min_spk_wall_dist,
        min_radius        = args.min_radius,
        verbose           = not args.quiet,
    )

    if not records:
        print(
            "\nNo matching entries found.  "
            "Try --no_spk_height_filter or --no_spk_wall_filter to relax constraints.",
            file=sys.stderr,
        )
        sys.exit(0)

    write_csv(records, args.out_csv)


if __name__ == "__main__":
    main()
