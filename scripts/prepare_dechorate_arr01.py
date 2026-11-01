#!/usr/bin/env python3
"""Build a BUT-style eval tree from selected dEchorate arrays.

The Zenodo v2 ``dEchorate_rirs_gzip7.hdf5`` (md5 8b42a0e99766acce8a100130774bd5b3)
has a superblock EOA of 2048 bytes and a broken local-heap pointer, so h5py
cannot list or read the ``rir`` group. RIRs are taken instead from the official
SOFA release (sofaconventions.org), which stores the same measured 48 kHz IRs.

By default keeps capsules on arrays 0/1, directional + omni chirp sources, inward after
rotating the nearest wall to y=0, resamples 48 kHz → 16 kHz, writes
rotated_metadata.json + RIR wav.

Primary tree: /storage/reie/data/dechorate-arr01
"""
from __future__ import annotations

import csv
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def compute_angle_radius_rotated(
    source_x: float, source_y: float, mic_x: float, mic_y: float
) -> Tuple[float, float]:
    dx = source_x - mic_x
    dy = source_y - mic_y
    radius_m = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    if angle_deg > 180.0:
        angle_deg = 360.0 - angle_deg
    return angle_deg, radius_m


SR_OUT = 16000
# Measured shoebox from SOFA RoomCornerB / annotations.h5 (paper ~6 x 6 x 2.4).
ROOM = (5.705, 5.965, 2.355)
RAW_DIR = Path("/storage/reie/data/dechorate-raw")
SOFA_DIR = RAW_DIR / "sofa"
OUT_DIR = Path("/storage/reie/data/dechorate-arr01")
CSV_URL = "https://zenodo.org/records/6576203/files/dEchorate_database.csv?download=1"
CSV_PATH = RAW_DIR / "dEchorate_database.csv"
SOFA_BASE = "https://www.sofaconventions.org/data/database/dechorate/"
ANNOT_PATH = SOFA_DIR / "dEchorate_annotations.h5"

# Table 5, 1000 Hz band (s). CSV furniture code is 020002 (paper 010001, ~0.26 s).
RT60_1KHZ = {
    "000000": 0.14,
    "010000": 0.19,
    "011000": 0.33,
    "011100": 0.34,
    "011110": 0.56,
    "011111": 0.73,
    "001000": 0.19,
    "000100": 0.19,
    "000010": 0.18,
    "000001": 0.19,
    "010001": 0.26,
    "020002": 0.26,
}

KEEP_SRC_TYPES = {"directional", "omnidirectional"}
DEFAULT_KEEP_ARRAYS = {0, 1}
SOFA_MICS = {
    0: "mics1-5",
    1: "mics6-10",
    2: "mics11-15",
    3: "mics16-20",
    4: "mics21-25",
    5: "mics26-30",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"exists {dest} ({dest.stat().st_size} bytes)", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "rec-rir-dechorate/1.0"})
    with urllib.request.urlopen(req, timeout=180) as src, tmp.open("wb") as out:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)", flush=True)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def nearest_wall(x: float, y: float, Lx: float, Ly: float) -> Tuple[str, float]:
    d = {"x0": x, "xL": Lx - x, "y0": y, "yL": Ly - y}
    wall = min(d, key=d.get)
    return wall, d[wall]


def rotate_to_y0(mx, my, mz, sx, sy, sz, wall, Lx, Ly):
    """Proper rotation so the mic's closest wall sits on y=0.

    The mic is not moved onto the wall; mic_y stays the wall offset.
    90° walls swap plan extents so FiLM room_sz is (along-wall, inward).

    These are orientation-preserving (det = +1). Mapping x=0 with
    (x, y) → (y, x) is a reflection and flips left/right; do not use it.
      y0: identity
      x0: 90° CCW   (x, y) → (Ly - y, x)
      yL: 180°      (x, y) → (Lx - x, Ly - y)
      xL: 90° CW    (x, y) → (y, Lx - x)
    """
    if wall == "y0":
        return mx, my, mz, sx, sy, sz, Lx, Ly, "nearest_wall_y0", 0
    if wall == "yL":
        return Lx - mx, Ly - my, mz, Lx - sx, Ly - sy, sz, Lx, Ly, "nearest_wall_yL", 180
    if wall == "x0":
        return Ly - my, mx, mz, Ly - sy, sx, sz, Ly, Lx, "nearest_wall_x0", 90
    if wall == "xL":
        return my, Lx - mx, mz, sy, Lx - sx, sz, Ly, Lx, "nearest_wall_xL", 270
    raise ValueError(wall)


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float64).ravel()
    if int(sr_in) == int(sr_out):
        return wav
    try:
        import librosa

        return librosa.resample(wav, orig_sr=int(sr_in), target_sr=int(sr_out))
    except ImportError:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(wav.astype(np.float32))
        return AF.resample(t, int(sr_in), int(sr_out)).numpy().astype(np.float64)


def _peak_norm(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x))
    return x / peak if peak > 1e-12 else x


def _sofa_name(room: str, src_id: int, arr: int) -> str:
    return f"dEchorate_room{room}_src{src_id + 1}_arr{arr + 1}_{SOFA_MICS[arr]}.sofa"


def _load_sofa_rir(path: Path, local_ch: int) -> Tuple[np.ndarray, int]:
    import h5py

    with h5py.File(path, "r") as f:
        ir = np.asarray(f["Data.IR"], dtype=np.float64)
        fs = int(np.asarray(f["Data.SamplingRate"]).reshape(-1)[0])
    if ir.ndim != 3 or ir.shape[1] <= local_ch:
        raise RuntimeError(f"{path} IR shape {ir.shape} cannot index ch={local_ch}")
    return ir[0, local_ch, :], fs


def _room_from_annotations() -> Tuple[float, float, float]:
    import h5py

    if not ANNOT_PATH.is_file():
        return ROOM
    with h5py.File(ANNOT_PATH, "r") as f:
        sz = np.asarray(f["room_size"], dtype=np.float64).ravel()
    if sz.size != 3:
        return ROOM
    return float(sz[0]), float(sz[1]), float(sz[2])


def rewrite_rotated_metadata(root: Path) -> dict:
    """Recompute rotated coords from stored original geometry. Wavs are not touched."""
    n = 0
    walls = {}
    for meta_path in sorted(root.rglob("rotated_metadata.json")):
        if meta_path.is_symlink():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        orig = meta["original"]
        mx = float(orig["mic"]["width"])
        my = float(orig["mic"]["depth"])
        mz = float(orig["mic"]["height"])
        sx = float(orig["source"]["width"])
        sy = float(orig["source"]["depth"])
        sz = float(orig["source"]["height"])
        Lx = float(orig["room"]["width"])
        Ly = float(orig["room"]["depth"])
        wall = str((meta.get("rotation") or {}).get("nearest_wall") or "")
        if not wall:
            wall, _ = nearest_wall(mx, my, Lx, Ly)
        mx2, my2, mz2, sx2, sy2, sz2, Lx2, Ly2, rot_mode, rot_deg = rotate_to_y0(
            mx, my, mz, sx, sy, sz, wall, Lx, Ly
        )
        angle, radius = compute_angle_radius_rotated(sx2, sy2, mx2, my2)
        meta.setdefault("rotation", {})
        meta["rotation"]["mode"] = rot_mode
        meta["rotation"]["nearest_wall"] = wall
        meta["rotation"]["degrees"] = rot_deg
        meta["rotation"]["handedness"] = "proper_rotation"
        rot = meta.setdefault("rotated", {})
        rot["room_sz_x"] = Lx2
        rot["room_sz_y"] = Ly2
        rot["mic_x"] = mx2
        rot["mic_y"] = my2
        rot["mic_z"] = mz2
        rot["source_x"] = sx2
        rot["source_y"] = sy2
        rot["source_z"] = sz2
        ac = meta.setdefault("acoustics_from_mic_meta", {})
        ac["rir_distance_m"] = radius
        ac["d_wall_m"] = my2
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        walls[wall] = walls.get(wall, 0) + 1
        n += 1
    summary_path = root / "prep_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["rotation"] = (
            "proper rotation, nearest wall -> y=0, mic keeps wall offset; "
            "90 deg walls swap plan extents; x0 is (Ly-y, x) not (y, x)"
        )
        summary["n_rewritten"] = n
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"n_rewritten": n, "walls": walls}


def main() -> None:
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rewrite-meta-only",
        action="store_true",
        help="Recompute rotated_metadata.json from original coords; do not touch wavs.",
    )
    parser.add_argument(
        "--arrays",
        default=",".join(str(a) for a in sorted(DEFAULT_KEEP_ARRAYS)),
        help="Comma-separated zero-based dEchorate array IDs to prepare, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output BUT-style tree root.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    keep_arrays = {int(x.strip()) for x in args.arrays.split(",") if x.strip()}
    unknown = sorted(keep_arrays - set(SOFA_MICS))
    if unknown:
        raise SystemExit(f"unsupported array ids {unknown}; known ids are {sorted(SOFA_MICS)}")
    if args.rewrite_meta_only:
        summary = rewrite_rotated_metadata(out_dir)
        print(json.dumps(summary, indent=2), flush=True)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SOFA_DIR.mkdir(parents=True, exist_ok=True)
    _download(CSV_URL, CSV_PATH)
    _download(SOFA_BASE + "dEchorate_annotations.h5", ANNOT_PATH)

    Lx, Ly, Lz = _room_from_annotations()
    vol = Lx * Ly * Lz
    print(f"room_sz=({Lx}, {Ly}, {Lz}) vol={vol:.3f} m^3", flush=True)

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    seen = set()
    kept: List[dict] = []
    skipped = {"type": 0, "array": 0, "back": 0, "nogeom": 0, "silence": 0}
    for r in rows:
        if (r.get("mic_type") or "") != "capsule":
            continue
        if (r.get("src_signal") or "") != "chirp":
            skipped["silence"] += 1
            continue
        if not r.get("src_pos_x"):
            skipped["nogeom"] += 1
            continue
        arr = int(float(r["array_id"]))
        if arr not in keep_arrays:
            skipped["array"] += 1
            continue
        if (r.get("src_type") or "") not in KEEP_SRC_TYPES:
            skipped["type"] += 1
            continue
        mid = int(float(r["mic_id"]))
        sid = int(float(r["src_id"]))
        room = str(r["room_code"])
        key = (room, arr, mid, sid)
        if key in seen:
            continue
        seen.add(key)
        mx, my, mz = _f(r["mic_pos_x"]), _f(r["mic_pos_y"]), _f(r["mic_pos_z"])
        sx, sy, sz = _f(r["src_pos_x"]), _f(r["src_pos_y"]), _f(r["src_pos_z"])
        wall, dmic = nearest_wall(mx, my, Lx, Ly)
        mx2, my2, mz2, sx2, sy2, sz2, Lx2, Ly2, rot_mode, rot_deg = rotate_to_y0(
            mx, my, mz, sx, sy, sz, wall, Lx, Ly
        )
        if (sy2 - my2) <= 0:
            skipped["back"] += 1
            continue
        angle, radius = compute_angle_radius_rotated(sx2, sy2, mx2, my2)
        kept.append({
            "room": room,
            "arr": arr,
            "mid": mid,
            "sid": sid,
            "src_type": r["src_type"],
            "wall": wall,
            "dmic": dmic,
            "rot_mode": rot_mode,
            "rot_deg": rot_deg,
            "Lx2": Lx2,
            "Ly2": Ly2,
            "mx": mx, "my": my, "mz": mz,
            "sx": sx, "sy": sy, "sz": sz,
            "mx2": mx2, "my2": my2, "mz2": mz2,
            "sx2": sx2, "sy2": sy2, "sz2": sz2,
            "angle": angle,
            "radius": radius,
            "furniture": str(r.get("room_fornitures")),
            "local_ch": mid - 5 * arr,
            "sofa": _sofa_name(room, sid, arr),
        })

    rooms = sorted({k["room"] for k in kept})
    srcs = sorted({k["sid"] for k in kept})
    sofa_names = sorted({k["sofa"] for k in kept})
    print(f"unique eval samples after filters: {len(kept)}", flush=True)
    print(f"  rooms={rooms}", flush=True)
    print(f"  srcs={srcs}", flush=True)
    print(f"  sofa files needed={len(sofa_names)}", flush=True)
    print(f"  skipped={skipped}", flush=True)

    missing = [n for n in sofa_names if not (SOFA_DIR / n).is_file() or (SOFA_DIR / n).stat().st_size <= 0]
    print(f"downloading {len(missing)} SOFA files ({len(sofa_names) - len(missing)} already present)", flush=True)

    def _one(name: str) -> str:
        _download(SOFA_BASE + name, SOFA_DIR / name)
        return name

    n_dl = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_one, n) for n in missing]
        for fut in as_completed(futs):
            fut.result()
            n_dl += 1
            if n_dl % 20 == 0 or n_dl == len(futs):
                print(f"  sofa downloaded {n_dl}/{len(futs)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0
    fail_reasons: Dict[str, int] = {}
    for rec in kept:
        sofa_path = SOFA_DIR / rec["sofa"]
        try:
            wav, fs_in = _load_sofa_rir(sofa_path, rec["local_ch"])
        except Exception as exc:  # noqa: BLE001
            reason = type(exc).__name__
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            if n_fail < 8:
                print(f"load fail {rec['sofa']} ch={rec['local_ch']}: {exc}", flush=True)
            n_fail += 1
            continue
        wav = _peak_norm(_resample(wav, fs_in, SR_OUT))
        sample_dir = (
            out_dir
            / f"dEchorate_{rec['room']}"
            / f"A{rec['arr']}_M{rec['mid']:02d}"
            / f"S{rec['sid']}"
        )
        rir_dir = sample_dir / "RIR"
        rir_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(rir_dir / "rir.wav"), wav.astype(np.float32), SR_OUT)
        rt60 = RT60_1KHZ.get(rec["room"], 0.4)
        meta = {
            "dataset": "dEchorate",
            "environment": {
                "env_id": rec["room"],
                "env_name": f"dEchorate_{rec['room']}",
                "env_type": "acoustic_lab",
            },
            "rotation": {
                "mode": rec["rot_mode"],
                "nearest_wall": rec["wall"],
                "degrees": rec["rot_deg"],
            },
            "original": {
                "room": {"width": Lx, "depth": Ly, "height": Lz, "volume": vol},
                "source": {"width": rec["sx"], "depth": rec["sy"], "height": rec["sz"]},
                "mic": {"width": rec["mx"], "depth": rec["my"], "height": rec["mz"]},
            },
            "rotated": {
                "RT60": rt60,
                "RT60_source": "paper_table5_1000Hz",
                "room_sz_x": rec["Lx2"],
                "room_sz_y": rec["Ly2"],
                "room_sz_z": Lz,
                "volume": vol,
                "source_x": rec["sx2"],
                "source_y": rec["sy2"],
                "source_z": rec["sz2"],
                "mic_x": rec["mx2"],
                "mic_y": rec["my2"],
                "mic_z": rec["mz2"],
            },
            "scenario": {
                "speaker": f"S{rec['sid']}",
                "speaker_setup_id": rec["sid"],
                "speaker_index": rec["sid"],
                "microphone": f"A{rec['arr']}_M{rec['mid']:02d}",
                "mic_setup_id": rec["arr"],
                "mic_index": rec["mid"],
                "src_type": rec["src_type"],
            },
            "acoustics_from_mic_meta": {
                "RT30": rt60,
                "RT20": rt60,
                "rir_distance_m": rec["radius"],
                "d_wall_m": rec["dmic"],
                "furniture": rec["furniture"],
            },
        }
        (sample_dir / "rotated_metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  wrote {n_ok}", flush=True)

    summary = {
        "n_ok": n_ok,
        "n_fail": n_fail,
        "fail_reasons": fail_reasons,
        "skipped": skipped,
        "arrays": sorted(keep_arrays),
        "out_dir": str(out_dir),
        "rooms": rooms,
        "srcs": srcs,
        "n_sofa": len(sofa_names),
        "room_sz": [Lx, Ly, Lz],
        "hdf5_layout": "sofa_Data.IR_M_R_N",
        "rir_source": "sofaconventions.org SOFA (gzip7 HDF5 unreadable)",
        "rotation": "proper rotation, nearest wall -> y=0, mic keeps wall offset; 90 deg walls swap plan extents",
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
