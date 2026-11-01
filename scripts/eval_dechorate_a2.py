#!/usr/bin/env python3
"""Evaluate the widened-band A0/A1 mixed checkpoint on held-out dEchorate A2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from eval_but import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="/storage/reie/experiments/epoch35-zfree-wideband-w16-mix-dechorate-A01-hold-A2-2gpu/ckpt/best.tar",
    )
    p.add_argument(
        "--out",
        default=str(PROJECT / "results/dechorate_A2_w16_mix_A01_holdout"),
    )
    p.add_argument(
        "--data",
        default="/storage/reie/data/dechorate-arr012-holdout-A2",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--film-mode",
        default="normal",
        choices=("normal", "off", "zero", "shuffle"),
        help="FiLM conditioning mode passed through to eval_but.run.",
    )
    args = p.parse_args()

    toml = PROJECT / "config/Rec-RIR-heads-only-epoch35-straight-to-15m-zfree-wideband-2gpu.toml"
    ckpt = Path(args.ckpt)
    data = Path(args.data)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not ready: {ckpt}")
    if not any(data.rglob("rotated_metadata.json")):
        raise SystemExit(f"hold-out tree not ready: {data}")
    run(
        config_path=toml,
        checkpoint_path=ckpt,
        but_csv=None,
        dataset_root=data,
        speech_txt=PROJECT / "config/reie_validation_speech_quick.txt",
        noise_txt=None,
        clips_per_rir=20,
        seq_len=4.0,
        snr_range=(20.0, 20.0),
        noisy_proportion=1.0,
        out_dir=Path(args.out),
        device_str=args.device,
        no_room_conditioning=False,
        estimate_rt60=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        filter_to_film_volume_range=True,
        film_mode=args.film_mode,
    )


if __name__ == "__main__":
    main()
