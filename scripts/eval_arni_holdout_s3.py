#!/usr/bin/env python3
"""Evaluate a checkpoint on Arni S3 source hold-out tree."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from eval_but import run  # noqa: E402


def main() -> None:
    film_mode = "normal"
    ckpt = Path(
        "/storage/reie/experiments/"
        "epoch35-zfree-wideband-w16-mix-arni-hold-S3-2gpu/ckpt/best.tar"
    )
    tag = "arni_mix"
    if len(sys.argv) > 1:
        film_mode = sys.argv[1]
    if len(sys.argv) > 2:
        ckpt = Path(sys.argv[2])
        tag = sys.argv[3] if len(sys.argv) > 3 else ckpt.parent.parent.name

    toml = PROJECT / (
        "config/Rec-RIR-heads-only-epoch35-straight-to-15m-zfree-wideband-"
        "w16-mix-arni-hold-S3-2gpu.toml"
    )
    data = Path("/storage/reie/data/arni-mckenzie-holdout-S3")
    speech = PROJECT / "config/reie_validation_speech_quick.txt"
    out = PROJECT / f"results/arni_holdout_S3_{tag}_{film_mode}"

    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    if not (data / "prep_summary.json").is_file():
        raise SystemExit(f"holdout tree not ready: {data}")

    run(
        config_path=toml,
        checkpoint_path=ckpt,
        but_csv=None,
        dataset_root=data,
        speech_txt=speech,
        noise_txt=None,
        clips_per_rir=20,
        seq_len=4.0,
        snr_range=(20.0, 20.0),
        noisy_proportion=1.0,
        out_dir=out,
        device_str="cuda",
        no_room_conditioning=False,
        estimate_rt60=False,
        batch_size=4,
        num_workers=0,
        seed=42,
        filter_to_film_volume_range=True,
        film_mode=film_mode,
    )


if __name__ == "__main__":
    main()
