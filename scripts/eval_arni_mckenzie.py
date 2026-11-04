#!/usr/bin/env python3
"""Zero-shot eval of the A0/A1 dEchorate-mixed checkpoint on Arni near-wall RIRs."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from eval_but import run  # noqa: E402


def main() -> None:
    toml = PROJECT / (
        "config/Rec-RIR-heads-only-epoch35-straight-to-15m-zfree-wideband-"
        "w16-mix-dechorate-A01-hold-A2-2gpu.toml"
    )
    ckpt = Path(
        "/storage/reie/experiments/"
        "epoch35-zfree-wideband-w16-mix-dechorate-A01-hold-A2-2gpu/ckpt/best.tar"
    )
    data = Path("/storage/reie/data/arni-mckenzie-nearwall")
    speech = PROJECT / "config/reie_validation_speech_quick.txt"
    out = PROJECT / "results/arni_mckenzie_nearwall_zero_shot_A01mix"

    film_mode = "normal"
    if len(sys.argv) > 1:
        film_mode = sys.argv[1]
        out = PROJECT / f"results/arni_mckenzie_nearwall_zero_shot_A01mix_{film_mode}"

    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    if not (data / "prep_summary.json").is_file():
        raise SystemExit(f"eval tree not ready: {data}")
    if not toml.is_file():
        raise SystemExit(f"config missing: {toml}")

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
