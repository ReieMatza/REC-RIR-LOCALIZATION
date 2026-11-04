# Arni / McKenzie zero-shot eval (first iteration)

Dataset: McKenzie et al., arXiv:2111.11882 / Zenodo 5720724 (Zylia SH Ambisonic W).
Prepared near-wall receivers R1–R5 only (1.3 m from south wall), all 5 absorber configs × 3 sources = **75 RIRs**.
Checkpoint: `epoch35-zfree-wideband-w16-mix-dechorate-A01-hold-A2-2gpu/ckpt/best.tar` (most dEchorate A0/A1 mix).
No fine-tuning. Protocol matches BUT/dEchorate eval (`eval_but.run`), 20 clips/RIR.

## Results

| Condition | Angle MAE | Radius MAE | r_within (angle) | vs per-group oracle (22.9°) |
| --- | ---: | ---: | ---: | --- |
| FiLM normal | 59.3° | 1.88 m | −0.41 | fails |
| FiLM shuffle | 73.4° | 1.92 m | +0.29 | fails |

Per-absorber angle MAE (normal): abs100 58.2°, abs75 61.0°, abs50 57.5°, abs25 59.1°, abs0 60.7°.

## Takeaway

Zero-shot transfer from the A0/A1 dEchorate-mixed model **does not localize** on Arni near-wall Ambisonic-W RIRs. Normal FiLM beats shuffle (~14°), so FiLM is not noise, but both are far above the 22.9° constant-per-mic oracle. Radius also fails vs a 0.70 m oracle.

Likely contributors: spherical-array / Ambisonic-W domain shift, only 3 source angles per mic, and room/array mismatch vs dEchorate capsules.

## Paths

- Data tree: `/storage/reie/data/arni-mckenzie-nearwall`
- Raw zip (local cache): `/tmp/arni-srir/6dof_SRIRs_zylia_SH.zip`
- Results: `results/arni_mckenzie_nearwall_zero_shot_A01mix_{normal,shuffle}/`
- Scripts: `scripts/prepare_arni_mckenzie.py`, `scripts/eval_arni_mckenzie.py`
