# Arni / McKenzie 80/20 mic few-shot

Leave-one-mic-out over near-wall receivers R1–R5 (train 4 / test 1), matching the dEchorate mic80 *split*.
Checkpoint: A0/A1 dEchorate-mixed `epoch35-zfree-wideband-w16-mix-dechorate-A01-hold-A2-2gpu`.
Heads-only fine-tune (unfrozen OOM’d here), 8 epochs, lr 1e-4, 8 train clips/RIR.

## Per-fold angle MAE

| Hold-out | Zero-shot | Few-shot | Shuffle | Oracle | Beats?
| --- | ---: | ---: | ---: | ---: | --- |
| R1 | 37.1 | 25.0 | 54.8 | 22.3 | no |
| R2 | 50.3 | 45.2 | 66.8 | 32.1 | no |
| R3 | 96.5 | 54.1 | 67.0 | 31.0 | no |
| R4 | 93.9 | 56.1 | 52.2 | 19.7 | no |
| R5 | 18.7 | 17.1 | 69.7 | 9.6 | no |
| **Mean** | **59.3** | **39.5** | **62.1** | **~23** | **0/5** |

## Takeaway

Few-shot helps vs zero-shot (~59° → ~40°) and beats shuffle, but **does not beat the per-group constant oracle** and `r_within` stays poor/negative. Unlike dEchorate mic80 (~3°), Arni near-wall Ambisonic-W does not unlock localization with an 80/20 mic adaptation.

Paths: `experiments/arni-mckenzie-mic80-wideband-unfrozen/`
