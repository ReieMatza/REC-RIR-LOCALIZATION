# Arni / McKenzie mixed fine-tune (S3 holdout)

Same recipe as dEchorate A01 mix: wideband sim + oversampled real Arni RIRs + dense gpuRIR twins,
fine-tuned from `epoch35-zfree-wideband-w16-mix-dechorate-A01-hold-A2-2gpu` best.

- Train sources: R1–R5 real near-wall Ambisonic-W, **hold out S3** (25 RIRs across abs0/25/50/75/100)
- Twins: ~50 exact + ~5k dense near-wall gpuRIR
- Mix: ~77k list entries, ~35% Arni oversample
- Train: 12 epochs, 2× RTX 6000 Ada, heads-only lr schedule as in config
- Best val (mix): epoch 10, `best_metric` 24.08 / angle MAE 20.7°

## S3 holdout angle MAE (500 clips)

| Model | Normal | Shuffle | Constant baseline |
| --- | ---: | ---: | ---: |
| A01 zero-shot | 55.4° | 78.7° | 25.0° |
| **Arni mix (this)** | **46.9°** | **76.4°** | **25.0°** |

Corr (pred vs truth, between groups): 0.46 → **0.58**. Shuffle still collapses (donor corr ~0).

## Takeaway

Mixed fine-tune + twins **helps** vs A01 zero-shot (~8.5° absolute) and beats FiLM-shuffle, but still
**does not beat the one-constant baseline (~25°)**. With one source held out, per-group oracle is
degenerate (`r_within` nan); the useful bar is overall MAE vs constant.

Paths:
- exp: `experiments/epoch35-zfree-wideband-w16-mix-arni-hold-S3-2gpu/`
- eval: `results/arni_holdout_S3_arni_mix_{normal,shuffle}/`
- prior zero: `results/arni_holdout_S3_A01_zero_{normal,shuffle}/`
