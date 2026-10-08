# reference

Prototype scripts from the CS 297 semester that came before GLADA. The
`glada` package does not import them. They are kept because they produced
the results in `experiment_results/_legacy_cs297/` and show where GLADA's
design came from.

| File | Purpose |
|---|---|
| `domain_adaptation_e2e.py` | First end-to-end domain adaptation baseline on raw audio: frozen WavLM, DANN / CORAL / MMD, optional LoRA. |
| `domain_adaptation_stable.py` | Attempts to stabilise adversarial training: stronger spectral-norm discriminator, discriminator warm-up, gradient penalty. |
| `domain_adaptation_ablation.py` | Ablation harness over DA method (DANN / CORAL / MMD), GRL schedule (linear / delayed / exponential), discriminator strength, gradient penalty, and label smoothing. Produced the `<method>_<schedule>_...` runs in `_legacy_cs297`. |
| `hierda.py` | Direct predecessor of `GLADAModel`: per-layer LayerNorm, grouped attention pooling, per-group discriminators, VIB, and speaker adversary in one script. |
