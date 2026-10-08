# glada.models

| File | Purpose |
|---|---|
| `glada_model.py` | `GLADAModel`: SSL backbone, per-layer LayerNorm, hierarchical pooling, a single post-pool VIB, the spoof classifier, per-group domain discriminators, and the speaker adversary. `PerGroupVIBModel` subclasses it to put one VIB on each layer group before its discriminator; `vib_disabled_groups` makes chosen groups deterministic (the g0-only and leave-one-out experiments) and `vib_betas` sets per-group KL weights. |
| `attention.py` | `PerLayerNorm`, `AttentiveStatisticsPooling`, `HierarchicalAttentionPooling` (splits layers into groups and pools each), and `compute_cka` for the CKA diversity loss. |
| `vib.py` | `VIBBottleneck`: Gaussian encoder with reparameterization, closed-form KL to N(0, I), and a clamped log-variance to prevent KL blow-up. |
| `grl.py` | `GradientReversal`: identity forward, negated and scaled gradient backward. Used by the domain and speaker adversaries. |
| `discriminators.py` | `GroupedLayerDomainDiscriminators`: one small MLP domain classifier per layer group. |
| `__init__.py` | Re-exports the classes above. |
