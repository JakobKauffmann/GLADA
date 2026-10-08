# glada.analysis

Used by `notebooks/06_analysis_and_figures/glada_acoustic_analysis.ipynb`.

| File | Purpose |
|---|---|
| `extract_reps.py` | `extract_group_reps`: loads a trained checkpoint and saves each layer group's representation for a set of utterances. |
| `probing.py` | Linear probes per group: `train_linear_probe`, `domain_probe` (ASVspoof vs ITW), and `codec_probe`. |
| `cka_acoustic.py` | `compute_cka` and `cka_vs_acoustic`: CKA between each group's representation and handcrafted acoustic features. |
| `__init__.py` | Re-exports the functions above. |
