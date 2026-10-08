# 03 Phase 2 diagnosis results

WavLM-Large, seed 42 unless noted.

| Folder / file | Produced by | Contents |
|---|---|---|
| `phase_a_mask_validation/` | `phase_a_mask_validation.ipynb` | One-epoch mask check. |
| `sweep_lr<lr>_freeze<n>_p1_<epochs>/` | `glada_sweep_experiment_diagnostics.ipynb` | Phase 2 sweep over backbone LR (1e-6, 2e-6, 5e-6), frozen bottom layers (0, 12, 18), and phase 1 length (5, 10). |
| `sweep_wavlm_large/` | Early February sweep run | Five-epoch WavLM-Large trial. |
| `P2_diag_control`, `P2_diag_no_ramp`, `P2_diag_glada_only`, `P2_diag_speaker_only` | `phase_b_sweep.ipynb` | Diagnostic sweep isolating each adversarial signal in phase 2. |
| `fix_*`, `phase2_fixes_results.json` | `utilities/april13_builders/phase2_fixes_sweep.py` | Phase 2 fixes: ultra-low LR, LR 1e-6 with 12 or 18 frozen layers, KL re-annealing, VIB re-initialisation, lower β. The JSON holds per-epoch phase 2 EERs and whether each run stayed stable. |
| `vib_stagger/` | `phase_vib_stagger.ipynb` | WavLM-Base-Plus: `per_group_vib` (11.67%, the first per-group VIB run), `staggered` (12.06%), `per_group_vib_staggered` (14.24%). |
| `vib_stagger_L1/` | `phase_vib_stagger.ipynb` | Per-group VIB on WavLM-Large (13.52%) with a multi-dataset evaluation. |
| `aug_validation/` | `phase_aug_validation.ipynb` | Full GLADA, no per-group DA, and WavLM-Large retrained with RawBoost + MUSAN + RIR. |
| `sweeps/sweep_results.json` | February screening sweep | Best EER per backbone screen run in `02_ablation_multibackbone/screen_A_*`. |
