# utilities

| File | Purpose |
|---|---|
| `download_eval_datasets.ipynb` | Downloads the evaluation datasets to Google Drive from Colab. |
| `april13_builders/_phase_common.py` | Shared cells (setup, imports, training loop) that the builders below assemble into notebooks. |
| `april13_builders/build_phase_vib_stagger.py` | Generates `03_phase2_diagnosis/phase_vib_stagger.ipynb`. |
| `april13_builders/build_phase_aug_validation.py` | Generates `03_phase2_diagnosis/phase_aug_validation.ipynb`. |
| `april13_builders/build_phase_a_leaveout.py` | Generates `04_pgvib_development/phase_a_leaveout.ipynb`. |
| `april13_builders/build_phase_a_norb.py` | Generates `04_pgvib_development/phase_a_norb.ipynb`. |
| `april13_builders/build_phase_pgvib_monolithic.py` | Generates `04_pgvib_development/phase_pgvib_monolithic.ipynb`. |
| `april13_builders/build_phase_pgvib_g0only.py` | Generates `04_pgvib_development/phase_pgvib_g0only.ipynb`. |
| `april13_builders/build_phase_pgvib_replicate.py` | Generates `04_pgvib_development/phase_pgvib_replicate.ipynb`. |
| `april13_builders/build_cross_dataset_eval_pgvib.py` | Generates the cross-dataset evaluation notebook that was split into the part A / part B notebooks in phase 05. |
| `april13_builders/phase2_fixes_sweep.py` | Command-line sweep of eight phase 2 fixes (lower LR, layer freezing, KL re-annealing, VIB re-initialisation, lower β, no adversaries). Produced the `fix_*` runs in phase 03. |

To regenerate a notebook (without running it): `python april13_builders/build_<name>.py`.
