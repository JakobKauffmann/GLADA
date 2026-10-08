# 04 Per-group VIB results

WavLM-Base-Plus unless noted. Each folder also has a `<notebook>_results.json`
with the best EER of every run in it.

| Folder | Produced by | Runs |
|---|---|---|
| `phase_a_leaveout/` | `phase_a_leaveout.ipynb` | `drop_vib_g0`, `drop_vib_g1`, `drop_vib_g2` (seed 42). |
| `phase_a_norb/` | `phase_a_norb.ipynb` | `per_group_vib_no_rb`: per-group VIB without RawBoost. |
| `phase_pgvib_monolithic/` | `phase_pgvib_monolithic.ipynb` | `monolithic_vib_s{42,123,456}`; β probe `g0_only_beta_x{0p5,5,10}` (β = 0.005, 0.05, 0.1). |
| `phase_pgvib_g0only/` | `phase_pgvib_g0only.ipynb` | `g0_only_vib_s{42,123,456}`; `g0_only_large_s42` (WavLM-Large). |
| `phase_pgvib_replicate/` | `phase_pgvib_replicate.ipynb` | `pergroup_vib_s{123,456}`, `drop_vib_g0_s{123,456}`; `g0_only_xlsr_s42` (XLS-R 300M). |

The seed-42 per-group VIB run is in `03_phase2_diagnosis/vib_stagger/per_group_vib`.
