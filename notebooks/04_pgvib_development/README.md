# 04 Per-group VIB development (May 2026)

WavLM-Base-Plus (3 layer groups: g0 lowest, g2 highest). These runs are the
ITW results in the ICTAI 2026 paper and the thesis.

| Notebook | What it does | Result |
|---|---|---|
| `phase_a_leaveout.ipynb` | Makes one group's VIB deterministic at a time (seed 42). | ITW EER rose to 17.61% without g0's bottleneck, 13.80% without g1's, and 11.99% without g2's (reference 11.67%). The benefit sits in g0. |
| `phase_a_norb.ipynb` | Per-group VIB without RawBoost. | 15.45% vs 11.67% with it, so RawBoost was kept. |
| `phase_pgvib_monolithic.ipynb` | Single post-pool VIB at seeds 42, 123, 456, matched to the per-group runs; then a β probe on g0-only VIB. | 14.28 / 16.52 / 13.17% (mean 14.66%). β probe at seed 42: 10.73% (β = 0.005), 11.77% (0.05), 11.81% (0.1); default β is 0.01. |
| `phase_pgvib_g0only.ipynb` | VIB on g0 only (`vib_disabled_groups=(1, 2)`) at three seeds, plus WavLM-Large. | 13.02 / 13.93 / 13.30% (mean 13.42%, most stable). WavLM-Large: 12.95%. |
| `phase_pgvib_replicate.ipynb` | Seeds 123 and 456 for per-group VIB and for all-but-g0; g0-only on XLS-R 300M. | Per-group 13.40 / 10.83% (mean with seed 42: 11.97%). All-but-g0 14.46 / 14.61% (mean 15.56%). XLS-R: 18.40%. |

Seed-42 per-group VIB comes from `03_phase2_diagnosis/phase_vib_stagger` and seed-42
all-but-g0 from `phase_a_leaveout`.

Results: [`experiment_results/04_pgvib_development`](../../experiment_results/04_pgvib_development).
