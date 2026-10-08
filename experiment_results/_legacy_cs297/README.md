# Legacy CS 297 results (December 2025 to February 2026)

Runs from the domain adaptation prototypes in [`../../reference/`](../../reference),
before GLADA. They are the baselines GLADA was built to improve on (for
example the DANN and CAAN rows in the thesis appendix).

| Folder pattern | Contents |
|---|---|
| `<method>_<schedule>_<strength>_dlr<w>_gp<0/1>_ls<s>_seed<n>` | `domain_adaptation_ablation.py` sweep: method (`dann`, `coral`, `mmd`, `none`), GRL schedule, discriminator strength, domain loss weight, gradient penalty, label smoothing. |
| `baseline_*`, `dann_*`, `coral_*`, `mmd_*` with `lora` or `frozen` | `domain_adaptation_e2e.py` runs with a frozen or LoRA-adapted backbone at seeds 42, 123, 456. |
| `ablation_*` | Early ablations of LoRA rank, DANN, and the combined prototype. `ablation_study/` holds the aggregated screening table. |
| `exp_A_weak_disc` to `exp_G_asvspoof2021df` | Follow-up experiments: weaker discriminators, bottleneck size, pseudo-labelling, mixup, a backbone × DA-method matrix, augmentation, and ASVspoof 2021 DF evaluation. |
| `dann_ablation_v2/`, `thesis_dann_v2/`, `thesis_full_v1/`, `final_ablation_results.csv` | Aggregated results tables from this period. |

Older runs log to `epoch_metrics.csv`, `step_metrics.csv`, and `training.log`
instead of `history.json`.
