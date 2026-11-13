# notebooks

Colab notebooks, one folder per project phase, in chronological order. Every
notebook imports from the `glada` package. Results each notebook wrote are in
the matching folder of [`../experiment_results/`](../experiment_results).
Some notebooks were cleared of outputs after running; their results are still
in `experiment_results/`.

| Folder | Question it answered |
|---|---|
| [`00_demo`](00_demo) | Colab demo: load the released detector and label clips as real or deepfake. |
| [`01_baseline_glada`](01_baseline_glada) | Does the full GLADA stack train end-to-end, and is the implementation correct? |
| [`02_ablation_multibackbone`](02_ablation_multibackbone) | Which components and which backbone help on ITW, measured on six datasets? |
| [`03_phase2_diagnosis`](03_phase2_diagnosis) | Why does ITW EER get worse once the backbone is unfrozen, and does moving the VIB help? |
| [`04_pgvib_development`](04_pgvib_development) | Which layer group's bottleneck matters, measured over three seeds? |
| [`05_cross_dataset_eval`](05_cross_dataset_eval) | Do the per-group VIB results hold on four held-out datasets? |
| [`06_analysis_and_figures`](06_analysis_and_figures) | What does each layer group encode, and the thesis tables and figures. |
| [`utilities`](utilities) | Dataset download and the scripts that generated the phase 03 and 04 notebooks. |

EER values below are best-epoch ITW EER unless stated otherwise.
