# 02 Ablation and backbone results

WavLM-Base-Plus, seed 42 unless the name says otherwise.

| Folder / file | Produced by | Contents |
|---|---|---|
| `ablation_full_glada`, `ablation_no_glada`, `ablation_no_vib`, `ablation_no_speaker`, `ablation_no_cka`, `ablation_no_rawboost`, `ablation_no_hier`, `ablation_global_da_only`, `ablation_hierarchical_only`, `ablation_no_da`, `ablation_backbone_only` | `glada_ablation.ipynb`, `colab_b_ablation_analysis.ipynb` | The 11 ablation configurations. |
| `ablation_full_<backbone>[_s<seed>]` | `glada_ablation_phase_2.ipynb`, `colab_b_ablation_analysis.ipynb` | Full GLADA on WavLM-Large (seeds 42, 123, 2026), XLS-R 300M, Data2Vec-Large, Wav2Vec2-Robust. |
| `ablation/ablation_results.json` | Ablation notebooks | Best ITW EER and epoch for each ablation run. |
| `ablation/averaged_results.json` | `colab_b_ablation_analysis.ipynb` | ITW and 2019 LA EER after averaging each run's top checkpoints. |
| `multiseed_*_s456/`, `multiseed_results.json` | `glada_multiseed.ipynb` | Seed 456 for full GLADA, no per-group DA, no CKA. |
| `comprehensive_eval/results.json` | `glada_comprehensive_eval*.ipynb`, `glada_eval_incremental.ipynb` | Every ablation and backbone checkpoint on six datasets with CIs. Source of the thesis ablation and backbone tables. |
| `screen_A_backbone_*`, `screen_B_groups_*` | February screening sweep (`glada.trainers.sweep`) | Short runs comparing five backbones and 3, 6, or 8 layer groups on WavLM-Large. Summarised in `03_phase2_diagnosis/sweeps/sweep_results.json`. |
| `sweep_results_A.json`, `sweep_results_B.json` | `glada_sweep_experiment_diagnostics.ipynb` | Scores for the phase 2 LR / freezing sweep whose runs are in `03_phase2_diagnosis/sweep_lr*`. |
