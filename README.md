# GLADA: Grouped Layer Adversarial Domain Adaptation for Audio Deepfake Detection

Code for the SJSU MS Computer Science thesis *Targeted Bottlenecking of
Self-Supervised Speech Layers for Generalizable Audio Deepfake Detection*
(Jakob Kauffmann, Spring 2026).

- **Advisor:** Dr. Faranak Abri
- **Committee:** Dr. Amith Kamath Belman, Dr. Saptarshi Sengupta

## Overview

Audio deepfake detectors trained on one corpus tend to break on unfamiliar
recording channels, codecs, and synthesizers. GLADA groups the hidden layers
of a self-supervised speech model (WavLM, XLS-R, Data2Vec, Wav2Vec2) and, per
layer group, applies attentive statistics pooling, a Variational Information
Bottleneck (VIB), and a gradient-reversed domain discriminator. A CKA penalty
keeps the groups diverse and a speaker adversary limits speaker leakage.

```
Raw audio (16 kHz, 4 s)
  -> RawBoost (training only)
  -> SSL backbone (unfrozen in phase 2, gradient checkpointing)
  -> per-layer LayerNorm
  -> layer groups g_1..g_K, attentive statistics pooling per group
  -> VIB (one per group, or a single post-pool VIB)
  -> spoof classifier
     + per-group domain discriminators (GRL)
     + speaker adversary (GRL)
```

Training objective:

```
L = L_spoof + α·L_GLADA + β·L_KL + γ·L_speaker + δ·L_CKA
```

Defaults in `GLADAConfig`: α = 0.1, β = 1e-3, γ = 0.01, δ = 0.01. Training
uses ASVspoof 2019 LA as the labelled source domain and In-the-Wild (ITW) as an
unlabelled target domain.

## Results

Multi-seed EER (%) on `wavlm-base-plus` (3 layer groups, seeds 42/123/456),
trained on ASVspoof 2019 LA only. Mean ± std over seeds; lower is better.

| VIB placement | ITW | ASVspoof 2021 LA | ASVspoof 2021 DF | WaveFake | CodecFake+ |
|---|---|---|---|---|---|
| Single post-pool VIB | 14.66 ± 1.71 | 28.69 ± 0.65 | 44.04 ± 0.01 | 21.36 ± 2.10 | 38.65 ± 1.02 |
| Per-group VIB (all groups) | **11.97 ± 1.31** | 31.18 ± 2.09 | 43.97 ± 0.12 | 23.12 ± 3.81 | 37.49 ± 2.78 |
| Per-group VIB, lowest group only (g0) | 13.42 ± 0.47 | **27.94 ± 0.65** | **43.89 ± 0.12** | **19.98 ± 2.94** | **36.46 ± 2.48** |
| Per-group VIB, all but g0 | 15.56 ± 1.78 | 29.67 ± 1.19 | 43.93 ± 0.15 | 22.61 ± 1.77 | 36.90 ± 1.59 |

Bottlenecking every group gives the best ITW EER. Bottlenecking only the lowest
group is best on every secondary out-of-domain dataset and is the most stable
across seeds. Per-group linear probes and CKA against acoustic descriptors
place most of the codec and channel signal in that lowest group. See the
thesis for the full ablations and analysis.

## Repository layout

```
glada/                  Python package (pip install -e glada/)
  glada/config.py         GLADAConfig (all hyperparameters) and BACKBONE_REGISTRY
  glada/models/           GLADAModel, PerGroupVIBModel, attention pooling, VIB, GRL, discriminators
  glada/trainers/         two-phase trainer (frozen heads, then LLRD fine-tuning), ablation sweeps
  glada/datasets/         ASVspoof 2019/2021, In-the-Wild, WaveFake, CodecFake+ loaders
  glada/augmentation/     RawBoost, MUSAN, RIR
  glada/evaluation/       EER, bootstrap CIs, multi-dataset evaluator, checkpoint averaging
  glada/optimizer/        layer-wise LR decay, OneCycle scheduler
  glada/analysis/         representation extraction, linear probes, CKA vs acoustic features
notebooks/              Colab notebooks, grouped by project phase (below)
experiment_results/     config.json / history.json / eval results for every run, same phase layout
reference/              earlier DANN / CORAL / MMD / hierarchical DA prototypes
DATASETS.md             dataset downloads and expected directory layout
```

Model checkpoints and raw audio are not included.

## Project phases

`notebooks/` and `experiment_results/` share the same numbered folders.

| Folder | What it covers |
|---|---|
| `01_baseline_glada` | First end-to-end GLADA training and evaluation on WavLM-Large |
| `02_ablation_multibackbone` | Component ablation, backbone screen (WavLM-Large, XLS-R 300M, Data2Vec, Wav2Vec2-Robust), group-count screen, multi-seed runs |
| `03_phase2_diagnosis` | Diagnosing instability once the backbone is unfrozen: LR / freezing sweeps, masking and augmentation checks, VIB staggering |
| `04_pgvib_development` | Per-group VIB: leave-one-group-out, single vs per-group VIB, g0-only, multi-seed replication |
| `05_cross_dataset_eval` | Evaluation on ITW, ASVspoof 2021 LA/DF, WaveFake, CodecFake+ |
| `06_analysis_and_figures` | Acoustic CKA analysis, speaker-leakage probe, latency benchmark, figures |
| `07_strengthening_may19` | Checkpoint-averaging runs after submission (not reported in the thesis) |
| `experiment_results/_legacy_cs297` | Results from the earlier CS 297 prototypes in `reference/` |

Most phase 03, 04, and 07 notebooks are generated from the Python builders in
`notebooks/utilities/`. A few phase 03 notebooks (`phase_b_confirm`,
`phase_c_experiment`, `phase_d_ablation`, `phase_e_multiseed`,
`phase_f_evaluation`, `phase_g_diagnostics`) are unexecuted templates; the rest
keep their Colab outputs.

## Setup

Python 3.10 or 3.11 and a CUDA GPU. The thesis runs used Colab A100s.
WavLM-Base-Plus also fits on a 16 GB T4.

```bash
pip install -e glada/
```

Every notebook starts with a setup cell that mounts Google Drive, copies the
package to local disk, and extracts the datasets to `/content/data/`. Edit the
path constants in that cell (`DRIVE_PKG`, `ASV_AUDIO`, `ITW_AUDIO`, and so on)
to match your environment. Download links and the expected layout are in
[`DATASETS.md`](DATASETS.md).

A few settings matter:

- Audio is read with `soundfile.read()`. `torchaudio.load()` and
  `torchaudio.set_audio_backend()` are not used.
- DataLoaders use `num_workers=0`. Higher values caused worker crashes on Colab.
- `torch.compile()` is not used because it breaks WavLM's dynamic masking.
- Keep `domain_loss_weight` at 0.1 or lower. 0.5 destabilized training.

## Reproducing the main results

Only ASVspoof 2019 LA and In-the-Wild (about 13 GB) are needed for training.

1. **Per-group and g0-only VIB.** Run the notebooks in
   `notebooks/04_pgvib_development/`:
   - `phase_pgvib_monolithic.ipynb` for the single post-pool VIB
   - `phase_pgvib_g0only.ipynb` for g0-only (`PerGroupVIBModel` with
     `vib_disabled_groups=(1, 2)`)
   - `phase_pgvib_replicate.ipynb` for the remaining per-group and
     all-but-g0 seeds
   - `phase_a_leaveout.ipynb` for the single-seed leave-one-group-out ablation
2. **Cross-dataset evaluation.** Run `cross_dataset_eval_pgvib_part_a.ipynb`
   (ITW, ASVspoof 2021 LA/DF), `cross_dataset_eval_pgvib_part_b.ipynb`
   (WaveFake, CodecFake+), then `cross_dataset_eval_pgvib_aggregate.ipynb` in
   `notebooks/05_cross_dataset_eval/`.

The exact configuration and per-epoch history of each run are in the
matching `experiment_results/04_pgvib_development/` folder.

## Citation

```bibtex
@mastersthesis{kauffmann2026glada,
  author = {Kauffmann, Jakob},
  title  = {Targeted Bottlenecking of Self-Supervised Speech Layers for Generalizable Audio Deepfake Detection},
  school = {San Jos\'e State University},
  year   = {2026}
}
```
