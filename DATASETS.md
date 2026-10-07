# Datasets

Raw audio files are not included in this repository (≈75–90 GB total uncompressed).
The table below lists every dataset the report uses, the canonical download
location, and the on-disk path the training and evaluation code expects.

## Sources

| Dataset | Role in the report | Size | Download |
|---|---|---|---|
| ASVspoof 2019 LA | Labelled source domain (training) | 5 GB | https://datashare.ed.ac.uk/handle/10283/3336 |
| ASVspoof 2021 LA | Held-out evaluation (telephony codec) | 3 GB | https://zenodo.org/records/4837263 |
| ASVspoof 2021 DF | Held-out evaluation (codec-compressed deepfakes) | 40 GB | https://zenodo.org/records/4835108 |
| In-the-Wild (ITW) | Target domain (training, unlabelled) + held-out evaluation | 8 GB | https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa |
| WaveFake | Held-out evaluation (neural vocoders) | 16 GB | https://zenodo.org/records/5642694 |
| CodecFake+ | Held-out evaluation (neural-codec resynthesis) | 12 GB | https://github.com/RoyChen22/CodecFake |
| MUSAN | Background-noise augmentation (optional) | 11 GB | https://www.openslr.org/17/ |
| RIRS_NOISES | Room impulse responses (optional) | 1 GB | https://www.openslr.org/28/ |

Original references for each corpus:
- ASVspoof 2019 LA: Wang et al., *Computer Speech & Language* 64 (2020).
- ASVspoof 2021 LA / DF: Yamagishi et al., *ASVspoof 2021 Challenge Evaluation Plan*.
- ITW: Müller et al., *Interspeech* (2022).
- WaveFake: Frank & Schönherr, *NeurIPS* (2021).
- CodecFake+: Xie et al., *arXiv* 2406.07237 (2024).
- MUSAN: Snyder et al., *arXiv* 1510.08484 (2015).
- RIRS_NOISES: Ko et al., *ICASSP* (2017).

## Bonafide reference for WaveFake

WaveFake distributes only the synthesized (spoof) audio. To compute EER you
also need the original LJSpeech-1.1 recordings:

```
LJSpeech-1.1: https://keithito.com/LJ-Speech-Dataset/
```

`glada/datasets/wavefake.py` expects LJSpeech under
`wavefake/ljspeech_wav/`.

## Expected on-disk layout

The notebooks expect Colab-style absolute paths under `/content/data/`. If you
store the datasets elsewhere, edit the path constants in each notebook's setup
cell.

```
/content/data/
  asvspoof/
    LA/                                  # ASVspoof 2019 LA root
      ASVspoof2019_LA_train/flac/
      ASVspoof2019_LA_dev/flac/
      ASVspoof2019_LA_eval/flac/
      ASVspoof2019_LA_cm_protocols/      # ASVspoof2019.LA.cm.{train,dev,eval}.trn.txt
    LA_2021/flac/                        # ASVspoof 2021 LA eval
    DF_2021/flac/                        # ASVspoof 2021 DF eval
  itw/
    release_in_the_wild/                 # In-the-Wild release (meta.csv + audio/)
  wavefake/
    ljspeech_wav/                        # bonafide reference (LJSpeech-1.1)
    ljspeech_melgan/
    ljspeech_full_band_melgan/
    ljspeech_multi_band_melgan/
    ljspeech_parallel_wavegan/
    ljspeech_waveglow/
    ljspeech_hifiGAN/
  codecfakeplus/                         # CodecFake+ tree (real/, fake/<codec>/...)
  musan/                                 # MUSAN noise / music / speech (optional)
  RIRS_NOISES/simulated_rirs/            # OpenSLR-28 simulated RIRs (optional)
```

## Pre-processing

None. Every dataset is loaded as raw 16 kHz mono via `soundfile.read()`.
ASVspoof 2021 and ITW arrive ready to use; only the ASVspoof 2019 LA
protocol files (`ASVspoof2019.LA.cm.{train,dev,eval}.trn.txt`) are parsed at
load time. WaveFake is resampled from 22.05 kHz to 16 kHz on the fly by
`librosa.resample`; no offline conversion is needed.

## Which notebook uses which dataset

| Notebook | Datasets read |
|---|---|
| `01_baseline_glada/glada_experiment.ipynb` | ASVspoof 2019 LA |
| `01_baseline_glada/glada_evaluation.ipynb` | ASVspoof 2019 LA eval + ITW |
| `02_ablation_multibackbone/glada_ablation*.ipynb` | ASVspoof 2019 LA + ITW |
| `02_ablation_multibackbone/glada_multiseed.ipynb` | ASVspoof 2019 LA + ITW |
| `03_phase2_diagnosis/phase_b_sweep.ipynb` | ASVspoof 2019 LA + ITW |
| `03_phase2_diagnosis/phase_aug_validation.ipynb` | ASVspoof 2019 LA + ITW + MUSAN + RIR |
| `04_pgvib_development/phase_pgvib_*.ipynb` | ASVspoof 2019 LA + ITW |
| `05_cross_dataset_eval/cross_dataset_eval_pgvib_*.ipynb` | ITW, ASVspoof 2021 LA, ASVspoof 2021 DF, WaveFake, CodecFake+ |
| `06_analysis_and_figures/glada_acoustic_analysis.ipynb` | ASVspoof 2019 LA + ITW |
