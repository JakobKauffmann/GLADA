"""
WaveFake dataset for cross-domain evaluation.

WaveFake (Frank & Schönherr, NeurIPS 2021) is a benchmark focused on neural
vocoder artifacts.  Six vocoders generate speech from LJSpeech-1.1 mel
spectrograms, providing a controlled test of detector generalisation to
unseen synthesis systems.

Vocoder coverage (spoof subdirectories):
  ljspeech_melgan/            MelGAN        (Kumar et al., NeurIPS 2019)
  ljspeech_full_band_melgan/  Full-Band MelGAN (Yangyang et al.)
  ljspeech_multi_band_melgan/ Multi-Band MelGAN (Yang et al., ICASSP 2021)
  ljspeech_parallel_wavegan/  Parallel WaveGAN (Yamamoto et al., ICASSP 2020)
  ljspeech_waveglow/          WaveGlow      (Prenger et al., ICASSP 2019)
  ljspeech_hifiGAN/           HiFi-GAN      (Kong et al., NeurIPS 2020)

Bonafide reference (must be downloaded separately):
  LJSpeech-1.1/wavs/         Original LJSpeech recordings (Ito & Johnson, 2017)
  Download: https://keithito.com/LJ-Speech-Dataset/
  WaveFake only distributes the GENERATED (spoof) audio.  Without bonafide
  samples the EER is undefined (denominator ROC computation fails).

Dataset statistics (approximate):
  ~117,985 spoof samples across 6 vocoders (~19,664 per vocoder)
  ~13,100 bonafide samples from LJSpeech-1.1
  Audio: 22.05 kHz .wav; loader resamples to 16 kHz via librosa.

Cross-domain role in GLADA experiments:
  WaveFake is a HELD-OUT evaluation set — never used during training or
  adaptation.  It tests whether GLADA's representations generalise beyond
  the ASVspoof/ITW distribution to a different speaker (LJSpeech) and
  different (but overlapping) set of vocoders.  Good WaveFake EER alongside
  good ITW EER confirms that the model learns general synthesis artefact
  detectors, not ITW-specific features.

Reference:
  Frank & Schönherr, "WaveFake: A Data Set to Facilitate Audio Deepfake
  Detection", NeurIPS 2021 Datasets and Benchmarks Track.
  https://proceedings.neurips.cc/paper/2021/hash/3c8a49145944fed83378fba56af6e95e-Abstract-Datasets_and_Benchmarks.html
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset

from glada.datasets.audio import load_audio


class WaveFakeDataset(Dataset):
    """
    WaveFake benchmark dataset for cross-domain evaluation (evaluation only).

    Label assignment strategy:
      Walks all subdirectories recursively.  A file is labelled bonafide (1)
      if 'ljspeech-1.1' appears (case-insensitive) anywhere in its path.
      All other .wav files are labelled spoof (0).

      This path-based heuristic is robust to various directory layouts
      (e.g., 'generated_audio/LJSpeech-1.1/wavs/' or just 'LJSpeech-1.1/').

    Deduplication:
      rglob() may visit the same file multiple times when directories are
      nested.  The loader deduplicates by absolute path to avoid inflated
      sample counts that would corrupt EER computation.

    Args:
        root_dir (str): Root directory containing vocoder subdirectories.
            Typically the 'generated_audio/' folder, or one level above it.
        max_len (int): Target audio length in samples (default 64000 = 4s @ 16kHz).
    """

    def __init__(self, root_dir, max_len=64000):
        self.max_len = max_len
        self.samples = []
        root = Path(root_dir)

        # Walk all subdirectories recursively
        for subdir in sorted(root.rglob('*')):
            if not subdir.is_dir():
                continue
            # Bonafide: directory path contains 'ljspeech-1.1' (case-insensitive)
            is_bonafide = 'ljspeech-1.1' in str(subdir).lower()
            label = 1 if is_bonafide else 0
            for wav_path in subdir.glob('*.wav'):
                self.samples.append({'path': str(wav_path), 'label': label})

        # Deduplicate (rglob can traverse same files via nested dirs)
        seen = set()
        unique = []
        for s in self.samples:
            if s['path'] not in seen:
                seen.add(s['path'])
                unique.append(s)
        self.samples = unique

        n_bona = sum(1 for s in self.samples if s['label'] == 1)
        n_spoof = len(self.samples) - n_bona
        print(f"  WaveFake: {len(self.samples)} samples "
              f"({n_bona} bonafide, {n_spoof} spoof)")
        if n_bona == 0:
            print("  WARNING: No bonafide samples found! EER will be undefined.")
            print("  Download LJSpeech-1.1 from https://keithito.com/LJ-Speech-Dataset/")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, actual_len = load_audio(s['path'], self.max_len, deterministic=True)
        mask = torch.zeros(self.max_len, dtype=torch.long)
        mask[:actual_len] = 1
        return {
            'input_values': wav,
            'attention_mask': mask,
            'task_label': torch.tensor(s['label'], dtype=torch.long),
        }
