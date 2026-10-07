"""
ASVspoof 2019 LA dataset classes for training and evaluation.

This dataset is the SOURCE DOMAIN (d=0) in GLADA's unsupervised domain
adaptation (UDA) formulation.  The domain discriminator in GLADA attempts to
distinguish source (d=0) from target (d=1, ITW) samples; the GRL forces the
backbone to produce domain-invariant representations.

Dataset statistics (LA partition):
  Training   : 2,580 samples (2,580 bonafide + 22,800 spoof),  20 speakers
  Development: 2,548 samples ( 2,548 bonafide +  6,300 spoof),  20 speakers
  Evaluation : 7,355 samples ( 7,355 bonafide + 63,882 spoof),  67 speakers

Attack coverage:
  Training/Dev: A01-A06 (6 TTS/VC systems — known attacks)
  Evaluation  : A07-A19 (13 systems — includes 7 unknown/unseen attacks)
  The gap between known (A01-A06) and unknown (A07-A19) attacks is why
  cross-domain generalization research uses ITW as the primary benchmark.

Protocol format (whitespace-separated):
  Column 0: Speaker ID       (e.g. LA_0079)
  Column 1: Audio filename   (e.g. LA_T_1138215, without extension)
  Column 2: Environment      ('-' for LA)
  Column 3: Attack system ID (e.g. A01, '-' for bonafide)
  Column 4: Label            ('bonafide' or 'spoof')

Audio files: 16 kHz FLAC, variable length (mean ~3.4 s), loaded with
soundfile.read() and padded/truncated to 4 s (64,000 samples) by load_audio().

Reference:
  Wang et al., "ASVspoof 2019: A Large-Scale Public Database of Synthesized,
  Converted and Replayed Speech", Computer Speech & Language, 2020.
  https://doi.org/10.1016/j.csl.2020.101114
"""

import random
from pathlib import Path
import torch
from torch.utils.data import Dataset

from glada.config import GLADAConfig
from glada.datasets.audio import load_audio
from glada.augmentation.rawboost import RawBoost

class ASVspoofDatasetWithSpeaker(Dataset):
    """
    ASVspoof 2019 LA training dataset with per-sample speaker labels.

    Returns both the spoof/bonafide task label AND the speaker identity label,
    which are used by two separate objectives in GLADA:

      1. Task classifier (L_spoof): binary spoof detection (BCE loss).
         label=1 bonafide, label=0 spoof.

      2. Speaker adversary (L_speaker, GRL): the model is PENALISED for
         encoding speaker identity, promoting speaker-invariant representations.
         This is theoretically motivated by the desideratum that a good
         anti-spoofing detector should fire on synthesis artefacts, not on
         speaker-specific acoustic properties.

    Speaker IDs are mapped to consecutive integers at load time.  The number of
    unique speakers (num_speakers) must be passed to the speaker adversary head
    so it can allocate the right number of output classes.

    Supports two augmentation modes (augmenter argument):
      - RawBoost class (legacy): external Bernoulli gate via rawboost_prob.
      - CombinedAugmenter instance: manages its own per-stage probabilities
        (RawBoost + MUSAN + RIR).  Detected by presence of 'musan_augmenter'
        attribute.
      - None: no augmentation (eval/test mode).
    """

    def __init__(self, protocol_path, audio_dir, config: GLADAConfig, augmenter=None):
        """
        Args:
            protocol_path (str): Path to the protocol text file.
            audio_dir (str): Path to the directory containing audio files.
            config (GLADAConfig): Configuration object.
            augmenter: One of:
              - RawBoost class (legacy): applies RawBoost.augment() with rawboost_prob
              - CombinedAugmenter instance: calls augmenter.augment() (handles all probs internally)
              - None: no augmentation
        """
        self.audio_dir = Path(audio_dir)
        self.max_len = config.max_audio_len
        self.rawboost_prob = config.rawboost_prob
        self.augmenter = augmenter
        # CombinedAugmenter instances manage their own probability logic
        self._is_combined = hasattr(augmenter, 'musan_augmenter')
        self.samples = []
        self.speaker_to_id = {}

        # Parse protocol file
        # Format: LA_0079 LA_T_1138215 - - bonafide
        with open(protocol_path, 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                speaker = p[0]
                if speaker not in self.speaker_to_id:
                    self.speaker_to_id[speaker] = len(self.speaker_to_id)
                
                self.samples.append({
                    'filename': p[1],
                    'speaker': speaker,
                    'speaker_id': self.speaker_to_id[speaker],
                    'label': 1 if p[4] == 'bonafide' else 0,
                })

        self.num_speakers = len(self.speaker_to_id)
        print(f"  ASVspoof: {len(self.samples)} samples, "
              f"{self.num_speakers} speakers, rawboost={self.rawboost_prob}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, actual_len = load_audio(
            str(self.audio_dir / f"{s['filename']}.flac"), self.max_len)

        # Apply augmentation if enabled.
        # CombinedAugmenter handles its own per-stage probabilities internally.
        # Legacy RawBoost class uses rawboost_prob from config.
        if self.augmenter:
            if self._is_combined:
                wav_np = wav.numpy()
                wav_np = self.augmenter.augment(wav_np)
                wav = torch.from_numpy(wav_np)
            elif self.rawboost_prob > 0 and random.random() < self.rawboost_prob:
                wav_np = wav.numpy()
                wav_np = self.augmenter.augment(wav_np)
                wav = torch.from_numpy(wav_np)

        # Build proper attention mask: 1 for real samples, 0 for zero-padded.
        # Ensures padded regions are excluded from ASP softmax downstream.
        mask = torch.zeros(self.max_len, dtype=torch.long)
        mask[:actual_len] = 1

        return {
            'input_values': wav,
            'attention_mask': mask,
            'task_label': torch.tensor(s['label'], dtype=torch.long),
            'speaker_id': torch.tensor(s['speaker_id'], dtype=torch.long),
        }


class ASVspoofEvalDataset(Dataset):
    """
    ASVspoof 2019 LA evaluation dataset (no augmentation, no speaker labels).

    Used to compute in-domain EER on the official ASVspoof 2019 LA evaluation
    partition (7,355 bonafide + 63,882 spoof, 13 attack types A01-A19).

    In-domain EER is a secondary metric in GLADA experiments; the primary
    metric is cross-domain ITW EER, which measures generalisation beyond the
    training distribution.  A model that achieves low ASVspoof 2019 eval EER
    but high ITW EER has overfit to the training attack types.

    No augmentation is applied during evaluation (augmenter is always None).
    Speaker labels are not returned because evaluation requires only score
    ranking, not speaker-level analysis.
    """

    def __init__(self, protocol_path, audio_dir, config: GLADAConfig):
        """
        Args:
            protocol_path (str): Path to the protocol text file.
            audio_dir (str): Path to the directory containing audio files.
            config (GLADAConfig): Configuration object.
        """
        self.audio_dir = Path(audio_dir)
        self.max_len = config.max_audio_len
        self.samples = []

        with open(protocol_path, 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    self.samples.append({
                        'filename': p[1],
                        'label': 1 if p[4] == 'bonafide' else 0,
                    })

        print(f"  ASVspoof Eval: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, actual_len = load_audio(
            str(self.audio_dir / f"{s['filename']}.flac"), self.max_len,
            deterministic=True)

        mask = torch.zeros(self.max_len, dtype=torch.long)
        mask[:actual_len] = 1

        return {
            'input_values': wav,
            'attention_mask': mask,
            'task_label': torch.tensor(s['label'], dtype=torch.long),
        }
