"""
In-the-Wild (ITW) dataset for cross-domain evaluation and domain adaptation.

This dataset is the TARGET DOMAIN (d=1) in GLADA's unsupervised domain
adaptation (UDA) formulation.  During training, ITW labels are NOT used
(use_labels=False); only the domain membership signal (d=1) drives the GRL
domain discriminators.  At evaluation time (use_labels=True), the ground-truth
labels are used to compute cross-domain EER — GLADA's primary metric.

Dataset statistics:
  Total: 19,963 samples (~54 hours): 11,816 bonafide, 8,147 spoof
  Sources: YouTube-scraped real speech + TTS/VC deepfakes from numerous
           real-world tools (identities unknown, not from ASVspoof attack list)
  Format: mixed MP3/WAV at varying bit rates; loader resamples to 16 kHz

Domain shift characteristics (ASVspoof 2019 LA → ITW):
  - Recording conditions : studio/lab vs. in-the-wild (phone, outdoor, etc.)
  - Background noise     : silent vs. ambient
  - Speaker demographics : 20 fixed speakers vs. ~2000+ diverse speakers
  - Attack systems       : 6 known TTS/VC vs. unknown real-world tools
  - Codec compression    : uncompressed FLAC vs. MP3/YouTube compression
  - Channel effects      : clean vs. GSM/VOIP transmission artefacts

Müller et al. (2022) showed that end-to-end models trained on ASVspoof 2019
typically achieve 31-38% EER on ITW (near random chance), motivating domain
adaptation.  With SSL fine-tuning alone: 5-19% EER.  With adversarial DA
(e.g., IB-CAAN, GLADA): 4-7% EER.

Two usage modes:
  Mode 1 (Adaptation, use_labels=False):
    Target domain during training.  Returns task_label=0 (dummy) for all
    samples — actual labels are irrelevant; only domain index d=1 is used.

  Mode 2 (Evaluation, use_labels=True):
    Ground-truth labels returned for EER computation.  This is the primary
    cross-domain evaluation in all GLADA experiments.

Metadata format (meta.csv):
  Header row, then one sample per line.
  Columns detected dynamically: first column is filename, label column
  identified by header name ('label' or 'class').

Reference:
  Müller et al., "Does Audio Deepfake Detection Generalize?",
  Interspeech 2022.  https://doi.org/10.21437/Interspeech.2022-1085
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset

from glada.config import GLADAConfig
from glada.datasets.audio import load_audio

class InTheWildDataset(Dataset):
    """
    In-the-Wild (ITW) dataset: unlabeled target domain for GLADA's UDA or
    labeled held-out set for cross-domain EER evaluation.

    The distinction between the two modes (use_labels) is critical for the
    UDA training setup:
      - Training  (use_labels=False): domain label d=1 drives GRL only.
        The model MUST NOT see ITW task labels during training — this
        would violate the unsupervised DA assumption and constitute
        label leakage.
      - Evaluation (use_labels=True): ground-truth returned for EER.

    Note: labels ARE always read from the CSV regardless of use_labels, so
    that balanced sampling (equal bonafide/spoof) can be applied in the
    DataLoader.  They are only RETURNED when use_labels=True.
    """
    
    BONAFIDE = {'bona-fide', 'bonafide', 'real', '1', 'true'}

    def __init__(self, meta_csv_path, audio_dir, config: GLADAConfig, use_labels=False):
        """
        Args:
            meta_csv_path (str): Path to the metadata CSV file.
            audio_dir (str): Path to the directory containing audio files.
            config (GLADAConfig): Configuration object.
            use_labels (bool): If True, returns ground truth labels. If False, returns 0.
                               (Labels are always read for balanced sampling).
        """
        self.audio_dir = Path(audio_dir)
        self.max_len = config.max_audio_len
        self.use_labels = use_labels
        self.samples = []

        # Read metadata
        with open(meta_csv_path, 'r') as f:
            # Header check: file, label (or similar)
            header = next(f).strip().lower().split(',')
            label_idx = next((i for i, c in enumerate(header) if c in ['label', 'class']), -1)
            
            for line in f:
                p = line.strip().split(',')
                if not p:
                    continue
                    
                # Determine label
                if label_idx >= 0:
                    raw = p[label_idx].strip().lower()
                else:
                    # Fallback or assume spoof if unknown? 
                    # Notebook assumes 'spoof' if label_idx is missing/fails?
                    # Actually notebook code: `raw = p[label_idx]... if label_idx >= 0 else 'spoof'`
                    raw = 'spoof'
                
                label = 1 if raw in self.BONAFIDE else 0
                self.samples.append({'filename': p[0], 'label': label})

        print(f"  ITW: {len(self.samples)} samples "
              f"({'labeled' if use_labels else 'unlabeled'} returned)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Evaluation (use_labels=True): deterministic crop for reproducibility.
        # Training/adaptation (use_labels=False): random crop for augmentation.
        wav, actual_len = load_audio(
            str(self.audio_dir / s['filename']), self.max_len,
            deterministic=self.use_labels)

        mask = torch.zeros(self.max_len, dtype=torch.long)
        mask[:actual_len] = 1

        return {
            'input_values': wav,
            'attention_mask': mask,
            # Return real label if use_labels=True, else 0 (spoof/dummy)
            'task_label': torch.tensor(s['label'] if self.use_labels else 0, dtype=torch.long),
        }
