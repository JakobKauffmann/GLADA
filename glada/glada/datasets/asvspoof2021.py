"""
ASVspoof 2021 LA and DF evaluation dataset classes.

ASVspoof 2021 extends 2019 with two harder cross-condition evaluation tracks:

  LA (Logical Access):
    Same TTS/VC attack systems as 2019, but audio transmitted through
    telephone and VoIP channels (GSM, G.711, G.722, etc.).  Channel effects
    add additive noise, bandwidth limitation (300-3400 Hz for PSTN), and
    codec artefacts, making detection harder.  Models that rely on high-freq
    artefacts from neural vocoders suffer a large accuracy drop vs. 2019.

  DF (Deepfake):
    Diverse set of TTS/VC systems whose audio has been further processed with
    lossy codecs (MP3, AAC, Opus) and other post-processing.  Designed to
    mimic deepfake audio shared on social media platforms.

Dataset statistics (approximate, official evaluation partition):
  LA track:  ~148,176 total trials (14,816 bonafide + 133,360 spoof)
  DF track:  ~611,829 total trials ( 22,617 bonafide + 589,212 spoof)
  Audio: 16 kHz FLAC (or WAV in some re-distributed versions).

Both tracks are used in GLADA experiments as HELD-OUT evaluation sets (never
seen during training or adaptation).  They measure whether GLADA's in-domain
adaptation (ASVspoof 2019 → ITW) also improves robustness to codec/channel
conditions that neither domain contains — testing general acoustic invariance.

Protocol formats (whitespace-separated):
  LA: SPEAKER_ID  TRIAL_KEY  -  ATTACK_TYPE  LABEL(bonafide/spoof)
  DF: SPEAKER_ID  TRIAL_KEY  CODEC  ATTACK_TYPE  LABEL  ... (variable columns)
  Both: column 1 = audio filename stem; 'bonafide' anywhere in line = genuine.

Key availability:
  - 2021 LA keys were released SEPARATELY from the audio after the challenge.
    If the dataset reports 0 samples, the keys file is likely missing or at a
    different path.  The keys are NOT bundled with the audio download.
  - 2021 DF keys are included in the evaluation package.

References:
  Yamagishi et al., "ASVspoof 2021: Accelerating Progress in Spoofed and
  Deepfake Speech Detection", ASVspoof 2021 Workshop, 2021.

  Delgado et al., "ASVspoof 2021: Automatic Speaker Verification Spoofing
  and Countermeasures Challenge Evaluation Plan", 2021.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset

from glada.datasets.audio import load_audio


class ASVspoof2021Dataset(Dataset):
    """
    ASVspoof 2021 LA or DF evaluation dataset (evaluation only, no training).

    A single class handles both LA and DF because their protocol files share
    the same key structure (column 1 = filename stem, 'bonafide' in line =
    genuine).  The caller specifies which subset via keys_path and audio_dir.

    Label assignment:
      label = 1  (bonafide)  if 'bonafide' appears anywhere in the line
      label = 0  (spoof)     otherwise
    This substring approach works for both LA (exact 5-column format) and DF
    (variable column count with different ordering across releases).

    Audio resolution:
      Tries <filename>.flac first, then falls back to <filename>.wav.
      This handles both the original distribution (.flac) and re-packaged
      versions (.wav) without requiring the caller to know the extension.

    Args:
        keys_path (str): Path to the protocol/keys text file.
        audio_dir (str): Path to the directory containing audio files (.flac/.wav).
        max_len (int): Target audio length in samples (default 64000 = 4s @ 16kHz).
    """

    def __init__(self, keys_path, audio_dir, max_len=64000):
        self.audio_dir = Path(audio_dir)
        self.max_len = max_len
        self.samples = []

        with open(keys_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                # Column 1 is the trial key (filename stem)
                filename = parts[1]
                # Label: 'bonafide' appears in the line for genuine samples
                label = 1 if 'bonafide' in line else 0
                self.samples.append({'filename': filename, 'label': label})

        n_bona = sum(1 for s in self.samples if s['label'] == 1)
        print(f"  ASVspoof 2021: {len(self.samples)} samples "
              f"({n_bona} bonafide, {len(self.samples) - n_bona} spoof)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Try .flac first, fall back to .wav
        audio_path = self.audio_dir / f"{s['filename']}.flac"
        if not audio_path.exists():
            audio_path = self.audio_dir / f"{s['filename']}.wav"
        wav, actual_len = load_audio(str(audio_path), self.max_len, deterministic=True)
        mask = torch.zeros(self.max_len, dtype=torch.long)
        mask[:actual_len] = 1
        return {
            'input_values': wav,
            'attention_mask': mask,
            'task_label': torch.tensor(s['label'], dtype=torch.long),
        }
