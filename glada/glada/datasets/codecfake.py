"""
CodecFake+ dataset for cross-domain evaluation.

CodecFake+ evaluates robustness to codec-based neural speech synthesis, a
rapidly growing attack vector driven by language-model-based TTS systems
(e.g., VALL-E, SoundStorm, UniAudio) that use discrete neural audio codecs
as intermediate representations.

Attack coverage:
  Neural audio codec TTS systems including:
    - Encodec-based synthesis  (Défossez et al., 2022)
    - SoundStream-based synthesis (Zeghidour et al., 2021)
    - VALL-E style LM decoders using codec tokens
  These systems produce deepfakes qualitatively different from the
  mel-vocoder attacks in WaveFake and ASVspoof — codec artefacts appear
  at different spectral frequencies.

Bonafide reference:
  Real speech from LibriSpeech and/or VCTK (dataset-version dependent).

Expected directory structure:
  real/           bonafide: original speech
  fake/           spoof: codec-generated speech
    encodec/
    soundstream/
    ...

The loader uses a flexible path-based heuristic for label assignment:
  any ancestor directory whose name matches _BONAFIDE_PATTERNS → bonafide (1)
  all other audio files → spoof (0)
This handles both the canonical structure above and alternative layouts.

Audio format: .wav or .flac, typically 16 kHz (no resampling needed).

Cross-domain role in GLADA experiments:
  CodecFake+ is a HELD-OUT evaluation set.  Codec-based deepfakes represent
  the frontier of attack technology (2023-2025); good performance here
  demonstrates that GLADA's domain-invariant SSL representations generalise
  to qualitatively new attack vectors not seen in any training set.

References:
  Xie et al., "CodecFake+: A Large-Scale Neural Codec-Based Deepfake Speech
  Dataset", 2024.  (Primary dataset reference)

  Défossez et al., "High Fidelity Neural Audio Compression",
  arXiv:2210.13438, 2022.  (Encodec — backbone of many CodecFake attacks)
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset

from glada.datasets.audio import load_audio


# Directory name patterns indicating bonafide samples (case-insensitive)
_BONAFIDE_PATTERNS = {'real', 'bonafide', 'genuine', 'original', 'clean'}


class CodecFakeDataset(Dataset):
    """
    CodecFake+ dataset for cross-domain evaluation (evaluation only).

    Label assignment strategy:
      Recursively walks root_dir for files matching `extensions`.
      For each file, all ancestor directory names (relative to root_dir) are
      checked against _BONAFIDE_PATTERNS (case-insensitive).  If ANY ancestor
      matches, the file is bonafide (1); otherwise spoof (0).

      This path-segment approach (rather than substring on the full path)
      prevents false matches — e.g., a file in 'fake/real_sounding/' would
      NOT be labelled bonafide because 'real_sounding' != 'real'.

    Warnings:
      - If no bonafide OR no spoof samples are found, a warning is printed
        (see __init__ below).  This usually indicates a structural mismatch
        between the expected and actual directory layout.
      - Check _BONAFIDE_PATTERNS if your dataset uses different directory
        names for genuine speech.

    Args:
        root_dir (str): Root directory of the CodecFake+ dataset.
        max_len (int): Target audio length in samples (default 64000 = 4s @ 16kHz).
        extensions (tuple): Audio file extensions to search for.
    """

    def __init__(self, root_dir, max_len=64000, extensions=('.wav', '.flac')):
        self.max_len = max_len
        self.samples = []
        root = Path(root_dir)

        # Walk directory tree and collect audio files with inferred labels
        for audio_path in sorted(root.rglob('*')):
            if not audio_path.is_file():
                continue
            if audio_path.suffix.lower() not in extensions:
                continue

            # Determine label from parent directory names in the path
            # Any ancestor directory matching a bonafide pattern → bonafide
            path_parts_lower = [p.lower() for p in audio_path.relative_to(root).parts]
            is_bonafide = any(
                part in _BONAFIDE_PATTERNS for part in path_parts_lower
            )
            label = 1 if is_bonafide else 0
            self.samples.append({'path': str(audio_path), 'label': label})

        n_bona = sum(1 for s in self.samples if s['label'] == 1)
        n_spoof = len(self.samples) - n_bona
        print(f"  CodecFake+: {len(self.samples)} samples "
              f"({n_bona} bonafide, {n_spoof} spoof)")
        if n_bona == 0:
            print("  WARNING: No bonafide samples found! Check directory structure.")
        if n_spoof == 0:
            print("  WARNING: No spoof samples found! Check directory structure.")

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
