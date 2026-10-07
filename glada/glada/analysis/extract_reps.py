"""
Per-group representation extractor for GLADA / PerGroupVIB checkpoints.

Purpose
-------
For the acoustic-feature analysis (Subtask C of the PGVIB experiments) we need
the K per-group pooled representations *as the trained model produces them*,
not derived embeddings. This module loads a saved checkpoint, runs a forward
pass on a list of audio files, and dumps the per-group tensors to disk for
downstream linear probing, t-SNE visualisation, and CKA-vs-acoustic analysis.

What is saved
-------------
A single ``torch.save``-d ``.pt`` archive with::

    {
        "group_reps":  Tensor (B, K, F),   # F = 2*hidden_dim for GLADAModel
                                            #   or bottleneck_dim for PerGroupVIBModel
        "global_z":    Tensor (B, bottleneck_dim),  # post-VIB bottleneck z
        "audio_files": list[str],          # absolute paths in the same order
        "model_class": "GLADAModel" | "PerGroupVIBModel",
        "num_groups":  int K,
        "feature_dim": int F (per group),
        "config":      dict (asdict of GLADAConfig if present in checkpoint),
    }

Why this representation choice
------------------------------
- For ``GLADAModel`` the ``group_outputs`` returned by ``HierarchicalAttention
  Pooling`` are the per-group ASP statistics ``g_k = [mu_k ; sigma_k] in R^{2D}``
  (Okabe et al., Odyssey 2018). These are exactly the inputs the GLADA per-group
  domain discriminators see, so probing them tells us what each discriminator
  could learn from.
- For ``PerGroupVIBModel`` (see ``Coding/notebooks/april13/_phase_common.py``,
  lines 140--280), the ``group_outputs`` field of the forward dict is already
  the *post-VIB* compressed ``z_k in R^{bottleneck_dim}`` (one per group).
  Probing these matches the PGVIB ablation's actual information channel.

Audio loading
-------------
Always uses ``soundfile.read()`` via ``glada.datasets.audio.load_audio``
(never ``torchaudio.load``). Evaluation-style deterministic
truncation (first 4 s) is used for reproducibility.

References
----------
Okabe et al., "Attentive Statistics Pooling for Deep Speaker Embedding",
Odyssey 2018 (ASP equation for g_k).
Alemi et al., "Deep Variational Information Bottleneck", ICLR 2017 (VIB z).
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import List, Optional, Sequence

import torch

from glada.config import GLADAConfig
from glada.datasets.audio import load_audio
from glada.models.glada_model import GLADAModel


def _build_attention_mask(actual_lengths: Sequence[int], max_len: int,
                          device: torch.device) -> torch.Tensor:
    """Construct a sample-level attention mask (B, T_raw).

    1 = real audio sample, 0 = zero-padding. Matches the convention expected
    by ``GLADAModel.forward``: it converts this to a frame-level mask using
    the WavLM/Wav2Vec2 CNN stride of 320.
    """
    B = len(actual_lengths)
    mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, L in enumerate(actual_lengths):
        # Clamp at max_len in case loader returned a length equal to it.
        mask[i, :min(L, max_len)] = 1
    return mask


def extract_group_reps(
    checkpoint_path: str,
    audio_files: List[str],
    output_path: str,
    model_class: str = "PerGroupVIBModel",
    config: Optional[GLADAConfig] = None,
    batch_size: int = 16,
    device: Optional[str] = None,
    max_len: int = 64000,
) -> dict:
    """Run a checkpoint's forward pass on ``audio_files`` and dump per-group reps.

    Args:
        checkpoint_path: Absolute path to a ``.pt`` checkpoint produced by the
            GLADA trainer (contains ``model`` state_dict and ``config`` dict).
        audio_files: List of absolute audio paths to score.
        output_path: Where to ``torch.save`` the result archive (``.pt``).
        model_class: ``"GLADAModel"`` (uses pre-VIB ASP outputs ``g_k``) or
            ``"PerGroupVIBModel"`` (uses post-VIB ``z_k``). The PGVIB class
            is constructed dynamically here using the same definition as
            ``Coding/notebooks/april13/_phase_common.py`` so this module
            stays Colab-portable without importing the notebook builder.
        config: Optional ``GLADAConfig`` override. If None, the function
            rebuilds the config from the checkpoint's ``config`` dict.
        batch_size: Inference batch size. Larger values are fine here because
            no gradients are computed; defaults to 16 to fit T4 GPUs safely.
        device: ``"cuda"`` / ``"cpu"`` / ``None`` (auto-pick CUDA if available).
        max_len: Audio length in samples. 64000 = 4 s @ 16 kHz (GLADA default).

    Returns:
        The same dict that is written to ``output_path``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load the checkpoint and rebuild config -----------------------------
    # weights_only=False because GLADA checkpoints contain the config dict.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if config is None:
        cfg_dict = ckpt.get("config", None)
        if cfg_dict is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no 'config' key and no "
                f"explicit config was passed."
            )
        config = GLADAConfig(**{k: v for k, v in cfg_dict.items()
                                if k in GLADAConfig.__dataclass_fields__})

    # --- Build the model. PerGroupVIBModel is defined inline because the
    #     class lives in the notebook builder and we don't want to import a
    #     notebook from a package module. The definition mirrors the one in
    #     `_phase_common.py` exactly. -----------------------------------------
    if model_class == "GLADAModel":
        model = GLADAModel(config)
    elif model_class == "PerGroupVIBModel":
        model = _build_pergroup_vib_model(config)
    else:
        raise ValueError(f"Unknown model_class={model_class!r}")

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # --- Forward pass in batches; collect per-group reps + global z ---------
    group_reps_all: List[torch.Tensor] = []
    z_all: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(audio_files), batch_size):
            chunk = audio_files[start:start + batch_size]
            wavs, lens = [], []
            for fp in chunk:
                # deterministic=True -> first 4s; matches evaluator behaviour.
                w, L = load_audio(fp, max_len=max_len, deterministic=True)
                wavs.append(w)
                lens.append(L)
            input_values = torch.stack(wavs).to(device)
            attention_mask = _build_attention_mask(lens, max_len, device)
            out = model(input_values, attention_mask=attention_mask)
            # group_outputs: list of K tensors (B, F).  F = 2D for GLADAModel,
            # F = bottleneck_dim for PerGroupVIBModel. Stack to (B, K, F).
            g = torch.stack(out["group_outputs"], dim=1).cpu()
            group_reps_all.append(g)
            z_all.append(out["features"].cpu())

    group_reps = torch.cat(group_reps_all, dim=0)  # (N, K, F)
    global_z = torch.cat(z_all, dim=0)             # (N, bottleneck_dim)

    payload = {
        "group_reps":  group_reps,
        "global_z":    global_z,
        "audio_files": [os.path.abspath(p) for p in audio_files],
        "model_class": model_class,
        "num_groups":  int(group_reps.shape[1]),
        "feature_dim": int(group_reps.shape[2]),
        "config":      asdict(config),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                exist_ok=True)
    torch.save(payload, output_path)
    return payload


def _build_pergroup_vib_model(config: GLADAConfig):
    """Construct ``PerGroupVIBModel`` from the package.

    Previously this module re-declared the class inline to avoid a notebook
    dependency.  The canonical class now lives in
    ``glada.models.glada_model`` (promoted from the april13 builder injection)
    so we just import it.  ``vib_disabled_groups`` / ``vib_betas`` default to
    no-op for analysis use, matching the prior inline body's behavior.
    """
    from glada.models.glada_model import PerGroupVIBModel
    return PerGroupVIBModel(config)
