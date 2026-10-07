"""
glada/backbones — SSL backbone wrappers for GLADA.

This subpackage is intentionally thin: GLADA uses HuggingFace ``AutoModel``
directly (loaded in ``GLADAModel.__init__``) rather than wrapping each backbone
in a custom class.  The subpackage exists as a namespace placeholder so that
backbone-specific utilities (e.g. layer-count helpers, gradient-checkpoint
configuration) can be added here without modifying the main model file.

Supported backbones (configured via BACKBONE_REGISTRY in glada/config.py):
  - microsoft/wavlm-large         (WavLM-Large, 24 layers, dim=1024)
  - facebook/wav2vec2-xls-r-300m  (XLS-R 300M, 24 layers, dim=1024)
  - facebook/data2vec-audio-large-960h  (Data2Vec-Large, 24 layers, dim=1024)
  - facebook/wav2vec2-large-robust (Wav2Vec2-Robust, 24 layers, dim=1024)
  - microsoft/wavlm-base-plus      (WavLM-Base+, 12 layers, dim=768)
  - facebook/wav2vec2-xls-r-1b     (XLS-R 1B, 48 layers, dim=1280)

All backbones are loaded with ``output_hidden_states=True`` so that GLADA can
access every intermediate layer for hierarchical pooling.  Gradient checkpointing
is enabled during Phase 2 to reduce GPU memory when fine-tuning 24+ layer models.
"""
