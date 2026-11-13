"""
glada/config.py — Central configuration for GLADA experiments.

Defines the backbone registry (mapping string keys to HuggingFace model IDs
and architecture parameters) and GLADAConfig, a dataclass that drives all
training decisions. A single GLADAConfig object is created at experiment start
and threaded through model construction, dataset loading, and training.

The total training loss is::

    L = L_spoof
      + alpha  * L_GLADA      # per-group domain adversarial loss
      + beta   * L_KL         # VIB KL divergence (information bottleneck)
      + gamma  * L_speaker    # speaker adversarial loss (GRL)
      + delta  * L_CKA        # CKA diversity regularisation

Default loss weights (validated empirically):
  alpha (domain_loss_weight)   = 0.10   — >0.1 destabilises training (tested at 0.5)
  beta  (vib_beta)             = 1e-3   — Alemi et al. (ICLR 2017) recommended starting point
  gamma (speaker_loss_weight)  = 0.01   — linearly scheduled up from 0 over speaker_grl_warmup_epochs
  delta (cka_loss_weight)      = 0.01   — small to prevent diversity loss dominating spoof loss

Key design choices preserved from glada_experiment.ipynb:
  - Ablation switches (use_vib, use_speaker_adversary, …) let us toggle every
    component independently for the ablation study.
  - grl_warmup_epochs prevents early domain-discriminator collapse
    (Ganin et al., JMLR 2016).
  - domain_loss_weight is hard-capped at 0.1 — tested higher (0.5) and it
    destabilised training.
  - kl_clamp_max guards against VIB KL blow-up observed in early runs
    (observed value: 133,598 without the clamp).
  - num_workers=0 always (Colab multiprocessing issues).
"""

import os
import random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict

# numpy and torch are imported lazily inside set_seed() so that config.py
# can be imported in pure-Python environments (unit tests, CI, local editors)
# without requiring the full Colab/GPU stack.


# =============================================================================
# Backbone Registry
# Maps short backbone keys to HuggingFace model IDs and architecture params.
#
# num_groups: how many SSL layer groups GLADA trains separate per-group domain
#   discriminators for (a separate GRL + domain discriminator at each SSL
#   layer group).
#   Precedent in vision: MLADA (Fang et al., Neurocomputing 2021).
#   Precedent in NLP: DAA (Ngo Trung et al., ACL-IJCNLP 2021 Findings).
#   First for audio deepfake detection: GLADA (this work).
#
# num_layers: number of Transformer encoder layers (not counting CNN embedding).
# dim: hidden state dimension.
# =============================================================================
BACKBONE_REGISTRY: Dict[str, Dict] = {
    "wavlm-large": {
        "hf_id": "microsoft/wavlm-large",
        "dim": 1024, "num_layers": 24, "num_groups": 4,
    },
    "xlsr-300m": {
        "hf_id": "facebook/wav2vec2-xls-r-300m",
        "dim": 1024, "num_layers": 24, "num_groups": 4,
    },
    "data2vec-large": {
        "hf_id": "facebook/data2vec-audio-large-960h",
        "dim": 1024, "num_layers": 24, "num_groups": 4,
    },
    "wav2vec2-robust": {
        "hf_id": "facebook/wav2vec2-large-robust",
        "dim": 1024, "num_layers": 24, "num_groups": 4,
    },
    "wavlm-base-plus": {
        "hf_id": "microsoft/wavlm-base-plus",
        "dim": 768, "num_layers": 12, "num_groups": 3,
    },
    "xlsr-1b": {
        "hf_id": "facebook/wav2vec2-xls-r-1b",
        "dim": 1280, "num_layers": 48, "num_groups": 6,
    },
}


# =============================================================================
# GLADAConfig
# Central dataclass: all hyperparameters, ablation switches, and paths.
# Instantiate once and pass everywhere. Serialise with dataclasses.asdict().
# =============================================================================
@dataclass
class GLADAConfig:
    """
    Central configuration for a GLADA experiment run.

    Instantiate once and pass to model construction, dataset loading, and training.
    Serialise to dict with ``dataclasses.asdict(config)`` for logging/checkpointing.

    Fields are grouped by concern:

    **Experiment**
        ``experiment_name``, ``run_id``, ``seed`` — used for directory naming and
        reproducibility. ``run_id`` defaults to a timestamp so parallel sweeps
        don't collide.

    **Backbone**
        ``backbone`` — short key into BACKBONE_REGISTRY (e.g. ``"wavlm-large"``).
        ``__post_init__`` fills ``hf_model_id``, ``hidden_dim``, ``num_layers``,
        ``num_groups``, and ``total_hidden_states`` from the registry entry.

    **Architecture**
        ``bottleneck_dim`` — VIB latent dimension (256 balances expressiveness
        and compression; halving to 128 slightly hurt EER in informal tests).
        ``vib_beta`` (β) — weight on the KL term in the VIB objective
        (Alemi et al., ICLR 2017: ``L = -I(Z;Y) + β·KL[q(z|x) || p(z)]``).
        ``num_speakers`` — number of speaker classes in the training set
        (107 for ASVspoof 2019 LA); used for speaker adversary head size.

    **Loss weights**
        ``domain_loss_weight`` (α), ``speaker_loss_weight`` (γ),
        ``cka_loss_weight`` (δ) — see module docstring for the full loss equation
        and empirically validated defaults.

    **GRL scheduling**
        ``grl_warmup_epochs`` / ``speaker_grl_warmup_epochs`` — epochs over which
        the GRL lambda ramps from 0 to 1 (Ganin et al., JMLR 2016).  Starting
        with lambda=0 prevents the domain discriminator from destabilising the
        backbone before the task classifier has converged.

    **Ablation switches**
        One boolean per GLADA component.  Setting all to True gives the full model;
        selectively setting them to False reproduces rows of the ablation table.

    **VIB stability**
        ``kl_clamp_max`` — raw KL is clamped to this value before multiplying by
        ``vib_beta``.  Without the clamp, an early run produced KL = 133,598,
        which caused the VIB term to dominate and collapse the representation.

    **Training**
        Gradient accumulation (``grad_accum_steps``) is computed automatically in
        ``__post_init__`` so that the effective batch size matches
        ``effective_batch_size``.  The scheduler must be constructed with
        optimizer-steps (not raw batches); see ``optimizer/scheduler.py``.

    **Data**
        ``num_workers=0`` is mandatory on Colab (fork-based multiprocessing breaks
        inside the Colab sandbox).  Do not change this default.

    **Paths**
        ``drive_log_dir`` — root directory on Google Drive for checkpoints and logs.
        ``get_run_dir()`` returns the full path for a specific run.
    """

    # --- Experiment ---
    experiment_name: str = "glada"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    seed: int = 42

    # --- Backbone ---
    backbone: str = "wavlm-large"  # key into BACKBONE_REGISTRY

    # --- Architecture ---
    bottleneck_dim: int = 256
    vib_beta: float = 1e-2          # KL weight (Alemi et al., ICLR 2017: "Deep Variational IB")
                                     # Effective KL contribution = vib_beta * kl_clamp_max;
                                     # 1e-2 * 10 = 0.1, comparable to spoof loss magnitude.
                                     # Previous values (1e-3, 1e-4) made KL penalty negligible,
                                     # causing the posterior to drift and KL to explode to 31,000+.
    num_speakers: int = 107         # ASVspoof 2019 LA training set speaker count

    # --- Loss weights ---
    # See module-level docstring for the full loss equation.
    domain_loss_weight: float = 0.1  # alpha; >0.1 destabilises training (tested at 0.5)
    speaker_loss_weight: float = 0.1 # gamma; fixed loss coefficient (like domain_loss_weight).
                                     # GRL reversal strength ramps separately 0→1 over
                                     # speaker_grl_warmup_epochs — see trainer._get_grl_alphas()
    cka_loss_weight: float = 0.01    # delta; small to avoid CKA term overshadowing spoof loss
                                     # (Kornblith et al., NeurIPS 2019)

    # --- GRL scheduling (Ganin et al., JMLR 2016) ---
    # GRL lambda ramps from 0 → 1 over warmup_epochs.  Starting at 0 lets the
    # task classifier stabilise before the domain discriminator is enforced;
    # skipping the ramp (lambda=1 from epoch 0) collapsed training in early runs.
    grl_warmup_epochs: int = 5       # domain GRL warmup; <5 causes instability
    speaker_grl_warmup_epochs: int = 10  # speaker GRL warmup — slower because speaker
                                          # identity is a finer-grained signal than domain

    # --- Phase 2 adversarial ramp (gradient shock prevention) ---
    # When backbone unfreezes at Phase 2 start, domain_alpha is already 1.0 from
    # Phase 1 warmup.  Full-strength reversed gradients from 5 adversarial signals
    # (4 GLADA discriminators + 1 speaker adversary) flow through freshly-trainable
    # layers with zero optimizer momentum, causing ITW EER to degrade from ~13.5%
    # to 20%+.  Fix: reset adversarial alphas at Phase 2 start and ramp back.
    # Analogous to IB-CAAN's GRL coefficient scheduling (Huang et al., 2025).
    phase2_adv_warmup_epochs: int = 3    # Ramp domain/speaker alphas from ramp_start
                                          # to full strength over this many Phase 2 epochs.
                                          # 0 = disabled (legacy behaviour, no ramp).
    phase2_adv_ramp_start: float = 0.0   # Starting alpha fraction at Phase 2 onset.
                                          # 0.0 = full reset; 0.5 = start at half strength.
    phase2_cka_ramp: bool = True         # Also ramp CKA weight during Phase 2 warmup.
                                          # CKA diversity pressure on group outputs can
                                          # conflict with backbone adaptation early on.

    # --- Confidence-aware adversarial weighting (post-ICTAI extension) ---
    # When True, the per-sample GLADA cross-entropy is weighted by the
    # discriminator's confidence margin |p_disc - 0.5| before being averaged
    # across the batch.  Samples the discriminator is uncertain about (p close
    # to 0.5) contribute LESS to the adversarial loss, so the GRL signal
    # passing back to the backbone is dominated by samples on which the
    # discriminator is confident.  This is the IB-CAAN technique (Huang et al.,
    # 2025; arxiv 2509.23618, equation 7) extended from a single domain disc
    # to GLADA's per-group disc bank.
    #
    # Default: False.  Preserves all existing pre-ICTAI results bit-for-bit.
    # When False, the loss aggregation in glada_trainer.py:_compute_glada_loss
    # falls back to the original mean of per-group cross-entropies.
    confidence_aware_adv_weight: bool = False

    # --- Ablation switches ---
    # Each switch independently toggles one GLADA component so the ablation table
    # (Table 4 in the thesis) can isolate each contribution.  All True = full model.
    # Setting use_glada=False while use_global_domain_disc=True gives the flat-DA baseline.
    use_vib: bool = True              # Toggle VIB; if False, deterministic linear projection is used instead
    use_speaker_adversary: bool = True  # Toggle speaker GRL (Meng et al., Interspeech 2021)
    use_global_domain_disc: bool = False  # Single domain disc on the pooled embedding z
                                          # (flat-DA baseline; not used in the full GLADA model)
    use_cka_loss: bool = True         # Toggle CKA diversity loss (Kornblith/Tran et al.)
    use_hierarchical: bool = True     # Grouped hierarchical attention; flat weighted sum if False
    use_rawboost: bool = True         # Toggle RawBoost augmentation (Tak et al., ICASSP 2022)
    use_glada: bool = True            # Per-group domain discriminators (the main GLADA component)

    # --- GLADA architecture ---
    glada_disc_hidden: int = 128      # Hidden dim for per-group domain discriminators
    num_groups_override: int = 0      # Override backbone default num_groups; 0 = use registry default

    # === VIB KL stability ===
    # Without this clamp, KL loss exploded to 133,598 in an early run, causing
    # the VIB term to dominate and collapse the representation.  Clamped to
    # kl_clamp_max before multiplying by vib_beta.
    kl_clamp_max: float = 10.0

    # --- KL Annealing (Bowman et al., ACL 2016) ---
    # Linearly ramp vib_beta from 0 to its target value over kl_anneal_epochs.
    # This lets the classifier converge before VIB compression kicks in,
    # preventing posterior collapse (beta too high too early) and uncontrolled
    # drift (beta too low to matter). IB-CAAN (Huang et al., 2025) used a
    # similar warm-start for VIB stability.
    kl_anneal_epochs: int = 5         # Ramp beta from 0 -> vib_beta over this many epochs
                                       # Default matches phase1_epochs so full beta arrives
                                       # exactly when the backbone unfreezes

    # --- Training ---
    # Two-phase training: Phase 1 warms up the randomly-initialised heads with the
    # backbone frozen, then Phase 2 fine-tunes end-to-end with LLRD.
    phase1_epochs: int = 5           # Frozen backbone, train heads only
    phase2_epochs: int = 45          # End-to-end fine-tuning with LLRD
    batch_size: int = 24             # Per-domain batch size; T4=24, A100-80GB=48+
                                     # Actual batch presented to the model is 2x (source + target)
    effective_batch_size: int = 384  # Target effective batch; grad_accum_steps computed from this
    grad_accum_steps: int = 1        # Overwritten in __post_init__ — do not set manually
    head_lr: float = 1e-3            # Heads are randomly initialised; need a large LR
    backbone_lr: float = 5e-6        # Phase 2 peak LR for the deepest backbone layer (with LLRD)
                                     # Reduced from 1e-5: with VIB KL properly weighted, the
                                     # backbone needs gentler updates to avoid distorting features.
                                     # Earlier layers get exponentially smaller LRs via llrd_decay.
    llrd_decay: float = 0.90         # Per-layer LR decay factor; lr_i = backbone_lr * 0.90^(L-i)
    freeze_bottom_n_layers: int = 0  # Keep bottom N transformer layers frozen in Phase 2.
                                     # These layers encode general acoustic features (phonemes,
                                     # pitch) that transfer well and don't need task adaptation.
                                     # 0 = unfreeze all (original behavior).
                                     # Recommended: 12 for WavLM-Large (freeze bottom half).
                                     # (Clark et al., EMNLP 2020; Howard & Ruder, ACL 2018)
    weight_decay: float = 0.01       # AdamW weight decay; small regularisation, not aggressive
    gradient_clip: float = 1.0       # Max gradient norm (L2); prevents exploding gradients from GRL
    patience: int = 15               # Early stopping patience (epochs without EER improvement)
    use_amp: bool = True
    amp_dtype: str = "bfloat16"      # bfloat16 for A100 (no overflow risk); float16 for T4

    # --- Data ---
    max_audio_len: int = 64000      # 4 s at 16 kHz
    rawboost_prob: float = 0.5      # Probability of applying RawBoost per sample
    num_workers: int = 0            # Always 0 on Colab (multiprocessing issues)

    # --- MUSAN + RIR Augmentation (Huang et al., IB-CAAN 2025) ---
    # IB-CAAN achieved 4.61% ITW EER with RawBoost + MUSAN + RIR, vs 4.93% with
    # RawBoost alone. Both corpora are downloadable from OpenSLR.
    # MUSAN: openslr.org/17 — noise/speech/music segments (Snyder et al., 2015)
    # RIR:   openslr.org/28 — simulated room impulse responses (Ko et al., 2017)
    musan_path: str = ""            # Root path to MUSAN corpus; empty = disabled
    musan_prob: float = 0.5         # P(apply MUSAN noise per sample)
    rir_path: str = ""              # Root path to RIR corpus; empty = disabled
    rir_prob: float = 0.5           # P(apply RIR convolution per sample)

    # --- Checkpointing ---
    # Checkpoint averaging over the top-k epochs (Polyak & Juditsky, 1992) consistently
    # improves EER by ~0.2–0.5% absolute.  IB-CAAN (Huang et al., 2025) used top-5
    # averaging to achieve 4.61% ITW EER.  Saving every epoch maximises the candidate pool.
    save_every_n_epochs: int = 1   # Save every epoch (needed for checkpoint averaging; IB-CAAN top-5)
    save_top_k: int = 10           # Keep at most this many periodic checkpoints (oldest pruned)

    # --- Adaptive VIB ---
    # If True, halves vib_beta when epoch-average kl_loss_raw > 1000.
    # DISABLED: adaptive halving creates a negative feedback loop where
    # reduced beta removes regularization pressure, allowing KL to explode
    # further. The per-batch clamp (kl_clamp_max=10.0) is sufficient to
    # prevent KL from dominating total loss. Keeping beta fixed maintains
    # nonzero compression pressure on the posterior.
    adaptive_vib_beta: bool = False

    # --- Debug / Diagnostics ---
    # Enable for a single diagnostic run to verify mask integration end-to-end,
    # then disable for production training. Gated behind a flag to avoid
    # impacting performance or cluttering logs during normal runs.
    debug_masks: bool = False          # Enable mask shape/value assertions + logging
    debug_mask_log_batches: int = 5    # Log mask stats for first N batches, then stop

    # --- Phase 1 Optimization ---
    # Wrap frozen backbone in torch.no_grad() to skip activation storage.
    # Safe because backbone params are frozen in Phase 1 — no gradients flow
    # through them anyway. Saves ~40% GPU memory without affecting training.
    # IMPORTANT: This is NOT pre-extracting embeddings (which failed at 30.8% EER).
    # The backbone remains in the computational graph; we just skip activation storage.
    cache_phase1_embeddings: bool = True

    # --- Paths ---
    drive_log_dir: str = "/content/drive/MyDrive/Jakob/Experiments"

    def __post_init__(self):
        """Derive backbone-dependent fields and recompute grad_accum_steps.

        Sets the following fields automatically (do not set them manually):
          ``hf_model_id``         — HuggingFace model string for ``from_pretrained``
          ``hidden_dim``          — backbone hidden state dimension
          ``num_layers``          — number of Transformer encoder layers
          ``num_groups``          — number of GLADA layer groups (may be overridden)
          ``total_hidden_states`` — num_layers + 1 (the +1 accounts for the CNN
                                    feature-extractor output, which counts as layer 0)
          ``grad_accum_steps``    — effective_batch_size / (batch_size * 2), clamped
                                    to >= 1.  The *2 is because each forward pass
                                    processes one source batch and one target batch.
          ``total_epochs``        — phase1_epochs + phase2_epochs
        """
        bb = BACKBONE_REGISTRY[self.backbone]
        self.hf_model_id = bb["hf_id"]
        self.hidden_dim = bb["dim"]
        self.num_layers = bb["num_layers"]
        if self.num_groups_override > 0:
            self.num_groups = self.num_groups_override
        else:
            self.num_groups = bb["num_groups"]
        self.total_hidden_states = self.num_layers + 1  # +1 for CNN embedding layer
        actual_batch = self.batch_size * 2  # source + target batches combined
        self.grad_accum_steps = max(1, self.effective_batch_size // actual_batch)
        self.total_epochs = self.phase1_epochs + self.phase2_epochs

    def get_run_dir(self) -> str:
        """Full Drive path for this run's checkpoints and logs."""
        return os.path.join(self.drive_log_dir, self.experiment_name, self.run_id)


# =============================================================================
# set_seed — Global reproducibility helper
# Call once immediately after constructing GLADAConfig (before model/data init).
# Seeds random, numpy, and all torch RNGs (CPU + all CUDA devices).
# =============================================================================
def set_seed(seed: int) -> None:
    """Set all RNG seeds for full reproducibility across random, numpy, and torch."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
