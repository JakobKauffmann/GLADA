"""Full GLADA (Grouped Layer Adversarial Domain Adaptation) model.

Architecture (chap3.tex Section 3, matching notation):
    Raw Audio (16 kHz, 4 s)
        -> RawBoost augmentation (training only; Tak et al., ICASSP 2022)
        -> Unfrozen SSL Backbone (gradient checkpointing; Chen et al., 2022 for WavLM)
        -> Per-Layer LayerNorm (Ba et al., 2016)
              h'_i = gamma_i * (h_i - E[h_i]) / sqrt(Var[h_i] + eps) + beta_i
        -> Hierarchical Attention Pooling (Liang et al., 2026)
              w_k = softmax(theta_k),  h_tilde_k = sum w_{k,i} * h'_i
              g_k = ASP(h_tilde_k) = [mu_k ; sigma_k]  (Okabe et al., Odyssey 2018)
              pooled = W_proj * [g_1;...;g_K] + b_proj
        -> VIB Bottleneck (Alemi et al., ICLR 2017)
              z ~ N(mu_phi(pooled), sigma^2_phi(pooled))
        -> Task Classifier (binary: spoof vs. bonafide)
         + Speaker Adversary via GRL (disentangles speaker identity)
         + GLADA Domain Discriminators via GRL (one per layer group)

Total loss (chap3.tex Section 3.5):
    L = L_task + alpha * L_GLADA + beta * L_KL + gamma * L_speaker + delta * L_CKA

    Default weights: alpha=0.1, beta=1e-3, gamma=0.01, delta=0.01

    where:
      L_task    = cross-entropy on spoof/bonafide classification
      L_GLADA   = mean cross-entropy across K per-group domain discriminators
                  (Ganin et al., JMLR 2016 for GRL; Fang et al., Neurocomputing 2021
                  for multi-layer extension)
      L_KL      = KL(q_phi(z|x) || N(0,I))  (Alemi et al., ICLR 2017)
      L_speaker = cross-entropy on speaker classification (adversarial via GRL)
      L_CKA     = mean pairwise CKA between group outputs
                  (Kornblith et al., ICML 2019; Tran et al., ACM MM 2025)

Per-group domain discriminators
-------------------------------
GLADA applies a separate GRL + domain discriminator at each SSL layer group,
so domain invariance is encouraged at each level of the representation
hierarchy rather than only after pooling. Multi-layer adversarial domain
adaptation has prior work in other modalities (below); GLADA uses it with
grouped SSL speech layers.

Precedent in other modalities:
    - Vision: MLADA (Fang et al., Neurocomputing 2021) -- multi-layer adversarial
      DA on CNNs
    - NLP: DAA (Ngo Trung et al., ACL-IJCNLP 2021 Findings) -- per-layer
      domain-specific adapters with adversarial training on BERT

Empirical motivation:
    El-Kheir et al. (NAACL 2025) show different SSL layers encode different
    deepfake information -- early layers capture acoustic artifacts, later layers
    capture semantic/phonetic patterns.  A single global discriminator cannot
    adapt all abstraction levels simultaneously.

Supporting components (each builds on prior work):
    - VIB: Alemi et al. (ICLR 2017); applied to deepfake detection by Huang
      et al. (IB-CAAN, 2025, arxiv 2509.23618)
    - CKA diversity: Kornblith et al. (ICML 2019); used by Tran et al. (ACM
      MM 2025, arxiv 2509.03409)
    - Speaker adversary: disentangles speaker identity from spoof features
    - RawBoost: Tak et al. (ICASSP 2022)

GLADA combines grouped-layer DA with these supporting components.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from glada.config import GLADAConfig
from glada.models.grl import GradientReversal
from glada.models.attention import PerLayerNorm, HierarchicalAttentionPooling
from glada.models.vib import VIBBottleneck
from glada.models.discriminators import GroupedLayerDomainDiscriminators

class GLADAModel(nn.Module):
    """Main model class for GLADA (Grouped Layer Adversarial Domain Adaptation).

    Forward pass (mathematical notation matching chap3.tex):
        1. h = [h_0, h_1, ..., h_L] = Backbone(x),   h_i in R^{B x T x D}
        2. h'_i = LayerNorm_i(h_i)                    (Ba et al., 2016; per-layer)
        3. For each group k = 1..K:
             w_k = softmax(theta_k)
             h_tilde_k = sum_{i in G_k} w_{k,i} * h'_i
             g_k = ASP(h_tilde_k) = [mu_k ; sigma_k]  in R^{B x 2D}
        4. pooled = W_proj * [g_1 ; ... ; g_K] + b_proj   in R^{B x D}
        5. z ~ N(mu_phi(pooled), sigma^2_phi(pooled))  (VIB, Alemi et al., ICLR 2017)
        6. y_hat_task = Classifier(z)
        7. y_hat_domain^{(k)} = D_k(GRL(g_k, alpha))  for k=1..K  (GLADA)
             GRL: Ganin et al., JMLR 2016
        8. y_hat_speaker = S(GRL(z, alpha_spk))

    Total loss:
        L = L_task + alpha * L_GLADA + beta * L_KL + gamma * L_speaker + delta * L_CKA
        Default: alpha=0.1, beta=1e-3, gamma=0.01, delta=0.01

    Two-phase training strategy:
        Phase 1 (5 epochs): Backbone frozen, train heads only.
            Allows randomly-initialized heads to reach reasonable loss before
            backbone gradients start flowing.
        Phase 2 (remaining epochs): Backbone unfrozen with LLRD (Clark et al.,
            2020) and gradient checkpointing.  Lower layers get smaller learning
            rates to preserve pre-trained acoustic features.

    Args:
        config: ``GLADAConfig`` instance with all hyperparameters.
    """
    def __init__(self, config: GLADAConfig):
        super().__init__()
        self.config = config

        # --- SSL Backbone (e.g., WavLM-Large, Chen et al. 2022) ---
        # SDPA (Scaled Dot-Product Attention) dispatches to FlashAttention v2 on
        # A100/H100 GPUs (2-3x faster attention, O(N) vs O(N^2) memory).
        # Falls back to standard attention on T4 or unsupported hardware.
        # Note: torch.compile() is NOT used -- incompatible with WavLM's
        # dynamic masking.
        try:
            self.backbone = AutoModel.from_pretrained(
                config.hf_model_id, attn_implementation="sdpa"
            )
        except (ValueError, NotImplementedError):
            self.backbone = AutoModel.from_pretrained(config.hf_model_id)
        # Freeze CNN feature extractor permanently.
        # Why: The CNN extracts low-level spectral features that are well-learned
        # during SSL pre-training and do not benefit from task-specific fine-tuning.
        # Freezing saves ~5% GPU memory and prevents overfitting to training-domain
        # recording conditions.
        if hasattr(self.backbone, 'feature_extractor'):
            self.backbone.feature_extractor._freeze_parameters()
        elif hasattr(self.backbone, 'feature_projection'):
            for p in self.backbone.feature_projection.parameters():
                p.requires_grad = False

        # --- Per-layer normalization ---
        self.layer_norms = PerLayerNorm(config.total_hidden_states, config.hidden_dim)

        # --- Hierarchical Attention Pooling (HierCon-inspired, Liang et al. 2026) ---
        self.pooling = HierarchicalAttentionPooling(
            num_hidden_states=config.total_hidden_states,
            hidden_dim=config.hidden_dim,
            num_groups=config.num_groups,
            output_dim=config.hidden_dim,
        )

        # --- VIB Bottleneck or deterministic projection (Alemi et al. 2017) ---
        if config.use_vib:
            self.vib = VIBBottleneck(config.hidden_dim, config.bottleneck_dim)
        else:
            self.vib = None
            self.deterministic_proj = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.bottleneck_dim),
            )

        # --- Task Classifier (binary: spoof vs. bonafide) ---
        # Single hidden layer with LayerNorm + GELU + Dropout.
        # Why LayerNorm here: stabilizes classifier gradients when the VIB
        # bottleneck output scale varies during training.
        self.task_classifier = nn.Sequential(
            nn.Linear(config.bottleneck_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

        # --- Speaker Adversary (3-layer MLP with GRL) ---
        # Why adversarial speaker disentanglement: spoof detectors can overfit
        # to speaker identity as a shortcut (e.g., memorizing which speakers
        # appear only in bonafide or only in spoof).  The GRL forces the
        # bottleneck z to be speaker-invariant, improving generalization.
        # Why deeper (3-layer) than task classifier: speaker classification is
        # a harder N-way task requiring more capacity to provide a meaningful
        # adversarial signal.
        if config.use_speaker_adversary:
            self.speaker_adversary = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, config.num_speakers),
            )
        else:
            self.speaker_adversary = None

        # --- Global Domain Discriminator (single-level baseline, off by default) ---
        # When enabled, applies a single domain discriminator on the VIB output z.
        # This serves as a baseline to compare against GLADA's per-group approach.
        # Ganin et al. (JMLR 2016) showed single-level DANN is effective for
        # homogeneous features; GLADA hypothesizes that SSL's heterogeneous
        # layer hierarchy requires per-group adaptation.
        if config.use_global_domain_disc:
            self.domain_disc = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2),
            )
        else:
            self.domain_disc = None

        # --- GLADA: Per-group Domain Discriminators ---
        # Main component studied: Separate GRL + domain discriminator per layer group.
        # Precedent: MLADA (Fang et al., Neurocomputing 2021, vision),
        #            DAA (Ngo Trung et al., ACL 2021, NLP).
        # First application to any speech task. Enforces domain invariance at each
        # abstraction level independently (acoustic → phonetic → semantic → artifact).
        if config.use_glada:
            self.glada_disc = GroupedLayerDomainDiscriminators(
                num_groups=config.num_groups,
                input_dim=2 * config.hidden_dim,  # ASP output: mean + std
                hidden_dim=config.glada_disc_hidden,
            )
        else:
            self.glada_disc = None

        # GRL reversal strength (Ganin et al., JMLR 2016).
        # Updated by the trainer each epoch via a warmup schedule:
        #   alpha(p) = 2 / (1 + exp(-10*p)) - 1,  p = epoch / max_epochs
        # Starts at 0 (no reversal) and increases to ~1 (full reversal).
        # Why warmup: early reversal before the discriminator is trained
        # produces noisy gradients that destabilize the backbone
        # (GRL warmup < 5 epochs caused instability in early runs).
        self.domain_grl_alpha = 0.0
        self.speaker_grl_alpha = 0.0

    def freeze_backbone(self):
        """Freeze all backbone parameters for Phase 1 training.

        Phase 1 freezes the backbone so that randomly-initialized downstream
        heads (classifier, discriminators, VIB) can converge to reasonable
        loss values before backbone gradients start flowing.  Without this,
        the backbone receives noisy gradients from untrained heads and
        catastrophically forgets pre-trained features.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, 'feature_extractor'):
            self.backbone.feature_extractor._freeze_parameters()

    def unfreeze_backbone(self):
        """Unfreeze backbone transformer layers for Phase 2 fine-tuning.

        The CNN feature extractor remains permanently frozen (see ``__init__``).
        If ``config.freeze_bottom_n_layers > 0``, the bottom N transformer
        layers also stay frozen — these encode general acoustic features
        (phonemes, pitch, timbre) that transfer well and don't need task
        adaptation. Only the top (num_layers - N) layers are fine-tuned.

        Gradient checkpointing is enabled to trade compute for memory: only
        ~1/3 of activations are stored, with the rest recomputed during
        backprop.  This is essential for fitting WavLM-Large (316M params)
        on a single GPU with batch size 24.
        """
        for p in self.backbone.parameters():
            p.requires_grad = True
        # CNN feature extractor always frozen
        if hasattr(self.backbone, 'feature_extractor'):
            self.backbone.feature_extractor._freeze_parameters()
        # Optionally keep bottom N transformer layers frozen
        n_freeze = getattr(self.config, 'freeze_bottom_n_layers', 0)
        if n_freeze > 0 and hasattr(self.backbone, 'encoder'):
            layers = self.backbone.encoder.layers
            for i in range(min(n_freeze, len(layers))):
                for p in layers[i].parameters():
                    p.requires_grad = False
            frozen_params = sum(p.numel() for p in self.backbone.parameters()
                                if not p.requires_grad)
            print(f"  Frozen bottom {n_freeze}/{len(layers)} transformer layers "
                  f"({frozen_params:,} params frozen)")
        self.backbone.gradient_checkpointing_enable()

    def extract_hidden_states(self, input_values, attention_mask=None):
        """Extract backbone hidden states only (utility for analysis/debugging).

        Note: Full disk-based caching of hidden states is infeasible due to
        memory (~450 GB for 45K samples x 25 layers x 200 frames x 1024 dims
        at float16).  The ``torch.no_grad()`` optimization in ``forward()``
        provides most of the Phase 1 speedup without caching complexity.
        """
        with torch.no_grad():
            outputs = self.backbone(
                input_values, attention_mask=attention_mask, output_hidden_states=True
            )
        return outputs.hidden_states

    def forward(self, input_values, attention_mask=None):
        """Full forward pass through the GLADA pipeline.

        Args:
            input_values: (B, T_raw) raw audio waveform at 16 kHz.
            attention_mask: (B, T_raw) optional mask for variable-length inputs.

        Returns:
            dict with keys:
                - ``spoof_logits``:  (B, 2) task classifier output.
                - ``kl_loss``:       Scalar VIB KL divergence (0.0 if VIB disabled).
                - ``cka_loss``:      Scalar mean pairwise CKA (0.0 if CKA disabled).
                - ``features``:      (B, bottleneck_dim) bottleneck representation z.
                - ``domain_logits``: (B, 2) global domain disc output (if enabled).
                - ``glada_domain_logits``: List of K (B, 2) per-group domain logits
                    (if GLADA enabled).
                - ``group_outputs``: List of K (B, 2D) per-group ASP outputs
                    (if GLADA enabled).
                - ``speaker_logits``: (B, num_speakers) speaker adversary output
                    (if enabled).
        """
        # === Step 1: h = [h_0, ..., h_L] = Backbone(x) ===
        # When backbone is frozen (Phase 1), wrap in no_grad() to skip storing
        # intermediate activations for backprop.  This saves ~40% GPU memory
        # without affecting training -- backbone params have requires_grad=False
        # so no gradients flow through them anyway.
        # IMPORTANT: This is NOT pre-extracting embeddings (which failed at
        # 30.8% EER in early experiments).  The backbone is called every batch;
        # we just skip activation storage for its frozen parameters.
        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        if backbone_frozen:
            with torch.no_grad():
                outputs = self.backbone(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        else:
            outputs = self.backbone(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = list(outputs.hidden_states)

        # === Step 1b: Convert sample-level mask to frame-level mask ===
        # The backbone's CNN feature extractor has total stride = 320 (for
        # WavLM/Wav2Vec2).  Frame count = floor((sample_count - 1) / 320) + 1,
        # matching HuggingFace's _get_feat_extract_output_lengths() formula.
        # The frame-level mask is passed to ASP so padded frames receive zero
        # attention weight (Vaswani et al., 2017, Section 3.2.3.1).
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)                    # (B,)
            output_lengths = ((input_lengths - 1) // 320) + 1           # (B,)
            T = hidden_states[0].shape[1]
            frame_mask = (torch.arange(T, device=attention_mask.device)
                          .unsqueeze(0) < output_lengths.unsqueeze(1))  # (B, T)
        else:
            frame_mask = None

        # --- Mask diagnostics (gated by config.debug_masks) ---
        if getattr(self.config, 'debug_masks', False) and attention_mask is not None:
            B = input_values.shape[0]
            assert attention_mask.shape == input_values.shape, \
                f"attention_mask {attention_mask.shape} != input {input_values.shape}"
            unique_vals = attention_mask.unique().tolist()
            assert all(v in (0, 1) for v in unique_vals), \
                f"Non-binary attention_mask values: {unique_vals}"
            assert frame_mask.shape == (B, T), \
                f"frame_mask {frame_mask.shape} != expected ({B}, {T})"
            # Contiguity check: once a 0 appears, all subsequent values should be 0
            # (mask is [1,1,...,1,0,...,0], not interleaved)
            for i in range(min(B, 4)):
                mask_row = frame_mask[i].long()
                diffs = mask_row[1:] - mask_row[:-1]
                assert (diffs >= 0).all() or (diffs <= 0).all(), \
                    f"Non-contiguous frame_mask at sample {i}: transitions={diffs.nonzero()}"
            # Warn (don't assert) on fully-padded samples — handled by nan_to_num
            n_empty = (~frame_mask.any(dim=1)).sum().item()
            if n_empty > 0:
                import warnings
                warnings.warn(f"[MASK] {n_empty}/{B} samples fully padded (actual_length=0)")

        # === Step 2: h'_i = LayerNorm_i(h_i) for each layer i ===
        hidden_states = self.layer_norms(hidden_states)

        # === Step 3: Hierarchical pooling -> pooled, L_CKA, [g_1,...,g_K] ===
        pooled, cka_loss, group_outputs = self.pooling(
            hidden_states, frame_mask=frame_mask)
        if not self.config.use_cka_loss:
            cka_loss = torch.tensor(0.0, device=pooled.device)

        # === Step 4: z ~ N(mu_phi(pooled), sigma^2_phi(pooled)) via VIB ===
        if self.vib is not None:
            z, kl_loss = self.vib(pooled, training=self.training)
        else:
            # Deterministic fallback (ablation: no VIB)
            z = self.deterministic_proj(pooled)
            kl_loss = torch.tensor(0.0, device=pooled.device)

        # === Step 5: y_hat_task = Classifier(z) ===
        spoof_logits = self.task_classifier(z)

        result = {
            "spoof_logits": spoof_logits,
            "kl_loss": kl_loss,
            "cka_loss": cka_loss,
            "features": z,
        }

        # === Step 6: Global domain disc (single-level baseline, off by default) ===
        # GRL reverses gradients so the backbone learns domain-invariant features
        # while the discriminator learns to classify domains (Ganin et al., JMLR 2016).
        if self.domain_disc is not None:
            z_domain = GradientReversal.apply(z, self.domain_grl_alpha)
            result["domain_logits"] = self.domain_disc(z_domain)

        # === Step 7: GLADA per-group domain discriminators ===
        # y_hat_domain^{(k)} = D_k(GRL(g_k, alpha)) for k=1..K
        # Each discriminator operates on the pre-VIB group output g_k, so domain
        # invariance is enforced at every level of the SSL hierarchy independently.
        if self.glada_disc is not None:
            result["glada_domain_logits"] = self.glada_disc(group_outputs, self.domain_grl_alpha)
            result["group_outputs"] = group_outputs

        # === Step 8: y_hat_speaker = S(GRL(z, alpha_spk)) ===
        if self.speaker_adversary is not None:
            z_speaker = GradientReversal.apply(z, self.speaker_grl_alpha)
            result["speaker_logits"] = self.speaker_adversary(z_speaker)

        return result


class PerGroupVIBModel(GLADAModel):
    """GLADA variant with one VIB per group, applied BEFORE the per-group discs.

    Architecture difference vs. ``GLADAModel`` (full GLADA, chap3.tex Section 3):
        full GLADA: pooled = W_proj * [g_1;...;g_K] + b_proj  -> z = VIB(pooled)
        per-group:  z_k    = VIB_k(g_k)  for k=1..K           -> z = W * [z_1;...;z_K]

    Why this matters (paper Section V.I, lines 415-446 of ICTAI main.tex):
        Section V.H diagnosed that full GLADA's per-group discriminators feed
        gradients into a SINGLE post-pool VIB, mixing compression decisions
        across abstraction levels and producing destructive interference.
        Per-group VIB places one stochastic bottleneck INSIDE each group's
        path so each level's compression rate is chosen by that level's
        discriminator alone.  Empirically on WavLM-Base-Plus this recovers
        an 8.04pp ITW EER gain over full GLADA (11.67% vs 19.71%).

    Math (full per-group objective, extending Alemi et al., ICLR 2017):
        For each group k=1..K:
            mu_k, log sigma^2_k = Encoder_k(g_k)
            z_k = mu_k + sigma_k * epsilon_k,   epsilon_k ~ N(0, I)
            KL_k = -0.5 * sum(1 + log sigma^2_k - mu_k^2 - sigma^2_k)
        Aggregate:
            z       = W_proj * [z_1; ...; z_K] + b_proj
            KL_total = (1/G) * sum_k KL_k                  (uniform mean, default)
                     = sum_k beta_k * KL_k                 (per-group beta override)
        The trainer multiplies KL_total by config.vib_beta a second time, so
        the effective per-group KL weight is config.vib_beta / G (default) or
        config.vib_beta * beta_k (override).

    The per-group discriminators receive the compressed z_k (dim = bottleneck_dim),
    NOT the raw 2D-wide ASP output, so ``GroupedLayerDomainDiscriminators`` is
    rebuilt at construction with the smaller input dim.

    Leave-one-out support:
        ``vib_disabled_groups`` accepts a tuple of group indices for which
        z_k = mu_k (deterministic, no reparameterization) and KL_k = 0.
        Used for two complementary probes:
          - Leave-one-out ablation (phase_a_leaveout): disable one group at
            a time, measure ITW EER delta to identify which group's bottleneck
            carries the load.
          - g0-only configuration (phase_pgvib_g0only): disable groups 1..K-1,
            leaving only g0 with an active bottleneck.  Tests whether the
            recovery effect is concentrated in the lowest (acoustic) layers.

    Args (in addition to ``GLADAConfig``):
        vib_disabled_groups: tuple of int, group indices where VIB is
            deterministic and KL = 0.  Empty tuple (default) keeps all groups
            stochastic.
        vib_betas: tuple of float of length K, per-group KL weights.  None
            (default) gives uniform-mean aggregation.  Used by phase_b_confirm
            to apply asymmetric compression rates informed by leave-one-out
            results.

    References:
        Alemi et al., ICLR 2017 (Deep Variational Information Bottleneck).
        Huang et al., 2025 (IB-CAAN, arxiv 2509.23618 -- single VIB after
            adversarial domain alignment, our closest engineering analogue).
        Fang et al., Neurocomputing 2021 (MLADA -- multi-layer adversarial
            DA without per-layer bottlenecks; we add the VIB per layer).
    """

    def __init__(self, config: GLADAConfig, vib_disabled_groups=(), vib_betas=None):
        super().__init__(config)
        K = config.num_groups
        D = config.hidden_dim
        bd = config.bottleneck_dim

        # One VIB per group operates on the (B, 2D) ASP output (mean+std concat).
        self.group_vibs = nn.ModuleList(
            [VIBBottleneck(2 * D, bd) for _ in range(K)]
        )
        # Linear projection over concatenated compressed representations.
        # We do NOT use a deeper MLP here: the per-group VIBs already provide
        # nonlinearity (sampling + Encoder MLP), and we want the post-concat
        # projection to remain a linear mixing of stochastic codes so that
        # interpreting the bottleneck output as a linear combination of
        # per-level latents is exact.
        self.group_compressed_proj = nn.Linear(K * bd, bd)

        # Rebuild per-group discriminators to consume compressed z_k (dim=bd).
        if self.glada_disc is not None:
            self.glada_disc = GroupedLayerDomainDiscriminators(
                num_groups=K,
                input_dim=bd,
                hidden_dim=config.glada_disc_hidden,
            )

        # Validate and store leave-one-out / per-group beta controls.
        for k in vib_disabled_groups:
            assert 0 <= k < K, (
                f"vib_disabled_groups index {k} not in [0,{K})"
            )
        self._vib_disabled = set(vib_disabled_groups)

        if vib_betas is not None:
            assert len(vib_betas) == K, (
                f"vib_betas must have length {K}, got {len(vib_betas)}"
            )
            # register_buffer keeps it on the right device under .to(device)
            self.register_buffer(
                "_vib_betas",
                torch.tensor(vib_betas, dtype=torch.float32),
            )
        else:
            self._vib_betas = None

    def forward(self, input_values, attention_mask=None):
        """Per-group VIB forward pass.

        Differences from ``GLADAModel.forward``:
            - Skips global VIB on the pooled vector.
            - Applies ``self.group_vibs[k]`` to each group output g_k.
            - Aggregates KL across groups (uniform mean or per-group betas).
            - Per-group discriminators consume z_k (compressed) not g_k (raw).
            - ``pooled`` from the hierarchical pool is discarded (only
              ``group_outputs`` and ``cka_loss`` are used).
        """
        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        if backbone_frozen:
            with torch.no_grad():
                outputs = self.backbone(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        else:
            outputs = self.backbone(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = list(outputs.hidden_states)

        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)
            output_lengths = ((input_lengths - 1) // 320) + 1
            T = hidden_states[0].shape[1]
            frame_mask = (
                torch.arange(T, device=attention_mask.device).unsqueeze(0)
                < output_lengths.unsqueeze(1)
            )
        else:
            frame_mask = None

        hidden_states = self.layer_norms(hidden_states)
        _, cka_loss, group_outputs = self.pooling(
            hidden_states, frame_mask=frame_mask
        )
        if not self.config.use_cka_loss:
            cka_loss = torch.tensor(0.0, device=group_outputs[0].device)

        z_groups = []
        kl_per_group = []
        for k, g_k in enumerate(group_outputs):
            if k in self._vib_disabled:
                # Deterministic: z_k = mu_k via training=False (no reparam noise).
                z_k, _ = self.group_vibs[k](g_k, training=False)
                kl_k = torch.zeros((), device=g_k.device)
            else:
                z_k, kl_k = self.group_vibs[k](g_k, training=self.training)
            z_groups.append(z_k)
            kl_per_group.append(kl_k)

        # KL aggregation:
        #   uniform-mean: kl_total = (1/G) * sum_k KL_k
        #                 -> trainer's config.vib_beta acts as the SHARED beta
        #                    across groups (effective per-group beta = vib_beta / G)
        #   per-group:    kl_total = sum_k beta_k * KL_k
        #                 -> trainer's config.vib_beta still multiplies the result,
        #                    so effective per-group beta = vib_beta * beta_k
        if self._vib_betas is not None:
            kl_total = sum(
                self._vib_betas[k] * kl_per_group[k]
                for k in range(len(group_outputs))
            )
        else:
            kl_total = sum(kl_per_group) / max(len(group_outputs), 1)

        z_concat = torch.cat(z_groups, dim=-1)
        z = self.group_compressed_proj(z_concat)

        spoof_logits = self.task_classifier(z)
        result = {
            "spoof_logits": spoof_logits,
            "kl_loss": kl_total,
            "cka_loss": cka_loss,
            "features": z,
        }

        if self.glada_disc is not None:
            # Per-group discs operate on compressed z_k, NOT raw g_k.
            result["glada_domain_logits"] = self.glada_disc(
                z_groups, self.domain_grl_alpha
            )
            result["group_outputs"] = z_groups

        if self.domain_disc is not None:
            z_domain = GradientReversal.apply(z, self.domain_grl_alpha)
            result["domain_logits"] = self.domain_disc(z_domain)

        if self.speaker_adversary is not None:
            z_speaker = GradientReversal.apply(z, self.speaker_grl_alpha)
            result["speaker_logits"] = self.speaker_adversary(z_speaker)

        return result
