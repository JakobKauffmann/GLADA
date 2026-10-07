"""
GLADA Trainer — two-phase adversarial training loop.

Overview
--------
Training is split into two phases to stabilize learning when fine-tuning large SSL
backbones (wav2vec2, WavLM, data2vec) end-to-end with multiple auxiliary losses:

  Phase 1 — Frozen backbone (phase1_epochs epochs, default ~5):
    Only the downstream heads (hierarchical pooling, VIB, classifiers, discriminators)
    are trained. The SSL backbone is frozen. This lets the heads reach a reasonable
    starting point before the backbone is released, preventing the large SSL gradients
    from immediately overwriting the randomly-initialised head weights.

  Phase 2 — Full fine-tuning with LLRD (phase2_epochs epochs):
    The backbone is unfrozen. A Layer-wise Learning Rate Decay (LLRD) optimizer is
    built: later (higher) transformer layers receive higher learning rates than earlier
    (lower) layers, which preserves low-level acoustic features already learned during
    SSL pre-training while allowing task-specific adaptation at higher layers.
    GRL alpha is NOT reset at Phase 2 start. With default grl_warmup_epochs=5 and
    phase1_epochs=5, domain alpha reaches 1.0 by Phase 2 and stays there — the
    discriminators are already informative, so maintaining full reversal provides
    stable adversarial pressure during backbone fine-tuning.

Total loss composition (per batch):
  L = L_task + alpha * L_GLADA + beta * L_KL + gamma(t) * L_speaker + delta * L_CKA

  where:
    L_task   = CE(spoof_logits[src], y_src)               source-only spoof detection
    L_GLADA  = (1/K) sum_k CE(D_k(GRL(g_k)), d)          per-group domain adversarial loss
    L_KL     = KL(q(z|x) || N(0,I))                      VIB information bottleneck
    L_speaker= CE(S(GRL_spk(z[src])), spk_src)            speaker disentanglement
    L_CKA    = mean_{i<j} CKA(g_i, g_j)                  CKA diversity penalty across groups

  Loss weights and constraints:
    alpha  = domain_loss_weight  (default 0.1; must stay <= 0.1 — higher values
                                  destabilize training)
    beta   = vib_beta            (default 1e-3, with per-batch KL clamping at kl_clamp_max)
    gamma  = speaker_loss_weight  (fixed 0.1; GRL ramp controls reversal strength separately)
    delta  = cka_loss_weight     (default 0.01)

GRL warmup schedules:
  Domain GRL:  alpha(t) = min(1, t / grl_warmup_epochs)         (default E_w = 5)
  Speaker GRL: alpha(t) = min(1, t / speaker_grl_warmup_epochs)      (default E_w = 10)
  Neither is reset at Phase 2 — the global epoch index is used throughout.

Stability safeguards (lessons from failed runs):
  - KL clamping: raw KL loss is clamped to kl_clamp_max before scaling by beta.
    Without clamping, early VIB KL divergences reached >130,000, collapsing training.
  - Loss sanity check: batches with total_loss > 100 or NaN/Inf are skipped entirely.
  - domain_loss_weight cap: validated experimentally; 0.5 destabilized task loss within
    5 epochs; 0.1 is stable.
  - Gradient accumulation: supported via grad_accum_steps for effective larger batches
    on Colab T4 (num_workers=0, batch_size=24).

Data flow per training step:
  1. Source batch: (audio, task_label, speaker_id)  from ASVspoof LA train set
  2. Target batch: (audio)                          from In-the-Wild (ITW) — no labels
  3. Concatenate along batch dim; single forward pass through shared backbone+heads
  4. Slice source portion [0:n_src] for task and speaker losses
  5. Use full batch [0:n_src+n_tgt] for domain discrimination losses

References:
  - Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
  - El-Kheir et al., "Layer-Wise Analysis of SSL Models", NAACL 2025.
  - Fang et al., "MLADA", Neurocomputing 2021.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Optional, Dict
from torch.amp import autocast

from glada.config import GLADAConfig, set_seed
from glada.models.glada_model import GLADAModel
from glada.trainers.base_trainer import BaseTrainer
from glada.optimizer import (
    build_llrd_optimizer,
    build_phase1_scheduler,
    build_phase2_scheduler
)

class GLADATrainer(BaseTrainer):
    """
    Orchestrates GLADA training: model construction, optimizer setup, two-phase
    scheduling, per-epoch training, evaluation, checkpointing, and early stopping.

    Inherits checkpoint/history/AMP infrastructure from BaseTrainer.

    Args:
        config          (GLADAConfig): Fully-specified configuration dataclass.
        source_loader   (DataLoader):  ASVspoof LA training set (labeled, spoofed+bonafide).
        target_loader   (DataLoader):  In-the-Wild set (unlabeled, used only for domain DA).
        eval_loader     (DataLoader):  Primary evaluation loader (ITW test split).
        extra_eval_loaders (dict):     Optional name->DataLoader for additional test sets
                                       (e.g. ASVspoof eval partition).
        num_speakers    (int|None):    If provided, overrides config.num_speakers with the
                                       value detected from the actual training dataset.
        resume_from     (str|None):    Path to a .pt checkpoint to resume from.
        model_class     (type|None):   Model constructor; defaults to ``GLADAModel``.
                                       Pass ``PerGroupVIBModel`` for the per-group-VIB
                                       repair variant.  The class must accept ``config``
                                       as its first positional argument; any extra
                                       kwargs go through ``model_kwargs``.
        model_kwargs    (dict|None):   Extra keyword arguments forwarded to ``model_class``
                                       (e.g. ``{'vib_disabled_groups': (1, 2)}`` for the
                                       g0-only PerGroupVIBModel probe).
    """

    def __init__(self, config: GLADAConfig, source_loader, target_loader,
                 eval_loader, extra_eval_loaders=None, num_speakers=None,
                 resume_from: Optional[str] = None,
                 model_class=None, model_kwargs=None):

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config.seed)

        # Update num_speakers if detected from dataset
        if num_speakers is not None and num_speakers != config.num_speakers:
            print(f"  Updating num_speakers: {config.num_speakers} -> {num_speakers}")
            config.num_speakers = num_speakers

        # Build Model — default is the full GLADAModel; callers can pass
        # PerGroupVIBModel (or any GLADAModel subclass) plus its kwargs.
        ModelCls = model_class if model_class is not None else GLADAModel
        mkwargs = model_kwargs or {}
        print(f"\nBuilding {ModelCls.__name__}: {config.backbone}" +
              (f"  kwargs={mkwargs}" if mkwargs else ""))
        model = ModelCls(config, **mkwargs)
        model.to(self.device)
        
        # Initialize BaseTrainer
        super().__init__(config, model, self.device)

        self.source_loader = source_loader
        self.target_loader = target_loader
        self.eval_loader = eval_loader
        self.extra_eval_loaders = extra_eval_loaders or {}

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.speaker_ce_loss = nn.CrossEntropyLoss()

        # KL monitoring: consecutive clamp counter for VIB health warnings
        self._kl_clamp_count = 0
        # Batch skip monitoring: total skipped batch counter
        self._skip_count = 0

        # Initialize Phase 1 by default
        self._setup_phase1()

        # Resume if requested
        if resume_from and os.path.exists(resume_from):
            self._resume(resume_from)

        # Log parameter counts
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,}")
        print(f"  Run dir: {self.run_dir}")

    def _setup_phase1(self):
        """
        Configure Phase 1: backbone frozen, heads-only training.

        A single-group AdamW optimizer is used with a flat head_lr (no LLRD needed
        because only the non-backbone parameters are being updated). The scheduler
        provides a short linear warmup followed by cosine decay.

        Rationale for freezing: SSL backbones contain hundreds of millions of
        parameters. Training them from random head initialisation without a warmup
        phase causes the early task gradients to distort the pre-trained features
        before the heads have converged enough to produce reliable supervision.
        """
        self.current_phase = 1
        self.model.freeze_backbone()

        head_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            head_params,
            lr=self.config.head_lr,
            weight_decay=self.config.weight_decay
        )

        steps_per_epoch = len(self.source_loader)
        self.scheduler = build_phase1_scheduler(self.optimizer, self.config, steps_per_epoch)

    def _setup_phase2(self, epoch=None):
        """
        Configure Phase 2: backbone unfrozen, end-to-end LLRD fine-tuning.

        LLRD (Layer-wise Learning Rate Decay): later transformer layers receive a higher
        learning rate (backbone_lr) while earlier layers are multiplied by
        llrd_decay^(num_layers - layer_index). This preserves low-level acoustic
        representations learned during SSL pre-training while allowing higher-level
        task-specific adaptation. Heads retain head_lr.

        Phase 2 adversarial ramp (phase2_adv_warmup_epochs > 0):
          At backbone unfreeze, GRL alphas are reset to phase2_adv_ramp_start and
          linearly ramped back to full strength over phase2_adv_warmup_epochs.
          This prevents "gradient shock" where 5 adversarial signals at full strength
          destabilise the freshly-trainable backbone layers.
          Analogous to IB-CAAN's GRL coefficient scheduling (Huang et al., 2025).

        Note: build_phase2_scheduler handles the per-group max_lr matching between
        the LLRD optimizer param groups and the OneCycleLR scheduler.

        Args:
            epoch: Global epoch at which Phase 2 starts.  Used to track the
                   Phase 2 adversarial ramp.  Defaults to config.phase1_epochs.
        """
        self.current_phase = 2
        self._phase2_start_epoch = epoch if epoch is not None else self.config.phase1_epochs
        self.model.unfreeze_backbone()

        # Rebuild optimizer with per-layer learning rates (LLRD)
        self.optimizer = build_llrd_optimizer(self.model, self.config)

        steps_per_epoch = len(self.source_loader)
        self.scheduler = build_phase2_scheduler(self.optimizer, self.config, steps_per_epoch)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n  Phase 2: Backbone unfrozen. Trainable: {trainable:,}")

    def _resume(self, path):
        """
        Restore model weights, optimizer state, and training history from a checkpoint.

        Phase-awareness: if the saved epoch is >= phase1_epochs, the LLRD optimizer
        must be built (via _setup_phase2) before loading its state dict, because
        the Phase-1 and Phase-2 optimizer param groups have different shapes.
        Attempting to load a Phase-2 optimizer state dict into a Phase-1 optimizer
        will raise a size mismatch error.
        """
        ckpt = self._load_checkpoint(path)

        if 'optimizer' in ckpt:
            # Rebuild the correct optimizer structure before loading state
            if self.start_epoch >= self.config.phase1_epochs:
                self._setup_phase2()
            self.optimizer.load_state_dict(ckpt['optimizer'])

        print(f"  Resumed at epoch {self.start_epoch}, best EER: {self.best_eer:.2f}%")

    def _get_grl_alphas(self, epoch):
        """
        Compute GRL reversal strengths for the domain and speaker adversaries.

        Domain GRL (Ganin et al., 2016):
          alpha_domain(t) = min(1, t / E_w)
          where t = epoch, E_w = grl_warmup_epochs (default 5).
          Clamped at 1.0 so reversal never exceeds gradient magnitude.

        Speaker GRL (slower warmup to avoid over-disentangling early):
          alpha_speaker(t) = min(1, t / E_ws)
          where E_ws = speaker_grl_warmup_epochs (default 10).
          Ramps to 1.0 (same as domain GRL).  The speaker_loss_weight is applied
          as a separate fixed coefficient in the loss aggregation step, matching
          the domain path where domain_loss_weight is separate from domain_alpha.

        Phase 2 adversarial ramp (phase2_adv_warmup_epochs > 0):
          When the backbone unfreezes, full-strength reversed gradients from
          5 adversarial signals cause "gradient shock".  The ramp resets alphas
          to phase2_adv_ramp_start and linearly restores them over N epochs:
            ramp_factor(t) = min(1, start + (1 - start) * (t - t_p2) / N)
          Both domain_alpha and speaker_alpha are multiplied by ramp_factor.

        Both alphas are intentionally epoch-level (not step-level) for simplicity.
        Ganin et al. use a step-level schedule; the epoch-level approximation is
        sufficient given the small number of epochs used here.

        Args:
            epoch (int): Current training epoch (0-indexed, not phase-relative).

        Returns:
            (float, float): (domain_alpha, speaker_alpha).
        """
        # Linear warmup from 0 to 1 over grl_warmup_epochs
        domain_alpha = min(1.0, epoch / max(self.config.grl_warmup_epochs, 1))

        # Linear warmup from 0 to 1 over speaker_grl_warmup_epochs
        # (matches domain GRL; speaker_loss_weight applied separately in loss aggregation)
        speaker_alpha = min(1.0, epoch / max(self.config.speaker_grl_warmup_epochs, 1))

        # Phase 2 adversarial ramp: prevent gradient shock at backbone unfreeze.
        # Without this, domain_alpha=1.0 from epoch 5 immediately drives full-strength
        # reversed gradients through the freshly unfrozen backbone (observed: ITW EER
        # 13.5% → 20%+).  Analogous to IB-CAAN's GRL scheduling (Huang et al., 2025).
        p2_warmup = self.config.phase2_adv_warmup_epochs
        if self.current_phase == 2 and p2_warmup > 0:
            p2_start = getattr(self, '_phase2_start_epoch', self.config.phase1_epochs)
            p2_progress = (epoch - p2_start) / max(p2_warmup, 1)
            ramp_start = self.config.phase2_adv_ramp_start
            self._p2_ramp_factor = min(1.0, ramp_start + (1.0 - ramp_start) * p2_progress)
            domain_alpha *= self._p2_ramp_factor
            speaker_alpha *= self._p2_ramp_factor
        else:
            self._p2_ramp_factor = 1.0

        return domain_alpha, speaker_alpha

    def _log_gradient_norms(self):
        """
        Compute per-module gradient L2 norms for Phase 2 diagnostics.

        When backbone unfreezes, reversed gradients from domain/speaker adversaries
        compete with task gradients.  If adversarial norms >> task norms, the backbone
        is being pulled toward domain confusion faster than it learns spoof cues.

        Only called during Phase 2 (every 50 optimizer steps) to limit overhead.

        Returns:
            dict: {grad_norm_<module>: float} for each module with gradients.
        """
        norms = {}
        for name in ('backbone', 'task_classifier', 'glada_disc',
                     'speaker_adversary', 'pooling', 'vib'):
            module = getattr(self.model, name, None)
            if module is None:
                continue
            total_sq = 0.0
            count = 0
            for p in module.parameters():
                if p.grad is not None:
                    total_sq += p.grad.data.norm(2).item() ** 2
                    count += 1
            if count > 0:
                norms[f"grad_norm_{name}"] = total_sq ** 0.5
        return norms

    def train_epoch(self, epoch):
        """
        Run one full pass over the source DataLoader with interleaved target batches.

        All loss terms are individually gated by config flags (use_glada,
        use_global_domain_disc, use_speaker_adversary, use_cka_loss) so that any
        subset can be disabled for ablation studies without modifying this method.

        Loss composition:
          L = L_task
            + alpha * L_GLADA       (if use_glada)
            + alpha * L_domain      (if use_global_domain_disc, baseline DANN)
            + beta  * L_KL          (always; VIB regularization)
            + gamma * L_speaker     (if use_speaker_adversary)
            + delta * L_CKA         (if use_cka_loss)

          alpha  = domain_loss_weight (default 0.1; > 0.1 destabilizes — see module docstring)
          beta   = vib_beta (default 1e-3); L_KL is clamped per batch at kl_clamp_max
          gamma  = speaker_loss_weight (fixed 0.1; GRL ramp controls reversal strength separately)
          delta  = cka_loss_weight (default 0.01)

        Gradient accumulation:
          Losses are divided by grad_accum_steps before backward. Optimizer.step()
          fires every grad_accum_steps batches. This allows effective batch sizes
          larger than GPU memory permits (important on Colab T4 with num_workers=0).

        Args:
            epoch (int): Current epoch index (0-indexed), used for GRL alpha warmup.

        Returns:
            dict: Per-metric averages over all steps in this epoch.
        """
        self.model.train()
        domain_alpha, speaker_alpha = self._get_grl_alphas(epoch)
        self.model.domain_grl_alpha = domain_alpha
        self.model.speaker_grl_alpha = speaker_alpha

        # Wire up mask debug counter for pooling module
        if getattr(self.config, 'debug_masks', False):
            self.model.pooling._debug_mask_batches = self.config.debug_mask_log_batches

        # KL annealing (Bowman et al., ACL 2016): linearly ramp vib_beta from 0 to
        # its target value over kl_anneal_epochs. This lets the classifier converge
        # before VIB compression kicks in, preventing both posterior collapse (beta
        # too high early) and uncontrolled KL drift (beta too low to matter).
        anneal_epochs = getattr(self.config, 'kl_anneal_epochs', 0)
        if anneal_epochs > 0 and epoch < anneal_epochs:
            # Linear ramp: epoch 0 -> beta_fraction = 1/anneal_epochs,
            #               epoch anneal_epochs-1 -> beta_fraction = 1.0
            beta_fraction = (epoch + 1) / anneal_epochs
            effective_beta = self.config.vib_beta * beta_fraction
        else:
            effective_beta = self.config.vib_beta

        metrics_acc = {
            'spoof_loss': 0, 'domain_loss': 0, 'glada_loss': 0,
            'speaker_loss': 0, 'kl_loss': 0, 'kl_raw': 0, # Log raw vs clamped
            'cka_loss': 0, 'total_loss': 0,
            'spoof_acc': 0, 'speaker_acc': 0,
        }
        grad_norm_acc = {}  # Phase 2 only: per-module gradient norms
        grad_norm_count = 0
        steps = 0
        target_iter = iter(self.target_loader)
        self.optimizer.zero_grad(set_to_none=True)

        for step, s_batch in enumerate(
                pbar := tqdm(self.source_loader, desc=f"Ep {epoch+1}")):
            
            # --- Prepare Data ---
            s_inp = s_batch['input_values'].to(self.device, non_blocking=True)
            s_mask = s_batch['attention_mask'].to(self.device, non_blocking=True)
            s_label = s_batch['task_label'].to(self.device, non_blocking=True)
            s_spk = s_batch['speaker_id'].to(self.device, non_blocking=True)

            try:
                t_batch = next(target_iter)
            except StopIteration:
                # Target dataset may be smaller than source; cycle it to ensure every
                # source batch has a corresponding target batch for domain adaptation.
                target_iter = iter(self.target_loader)
                t_batch = next(target_iter)
            t_inp = t_batch['input_values'].to(self.device, non_blocking=True)
            t_mask = t_batch['attention_mask'].to(self.device, non_blocking=True)

            # Concatenate Source + Target
            c_inp = torch.cat([s_inp, t_inp])
            c_mask = torch.cat([s_mask, t_mask])
            n_src = s_inp.size(0)

            # --- Mask diagnostics (first N batches only) ---
            if getattr(self.config, 'debug_masks', False) and step < self.config.debug_mask_log_batches:
                src_pad = 1.0 - s_mask.float().mean().item()
                tgt_pad = 1.0 - t_mask.float().mean().item()
                print(f"  [MASK] Step {step}: src_pad={src_pad:.1%}, tgt_pad={tgt_pad:.1%}, "
                      f"src_lens=[{s_mask.sum(1).min().item():.0f}-{s_mask.sum(1).max().item():.0f}], "
                      f"tgt_lens=[{t_mask.sum(1).min().item():.0f}-{t_mask.sum(1).max().item():.0f}]")

            # --- Forward Pass ---
            with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
                out = self.model(c_inp, attention_mask=c_mask)

                # 1. Spoof Loss (Source only)
                spoof_loss = self.ce_loss(out['spoof_logits'][:n_src], s_label)

                # 2. Domain labels: source=0, target=1 (binary classification for each
                #    discriminator). Used for both the global baseline and GLADA losses.
                domain_labels = torch.cat([
                    torch.zeros(n_src, dtype=torch.long, device=self.device),
                    torch.ones(t_inp.size(0), dtype=torch.long, device=self.device),
                ])

                # Global domain discriminator (standard DANN baseline; use_global_domain_disc)
                domain_loss = torch.tensor(0.0, device=self.device)
                if 'domain_logits' in out:
                    domain_loss = self.ce_loss(out['domain_logits'], domain_labels)

                # GLADA per-group domain loss: average CE across K group discriminators.
                # L_GLADA = (1/K) * sum_k CE(D_k(GRL(g_k)), d)
                #
                # Two reduction modes:
                #   1) Standard (config.confidence_aware_adv_weight=False, default):
                #        per-group loss = E_b [ CE(D_k(g_k_b), d_b) ]
                #      i.e. the existing mean-over-batch behavior.  This preserves
                #      every pre-extension result bit-for-bit.
                #
                #   2) Confidence-aware (config.confidence_aware_adv_weight=True):
                #        m_k_b = |softmax(D_k(g_k_b))[1] - 0.5|                  (margin)
                #        per-group loss = (sum_b m_k_b * CE(...)_b) / sum_b m_k_b
                #      Samples on which the discriminator is uncertain (m -> 0)
                #      receive ~zero adversarial weight, so the GRL gradient
                #      flowing back to the backbone is dominated by samples where
                #      the discriminator is confident.  Following Huang et al.,
                #      IB-CAAN 2025 (arxiv 2509.23618, eq. 7), extended from a
                #      single domain disc to the GLADA per-group disc bank.
                #
                # In both modes we then mean across the K groups so L_GLADA's
                # magnitude is independent of the group count.
                glada_loss = torch.tensor(0.0, device=self.device)
                if 'glada_domain_logits' in out:
                    if getattr(self.config, 'confidence_aware_adv_weight', False):
                        group_losses = []
                        for lg in out['glada_domain_logits']:
                            # Confidence margin |p(d=target) - 0.5|, no grad
                            # through the weight (we want it to gate, not learn).
                            probs = torch.softmax(lg.detach(), dim=1)[:, 1]
                            margin = torch.abs(probs - 0.5)
                            ce_per_sample = F.cross_entropy(
                                lg, domain_labels, reduction='none')
                            # Weighted mean: sum(w * ce) / sum(w), with a small
                            # eps so the denominator never hits zero if the
                            # discriminator collapses to 0.5 on every sample.
                            weighted = (margin * ce_per_sample).sum() / (
                                margin.sum() + 1e-8)
                            group_losses.append(weighted)
                    else:
                        group_losses = [self.ce_loss(lg, domain_labels)
                                        for lg in out['glada_domain_logits']]
                    # Mean (not sum) so L_GLADA magnitude does not grow with K.
                    glada_loss = torch.stack(group_losses).mean()

                # 3. Speaker Loss (Source only)
                speaker_loss = torch.tensor(0.0, device=self.device)
                if 'speaker_logits' in out:
                    speaker_loss = self.speaker_ce_loss(
                        out['speaker_logits'][:n_src], s_spk)

                # 4. KL Loss (VIB) - CRITICAL FIX: CLAMPING
                # Without this clamp, KL loss exploded to 133,598 in initial run,
                # causing total training collapse. Track both raw and clamped
                # separately to monitor VIB health.
                kl_raw = out['kl_loss']
                kl_loss = torch.clamp(kl_raw, max=self.config.kl_clamp_max)

                # KL clamp monitoring: throttled to milestone counts to avoid log spam
                if kl_raw.item() > self.config.kl_clamp_max:
                    self._kl_clamp_count += 1
                    n = self._kl_clamp_count
                    if n in (3, 10, 100, 1000) or (n > 1000 and n % 1000 == 0):
                        print(f"  [WARN] KL clamped {n}x consecutive "
                              f"(raw={kl_raw.item():.3e}, clamp={self.config.kl_clamp_max})")
                else:
                    if self._kl_clamp_count >= 3:
                        print(f"  [INFO] KL clamp resolved after {self._kl_clamp_count} consecutive batches")
                    self._kl_clamp_count = 0

                # 5. CKA Loss (Diversity)
                cka_loss = out['cka_loss']

                # --- Total Loss Aggregation ---
                total_loss = spoof_loss + effective_beta * kl_loss
                
                if self.config.use_glada:
                    total_loss = total_loss + self.config.domain_loss_weight * glada_loss
                if self.config.use_global_domain_disc:
                    total_loss = total_loss + self.config.domain_loss_weight * domain_loss
                if self.config.use_speaker_adversary:
                    total_loss = total_loss + self.config.speaker_loss_weight * speaker_loss
                if self.config.use_cka_loss:
                    # Apply Phase 2 ramp to CKA weight if enabled — CKA diversity
                    # pressure on group outputs can conflict with backbone adaptation
                    # during early Phase 2, compounding the adversarial gradient shock.
                    effective_cka_w = self.config.cka_loss_weight
                    if self.config.phase2_cka_ramp and self.current_phase == 2:
                        effective_cka_w *= self._p2_ramp_factor
                    total_loss = total_loss + effective_cka_w * cka_loss
                
                # Divide by grad_accum_steps so that the accumulated gradient equals
                # the gradient of a single large batch of size (batch_size * accum_steps).
                total_loss = total_loss / self.config.grad_accum_steps

            # --- CRITICAL FIX: Loss Sanity Check ---
            # Skip update if loss is exploding (NaN/Inf or extremely high).
            # Warning throttled to milestone counts to avoid log spam.
            if (torch.isnan(total_loss) or torch.isinf(total_loss) or
                total_loss.item() * self.config.grad_accum_steps > 100):
                self._skip_count += 1
                s = self._skip_count
                if s in (1, 10) or s % 100 == 0:
                    raw_loss = total_loss.item() * self.config.grad_accum_steps
                    print(f"Warning: Skipping batch (#{s} total, loss={raw_loss:.3e})")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # --- Backward Pass ---
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # --- Optimizer Step ---
            if (step + 1) % self.config.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip)

                # Phase 2 gradient norm logging (every 50 optimizer steps)
                if self.current_phase == 2 and (step + 1) % (50 * self.config.grad_accum_steps) == 0:
                    gn = self._log_gradient_norms()
                    for k, v in gn.items():
                        grad_norm_acc[k] = grad_norm_acc.get(k, 0.0) + v
                    grad_norm_count += 1

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # --- Metrics logging ---
            with torch.no_grad():
                spoof_pred = out['spoof_logits'][:n_src].argmax(1)
                spoof_acc = (spoof_pred == s_label).float().mean().item()
                spk_acc = 0.0
                if 'speaker_logits' in out:
                    spk_pred = out['speaker_logits'][:n_src].argmax(1)
                    spk_acc = (spk_pred == s_spk).float().mean().item()

            metrics_acc['spoof_loss'] += spoof_loss.item()
            metrics_acc['domain_loss'] += domain_loss.item()
            metrics_acc['glada_loss'] += glada_loss.item()
            metrics_acc['speaker_loss'] += speaker_loss.item()
            metrics_acc['kl_loss'] += kl_loss.item()
            metrics_acc['kl_raw'] += kl_raw.item()
            metrics_acc['cka_loss'] += cka_loss.item()
            metrics_acc['total_loss'] += total_loss.item() * self.config.grad_accum_steps
            metrics_acc['spoof_acc'] += spoof_acc
            metrics_acc['speaker_acc'] += spk_acc
            steps += 1

            pbar.set_postfix({
                'L': f"{metrics_acc['total_loss']/steps:.3f}",
                'spf': f"{spoof_acc:.2f}",
                'KL': f"{metrics_acc['kl_loss']/steps:.2f}", 
            })

        epoch_metrics = {k: v / max(steps, 1) for k, v in metrics_acc.items()}

        # Adaptive VIB beta: if enabled and epoch-average raw KL > 1000, halve beta.
        # Reduces VIB regularization pressure when KL diverges.
        # Disabled by default (adaptive_vib_beta=False) to preserve reproducibility.
        if self.config.adaptive_vib_beta and epoch_metrics.get('kl_raw', 0) > 1000:
            old_beta = self.config.vib_beta
            self.config.vib_beta *= 0.5
            print(f"  [ADAPT] VIB beta halved: {old_beta} -> {self.config.vib_beta} "
                  f"(KL raw avg={epoch_metrics['kl_raw']:.1f} > 1000)")

        # Store effective beta in metrics for logging
        epoch_metrics['effective_beta'] = effective_beta

        # Add averaged gradient norms (Phase 2 only)
        if grad_norm_count > 0:
            for k, v in grad_norm_acc.items():
                epoch_metrics[k] = v / grad_norm_count

        return epoch_metrics

    def evaluate_all(self):
        """
        Evaluate on the primary eval set (ITW) and any additional registered loaders.

        ITW (In-the-Wild) is the primary metric used for early stopping and best-model
        selection because it represents the out-of-domain generalisation target.
        ASVspoof eval is tracked as secondary to monitor in-domain performance.

        Returns:
            dict: {dataset_name: {"eer": float, "auc": float, "acc": float, "name": str}}
        """
        results = {}
        results["ITW"] = self.evaluate(self.eval_loader, "ITW")
        for name, loader in self.extra_eval_loaders.items():
            results[name] = self.evaluate(loader, name)
        return results

    def train(self):
        """
        Execute the complete two-phase training schedule.

        Phase transition: at epoch == phase1_epochs, a phase-transition checkpoint is
        saved (the best frozen-backbone snapshot), then _setup_phase2() is called to
        unfreeze the backbone and rebuild the LLRD optimizer.

        Early stopping: only active in Phase 2 to avoid stopping prematurely during
        the frozen-backbone warmup. Patience is counted in epochs without ITW EER
        improvement. Phase 1 always runs to completion.

        Returns:
            dict with keys: best_eer, best_epoch, run_dir, history.
        """
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"GLADA Training | {cfg.backbone} | seed={cfg.seed}")
        print(f"  Phase 1: {cfg.phase1_epochs} ep | Phase 2: {cfg.phase2_epochs} ep")
        print(f"  Loss Weights: D={cfg.domain_loss_weight}, KL={cfg.vib_beta}, "
              f"SPK={cfg.speaker_loss_weight}, CKA={cfg.cka_loss_weight}")
        if getattr(cfg, 'kl_anneal_epochs', 0) > 0:
            print(f"  KL Annealing: 0 -> {cfg.vib_beta} over {cfg.kl_anneal_epochs} epochs")
        if cfg.phase2_adv_warmup_epochs > 0:
            print(f"  Phase 2 Adv Ramp: {cfg.phase2_adv_ramp_start} -> 1.0 "
                  f"over {cfg.phase2_adv_warmup_epochs} P2 epochs"
                  f"{' (+CKA)' if cfg.phase2_cka_ramp else ''}")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(self.start_epoch, cfg.total_epochs):
            # Switch to Phase 2
            if epoch == cfg.phase1_epochs and self.current_phase == 1:
                # Always save at phase transition — this is the best frozen-backbone checkpoint
                self._save_checkpoint(epoch - 1, self.optimizer, phase_transition=True)
                print(f"\n{'='*40} PHASE 2: Unfreezing backbone {'='*40}")
                self._setup_phase2(epoch)

            train_m = self.train_epoch(epoch)
            eval_results = self.evaluate_all()
            primary_eer = eval_results["ITW"]["eer"]

            # Log History (includes both raw and clamped KL for VIB monitoring)
            epoch_log = {
                "epoch": epoch + 1,
                "phase": self.current_phase,
                "train": train_m,
                "kl_loss_raw": train_m.get("kl_raw", 0),
                "kl_loss_clamped": train_m.get("kl_loss", 0),
                "eval": {k: v for k, v in eval_results.items()},
                "domain_alpha": self.model.domain_grl_alpha,
                "speaker_alpha": self.model.speaker_grl_alpha,
                "effective_speaker_weight": self.config.speaker_loss_weight * self.model.speaker_grl_alpha,
                "p2_adv_ramp_factor": getattr(self, '_p2_ramp_factor', 1.0),
                "lr": self.optimizer.param_groups[0]['lr'],
            }
            self.history.append(epoch_log)
            self._save_history()

            # Best Model Logic
            is_best = primary_eer < self.best_eer
            if is_best:
                self.best_eer = primary_eer
                self.best_epoch = epoch + 1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self._save_checkpoint(epoch, self.optimizer, is_best=is_best)

            # Print Status
            eval_str = " | ".join(
                f"{r['name']}={r['eer']:.2f}%" for r in eval_results.values()
            )
            best_str = " BEST" if is_best else ""
            print(f"  Ep {epoch+1} P{self.current_phase}: "
                  f"L={train_m['total_loss']:.3f} KL={train_m['kl_loss']:.2f} "
                  f"| {eval_str}{best_str}")

            # Early Stopping (Phase 2 Only)
            if self.current_phase == 2 and self.patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

            torch.cuda.empty_cache()

        elapsed = (time.time() - start_time) / 60
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f} min | Best ITW EER: {self.best_eer:.2f}% (Ep {self.best_epoch})")
        print(f"{'='*60}")

        return {
            "best_eer": self.best_eer,
            "best_epoch": self.best_epoch,
            "run_dir": self.run_dir,
            "history": self.history,
        }
