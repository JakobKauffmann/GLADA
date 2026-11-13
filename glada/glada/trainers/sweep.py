"""
Multi-configuration sweep runner for GLADA experiments.

Two-phase approach:
  Phase 1 (Screening): ~50+ configs with short training (15 epochs) to
      identify promising directions across backbones, num_groups, frozen
      epochs, ablation toggles, and hyperparameters.
  Phase 2 (Full Training): Top K configs (default 15) trained to
      convergence with full epoch budget.

Loads datasets once and reuses across configurations, eliminating redundant
I/O overhead. Based on patterns from domain_adaptation_ablation.py which
successfully ran 137 configs in a single session.
"""

import copy
import gc
import json
import os
import time
from typing import List, Dict, Optional

import torch

from glada.config import GLADAConfig, set_seed
from glada.models.glada_model import GLADAModel
from glada.trainers.glada_trainer import GLADATrainer
from glada.datasets.loader import create_dataloaders


# ============================================================================
# Config name generation
# ============================================================================

def _config_name(cfg: GLADAConfig) -> str:
    """Generate a descriptive name from config settings."""
    parts = [cfg.backbone.replace("-", ""), f"s{cfg.seed}"]

    # Architecture overrides
    if cfg.num_groups_override > 0:
        parts.append(f"g{cfg.num_groups}")
    if cfg.phase1_epochs != 5:
        parts.append(f"frz{cfg.phase1_epochs}")

    # Ablation flags (only note disabled ones)
    if not cfg.use_glada:
        parts.append("no-glada")
    if cfg.use_global_domain_disc:
        parts.append("globalDA")
    if not cfg.use_vib:
        parts.append("no-vib")
    if not cfg.use_speaker_adversary:
        parts.append("no-spk")
    if not cfg.use_cka_loss:
        parts.append("no-cka")
    if not cfg.use_hierarchical:
        parts.append("no-hier")
    if not cfg.use_rawboost:
        parts.append("no-rb")

    # Hyperparameter deviations from defaults
    if cfg.domain_loss_weight != 0.1:
        parts.append(f"dlw{cfg.domain_loss_weight}")
    if cfg.speaker_loss_weight != 0.1:
        parts.append(f"slw{cfg.speaker_loss_weight}")
    if cfg.cka_loss_weight != 0.01:
        parts.append(f"cka{cfg.cka_loss_weight}")
    if cfg.vib_beta != 1e-4:
        parts.append(f"vb{cfg.vib_beta}")
    if cfg.backbone_lr != 1e-5:
        parts.append(f"blr{cfg.backbone_lr}")
    if cfg.head_lr != 1e-3:
        parts.append(f"hlr{cfg.head_lr}")
    if cfg.llrd_decay != 0.95:
        parts.append(f"llrd{cfg.llrd_decay}")
    if cfg.bottleneck_dim != 256:
        parts.append(f"bn{cfg.bottleneck_dim}")
    if cfg.glada_disc_hidden != 128:
        parts.append(f"dh{cfg.glada_disc_hidden}")

    return "_".join(parts)


# ============================================================================
# Sweep config generators
# ============================================================================

def _make(base: GLADAConfig, name: str, **overrides) -> GLADAConfig:
    """Deep-copy base config, apply overrides, set experiment_name."""
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()  # recompute derived fields
    cfg.experiment_name = name
    return cfg


def generate_sweep_configs(
    batch_size: int = 24,
    amp_dtype: str = "float16",
    num_workers: int = 0,
) -> List[GLADAConfig]:
    """Generate 54 diverse configs spanning the full GLADA design space.

    Blocks:
      A. Backbone comparison (5): one per backbone, all defaults
      B. num_groups on wavlm-large (3): 3, 6, 8
      C. Frozen epochs on wavlm-large (2): 3, 10
      D. Single ablations on wavlm-large (7): -GLADA, global-DA, -VIB,
         -Speaker, -CKA, -Hierarchical, -RawBoost
      E. Pairwise ablations on wavlm-large (6)
      F. Hyperparameter sweep on wavlm-large (18): loss weights, LRs, dims
      G. Cross-backbone ablation (6): key ablations on xlsr-300m, data2vec
      H. Interesting combos (5): multi-axis variations
      I. Multi-seed on wavlm-large (2): seeds 123, 456

    Returns:
        List of 54 GLADAConfig objects.
    """
    configs = []

    # Shared base
    base = GLADAConfig(
        batch_size=batch_size,
        amp_dtype=amp_dtype,
        num_workers=num_workers,
    )

    # --- Block A: Backbone comparison (5) ---
    for bb in ["wavlm-large", "xlsr-300m", "data2vec-large",
               "wav2vec2-robust", "wavlm-base-plus"]:
        c = _make(base, f"A_backbone_{bb}", backbone=bb)
        configs.append(c)

    # --- Block B: num_groups on wavlm-large (3) ---
    for ng in [3, 6, 8]:
        c = _make(base, f"B_groups_{ng}", num_groups_override=ng)
        configs.append(c)

    # --- Block C: Frozen epochs on wavlm-large (2) ---
    for frz in [3, 10]:
        c = _make(base, f"C_frozen_{frz}", phase1_epochs=frz)
        configs.append(c)

    # --- Block D: Single ablations on wavlm-large (7) ---
    # D1: -GLADA (key ablation)
    configs.append(_make(base, "D_no_glada", use_glada=False))
    # D2: Global DA baseline (no per-group, single global disc)
    configs.append(_make(base, "D_global_da",
                         use_glada=False, use_global_domain_disc=True))
    # D3-D7: other single ablations
    configs.append(_make(base, "D_no_vib", use_vib=False))
    configs.append(_make(base, "D_no_speaker", use_speaker_adversary=False))
    configs.append(_make(base, "D_no_cka", use_cka_loss=False))
    configs.append(_make(base, "D_no_hier", use_hierarchical=False))
    configs.append(_make(base, "D_no_rawboost", use_rawboost=False))

    # --- Block E: Pairwise ablations on wavlm-large (6) ---
    configs.append(_make(base, "E_no_glada_vib",
                         use_glada=False, use_vib=False))
    configs.append(_make(base, "E_no_glada_cka",
                         use_glada=False, use_cka_loss=False))
    configs.append(_make(base, "E_no_vib_speaker",
                         use_vib=False, use_speaker_adversary=False))
    configs.append(_make(base, "E_no_vib_cka",
                         use_vib=False, use_cka_loss=False))
    configs.append(_make(base, "E_no_speaker_cka",
                         use_speaker_adversary=False, use_cka_loss=False))
    configs.append(_make(base, "E_no_hier_rawboost",
                         use_hierarchical=False, use_rawboost=False))

    # --- Block F: Hyperparameter sweep on wavlm-large (18) ---
    # Loss weights
    configs.append(_make(base, "F_dlw_0.01", domain_loss_weight=0.01))
    configs.append(_make(base, "F_dlw_0.05", domain_loss_weight=0.05))
    configs.append(_make(base, "F_slw_0.01", speaker_loss_weight=0.01))
    configs.append(_make(base, "F_slw_0.05", speaker_loss_weight=0.05))
    configs.append(_make(base, "F_slw_0.2", speaker_loss_weight=0.2))
    configs.append(_make(base, "F_cka_0.001", cka_loss_weight=0.001))
    configs.append(_make(base, "F_cka_0.05", cka_loss_weight=0.05))
    # VIB beta
    configs.append(_make(base, "F_vb_1e-5", vib_beta=1e-5))
    configs.append(_make(base, "F_vb_1e-3", vib_beta=1e-3))
    # Learning rates
    configs.append(_make(base, "F_blr_5e-6", backbone_lr=5e-6))
    configs.append(_make(base, "F_blr_2e-5", backbone_lr=2e-5))
    configs.append(_make(base, "F_hlr_5e-4", head_lr=5e-4))
    # LLRD
    configs.append(_make(base, "F_llrd_0.9", llrd_decay=0.9))
    configs.append(_make(base, "F_llrd_0.99", llrd_decay=0.99))
    # Dimensions
    configs.append(_make(base, "F_bn_128", bottleneck_dim=128))
    configs.append(_make(base, "F_bn_512", bottleneck_dim=512))
    configs.append(_make(base, "F_dh_64", glada_disc_hidden=64))
    configs.append(_make(base, "F_dh_256", glada_disc_hidden=256))

    # --- Block G: Cross-backbone ablation (6) ---
    for bb in ["xlsr-300m", "data2vec-large"]:
        configs.append(_make(base, f"G_{bb}_no_glada",
                             backbone=bb, use_glada=False))
        configs.append(_make(base, f"G_{bb}_no_vib",
                             backbone=bb, use_vib=False))
        configs.append(_make(base, f"G_{bb}_no_hier",
                             backbone=bb, use_hierarchical=False))

    # --- Block H: Interesting combos (5) ---
    configs.append(_make(base, "H_g6_dlw0.05",
                         num_groups_override=6, domain_loss_weight=0.05))
    configs.append(_make(base, "H_xlsr_blr2e-5",
                         backbone=("xlsr-300m"), backbone_lr=2e-5))
    configs.append(_make(base, "H_frz10_blr5e-6",
                         phase1_epochs=10, backbone_lr=5e-6))
    configs.append(_make(base, "H_bn512_vb1e-5",
                         bottleneck_dim=512, vib_beta=1e-5))
    configs.append(_make(base, "H_wbp_frz3",
                         backbone="wavlm-base-plus", phase1_epochs=3))

    # --- Block I: Multi-seed on wavlm-large (2) ---
    configs.append(_make(base, "I_seed123", seed=123))
    configs.append(_make(base, "I_seed456", seed=456))

    # --- Block J: IB-CAAN baseline (1) ---
    # IB-CAAN (Huang et al., 2025): single DANN + VIB, no per-group DA / speaker / CKA.
    # Tests whether backbone fine-tuning works with a single adversarial signal (1 GRL)
    # vs. GLADA's 5 simultaneous GRL signals.  If this baseline is stable in Phase 2
    # but full GLADA is not, the multi-adversarial architecture is the root cause.
    configs.append(_make(base, "J_ibcaan_baseline",
                         use_glada=False,
                         use_global_domain_disc=True,
                         use_speaker_adversary=False,
                         use_cka_loss=False))

    return configs


def generate_phase2_diagnostic_configs(
    batch_size: int = 24,
    amp_dtype: str = "float16",
    num_workers: int = 0,
) -> List[GLADAConfig]:
    """Generate 7 configs to diagnose Phase 2 instability root cause.

    All configs use short runs (5+10 epochs) to quickly identify whether
    instability comes from: (a) multiple adversarial signals, (b) the
    Phase 2 ramp itself, or (c) some other pipeline component.

    Designed to answer Dr. Abri's question: "Is the instability coming from
    the multiple adversarial signals, the loss scaling, or another part of
    the pipeline?"

    Returns:
        List of 7 GLADAConfig objects for the diagnostic sweep.
    """
    base = GLADAConfig(
        batch_size=batch_size,
        amp_dtype=amp_dtype,
        num_workers=num_workers,
        phase1_epochs=5,
        phase2_epochs=10,
    )

    configs = []

    # 1. Control: full model with Phase 2 ramp (post-bug-fix baseline)
    configs.append(_make(base, "P2_diag_control",
                         phase2_adv_warmup_epochs=3))

    # 2. No ramp: is the ramp still needed after the speaker scaling bug fix?
    configs.append(_make(base, "P2_diag_no_ramp",
                         phase2_adv_warmup_epochs=0))

    # 3. GLADA only (no speaker): isolate GLADA's contribution to instability
    configs.append(_make(base, "P2_diag_glada_only",
                         use_speaker_adversary=False,
                         phase2_adv_warmup_epochs=0))

    # 4. Speaker only (no GLADA): isolate speaker adversary's contribution
    configs.append(_make(base, "P2_diag_speaker_only",
                         use_glada=False,
                         phase2_adv_warmup_epochs=0))

    # 5. No adversarial losses: VIB + CKA only — is instability adversarial-specific?
    configs.append(_make(base, "P2_diag_no_adversarial",
                         use_glada=False,
                         use_speaker_adversary=False,
                         use_global_domain_disc=False,
                         phase2_adv_warmup_epochs=0))

    # 6. Longer ramp (5 epochs): does more gradual introduction help?
    configs.append(_make(base, "P2_diag_ramp5",
                         phase2_adv_warmup_epochs=5))

    # 7. Half-strength start: does starting at 50% instead of 0% help?
    configs.append(_make(base, "P2_diag_ramp3_start05",
                         phase2_adv_warmup_epochs=3,
                         phase2_adv_ramp_start=0.5))

    return configs


def generate_ablation_configs(base_config: GLADAConfig) -> List[GLADAConfig]:
    """Generate standard ablation configurations from a base config.

    Returns configs for:
    1. Full GLADA (all components on)
    2. -GLADA (use_glada=False) -- THE most important ablation, directly tests
       the thesis contribution
    3. -VIB (use_vib=False)
    4. -Speaker (use_speaker_adversary=False)
    5. -CKA (use_cka_loss=False)
    6. -Hierarchical (use_hierarchical=False)
    7. -RawBoost (use_rawboost=False)
    """
    configs = []

    # 1. Full GLADA (baseline: all components on)
    full = copy.deepcopy(base_config)
    full.experiment_name = "ablation_full"
    configs.append(full)

    # 2. -GLADA (removes the per-group component — key comparison)
    no_glada = copy.deepcopy(base_config)
    no_glada.use_glada = False
    no_glada.experiment_name = "ablation_no_glada"
    configs.append(no_glada)

    # 3. -VIB
    no_vib = copy.deepcopy(base_config)
    no_vib.use_vib = False
    no_vib.experiment_name = "ablation_no_vib"
    configs.append(no_vib)

    # 4. -Speaker Adversary
    no_speaker = copy.deepcopy(base_config)
    no_speaker.use_speaker_adversary = False
    no_speaker.experiment_name = "ablation_no_speaker"
    configs.append(no_speaker)

    # 5. -CKA Diversity Loss
    no_cka = copy.deepcopy(base_config)
    no_cka.use_cka_loss = False
    no_cka.experiment_name = "ablation_no_cka"
    configs.append(no_cka)

    # 6. -Hierarchical Attention
    no_hier = copy.deepcopy(base_config)
    no_hier.use_hierarchical = False
    no_hier.experiment_name = "ablation_no_hierarchical"
    configs.append(no_hier)

    # 7. -RawBoost Augmentation
    no_rawboost = copy.deepcopy(base_config)
    no_rawboost.use_rawboost = False
    no_rawboost.experiment_name = "ablation_no_rawboost"
    configs.append(no_rawboost)

    return configs


# ============================================================================
# Sweep runners
# ============================================================================

def run_sweep(
    configs: List[GLADAConfig],
    data_paths: dict,
    device: str = "cuda",
) -> dict:
    """Run multiple GLADA training configurations sequentially.

    Datasets are loaded ONCE and shared across all configs. DataLoaders are
    re-created per config because batch_size may differ. GPU memory is
    cleaned between configs.

    Args:
        configs: List of GLADAConfig objects to train.
        data_paths: Dict with keys:
            'source_ds'  — ASVspoof training dataset (already constructed)
            'target_ds'  — ITW training dataset (already constructed)
            'eval_ds'    — Primary evaluation dataset (ITW eval)
            'extra_eval' — Optional dict of name -> Dataset for extra eval sets
        device: CUDA device string.

    Returns:
        Dict mapping config names to their results.
    """
    all_results = {}
    total = len(configs)
    sweep_start = time.time()

    # Shared datasets — loaded once, reused across all configs
    source_ds = data_paths["source_ds"]
    target_ds = data_paths["target_ds"]
    eval_ds = data_paths["eval_ds"]
    extra_eval = data_paths.get("extra_eval", {})

    # Determine sweep output directory (use first config's drive_log_dir)
    sweep_dir = os.path.join(configs[0].drive_log_dir, "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_results_path = os.path.join(sweep_dir, "sweep_results.json")

    for i, cfg in enumerate(configs):
        config_name = _config_name(cfg)

        elapsed = time.time() - sweep_start
        eta = (elapsed / max(i, 1)) * (total - i) if i > 0 else 0

        print(f"\n{'='*60}")
        print(f"Config {i+1}/{total}: {config_name}")
        print(f"  backbone={cfg.backbone}, seed={cfg.seed}, "
              f"groups={cfg.num_groups}, phase1={cfg.phase1_epochs}e, "
              f"total={cfg.total_epochs}e")
        print(f"  glada={cfg.use_glada}, vib={cfg.use_vib}, "
              f"spk={cfg.use_speaker_adversary}, cka={cfg.use_cka_loss}, "
              f"hier={cfg.use_hierarchical}, rb={cfg.use_rawboost}")
        print(f"  Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")
        print(f"{'='*60}")

        set_seed(cfg.seed)

        # Re-create DataLoaders per config (batch_size might differ)
        loaders = create_dataloaders(cfg, source_ds, target_ds, eval_ds, extra_eval)

        # Detect number of speakers from source dataset
        num_speakers = None
        if hasattr(source_ds, "speaker_to_id"):
            num_speakers = len(source_ds.speaker_to_id)

        try:
            # Build and train
            trainer = GLADATrainer(
                config=cfg,
                source_loader=loaders["source"],
                target_loader=loaders["target"],
                eval_loader=loaders["eval"],
                extra_eval_loaders={k: v for k, v in loaders.items()
                                    if k not in ("source", "target", "eval")},
                num_speakers=num_speakers,
            )
            results = trainer.train()
            all_results[config_name] = results
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[config_name] = {"error": str(e)}

        # Save results incrementally (in case of Colab disconnection)
        _save_results = {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                         for k, v in all_results.items()}
        with open(sweep_results_path, "w") as f:
            json.dump(_save_results, f, indent=2, default=str)

        # --- GPU cleanup between configs ---
        if "trainer" in dir():
            del trainer
        torch.cuda.empty_cache()
        gc.collect()

    total_time = (time.time() - sweep_start) / 60
    print(f"\n{'='*60}")
    print(f"Sweep complete: {total} configs in {total_time:.1f} min")
    print(f"Results saved to {sweep_results_path}")
    print(f"{'='*60}")

    return all_results


def _stability_score(history: list) -> float:
    """Compute stability-aware EER score for sweep ranking.

    Instead of min(EER) — which rewards one lucky epoch before collapse —
    use the last Phase 2 EER plus a penalty for upward trend (degradation).

        score = last_p2_eer + 0.5 * max(0, p2_trend)

    where p2_trend = last_p2_eer - first_p2_eer (positive = degrading).
    This penalises configs that start well but collapse, which is exactly
    the Phase 2 instability pattern observed in the Apr 2026 sweep.

    Args:
        history: List of epoch log dicts from GLADATrainer.train().

    Returns:
        float: Stability-aware score (lower = better).  inf if no Phase 2 data.
    """
    p2_eers = []
    for entry in history:
        if entry.get("phase") == 2:
            itw = entry.get("eval", {}).get("ITW", {})
            eer = itw.get("eer")
            if eer is not None:
                p2_eers.append(eer)

    if not p2_eers:
        return float("inf")

    last_eer = p2_eers[-1]
    trend = p2_eers[-1] - p2_eers[0]  # positive = getting worse
    return last_eer + 0.5 * max(0.0, trend)


def run_two_phase_sweep(
    configs: List[GLADAConfig],
    data_paths: dict,
    device: str = "cuda",
    screening_phase2_epochs: int = 10,
    top_k: int = 15,
    full_phase2_epochs: int = 45,
) -> dict:
    """Two-phase sweep: screen many configs quickly, then train top-K fully.

    Phase 1 (Screening):
        All configs trained with reduced phase2_epochs (default 10, so ~15
        total epochs with phase1=5). Ranks configs by best_eer.

    Phase 2 (Full Training):
        Top-K configs from screening re-trained from scratch with full
        phase2_epochs (default 45, so ~50 total epochs). Seeds are preserved.

    Args:
        configs: Full list of screening configs.
        data_paths: Dataset dict (same as run_sweep).
        device: CUDA device string.
        screening_phase2_epochs: Phase-2 epochs for screening (default 10).
        top_k: Number of top configs to advance to full training.
        full_phase2_epochs: Phase-2 epochs for full training (default 45).

    Returns:
        Dict with keys:
            'screening': results from phase 1 (all configs)
            'full': results from phase 2 (top-K configs)
            'ranking': sorted list of (config_name, eer) from screening
    """
    total = len(configs)
    print(f"{'#'*60}")
    print(f"# TWO-PHASE SWEEP")
    print(f"# Phase 1: Screen {total} configs "
          f"(~{screening_phase2_epochs + 5} epochs each)")
    print(f"# Phase 2: Top {top_k} configs "
          f"(~{full_phase2_epochs + 5} epochs each)")
    print(f"{'#'*60}\n")

    # ----------------------------------------------------------------
    # Phase 1: Screening — short training on all configs
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 1: SCREENING ({total} configs)")
    print(f"{'='*60}\n")

    screening_configs = []
    original_phase2 = {}  # save original phase2_epochs per config
    for cfg in configs:
        c = copy.deepcopy(cfg)
        original_phase2[cfg.experiment_name] = cfg.phase2_epochs
        c.phase2_epochs = screening_phase2_epochs
        c.__post_init__()  # recompute total_epochs
        c.experiment_name = f"screen_{cfg.experiment_name}"
        screening_configs.append(c)

    screening_results = run_sweep(screening_configs, data_paths, device)

    # ----------------------------------------------------------------
    # Rank by stability-aware score (not min EER, which rewards lucky epochs)
    # ----------------------------------------------------------------
    ranking = []
    for name, res in screening_results.items():
        if "error" in res:
            score = float("inf")
        else:
            history = res.get("history", [])
            score = _stability_score(history) if history else res.get("best_eer", float("inf"))
        ranking.append((name, score))

    ranking.sort(key=lambda x: x[1])

    print(f"\n{'='*60}")
    print(f"SCREENING RANKING — stability score (top {min(top_k, len(ranking))} of {len(ranking)}):")
    print(f"{'='*60}")
    for rank, (name, score) in enumerate(ranking[:top_k], 1):
        marker = " <<" if rank <= top_k else ""
        print(f"  {rank:3d}. {name:<50s} score={score:.2f}%{marker}")
    if len(ranking) > top_k:
        print(f"  ... ({len(ranking) - top_k} more configs not advancing)")

    # ----------------------------------------------------------------
    # Phase 2: Full training on top-K configs
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 2: FULL TRAINING (top {min(top_k, len(ranking))} configs)")
    print(f"{'='*60}\n")

    # Map screening names back to original configs
    screen_to_orig = {}
    for cfg, scfg in zip(configs, screening_configs):
        screen_to_orig[_config_name(scfg)] = cfg

    full_configs = []
    for name, eer in ranking[:top_k]:
        orig = screen_to_orig.get(name)
        if orig is None:
            print(f"  WARNING: Could not find original config for {name}, skipping")
            continue
        c = copy.deepcopy(orig)
        c.phase2_epochs = full_phase2_epochs
        c.__post_init__()
        c.experiment_name = f"full_{orig.experiment_name}"
        full_configs.append(c)

    full_results = run_sweep(full_configs, data_paths, device)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    final_ranking = []
    for name, res in full_results.items():
        if "error" in res:
            score = float("inf")
        else:
            history = res.get("history", [])
            score = _stability_score(history) if history else res.get("best_eer", float("inf"))
        final_ranking.append((name, score))
    final_ranking.sort(key=lambda x: x[1])

    print(f"\n{'#'*60}")
    print(f"# FINAL RESULTS — stability score (Phase 2, full training)")
    print(f"{'#'*60}")
    for rank, (name, score) in enumerate(final_ranking, 1):
        print(f"  {rank:3d}. {name:<50s} score={score:.2f}%")

    return {
        "screening": screening_results,
        "full": full_results,
        "ranking": ranking,
        "final_ranking": final_ranking,
    }
