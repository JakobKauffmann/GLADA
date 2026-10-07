"""run_pgvib.py — CLI runner for the May 19 pg-VIB strengthening campaign.

Single entry point for every pg-VIB experiment in the campaign.  Builds a
``GLADAConfig`` from CLI flags, instantiates either ``GLADAModel`` (full GLADA
baseline) or ``PerGroupVIBModel`` (the per-group bottleneck variant that lands
the 11.67% ITW EER paper headline), wires the standard ASVspoof19 source /
In-the-Wild target / ITW eval loaders, and trains via the package's
``GLADATrainer``.  All logic lives in ``Coding/glada/``; this script is a thin
orchestrator (Rule 13: no model/training logic inlined in scripts or
notebooks).

Resume contract
---------------
On launch the script checks ``<output_dir>/history.json``:
  - If it exists and contains ``num_epochs >= total_epochs``, exits 0 (skip).
  - If it exists but is partial, ``GLADATrainer`` resumes from the latest
    checkpoint in ``output_dir`` via the ``resume_from`` argument.
  - Otherwise, training starts from scratch.

This makes the notebook safely re-runnable after Colab disconnects.

Logging
-------
- stdout / stderr: redirected by the launcher (notebook) to a Drive-backed
  log file via ``subprocess.Popen(..., stdout=open(log_path,'w'),
  stderr=subprocess.STDOUT)``.
- ``output_dir/history.json``: per-epoch metrics (train losses, eval EER
  on ITW + extra eval sets).  Written incrementally by ``BaseTrainer``.
- ``output_dir/eval.json``: final best-checkpoint evaluation on every loader.
- ``output_dir/config.json``: full config snapshot (asdict of GLADAConfig
  plus the CLI args).

Reproducibility
---------------
``set_seed(args.seed)`` is called before any data or model construction.
``num_workers=0`` (default in GLADAConfig) keeps DataLoader RNG deterministic
on Colab.  CUDA's nondeterministic kernels are not forced to deterministic
mode (would slow training ~2x); we accept the resulting <0.3pp EER jitter,
which is well below the seed-to-seed variance the campaign is measuring.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path


def _parse_groups_tuple(s: str) -> tuple:
    """Parse '1,2' -> (1, 2); empty string -> ()."""
    if not s:
        return ()
    return tuple(int(x) for x in s.split(",") if x.strip())


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one pg-VIB / GLADA variant for the May 19 campaign.",
    )

    # Core experiment knobs.
    p.add_argument("--output", required=True,
                   help="Drive-backed output dir.  Subfiles: history.json, "
                        "eval.json, config.json, checkpoints/")
    p.add_argument("--backbone", default="wavlm-base-plus",
                   choices=["wavlm-base-plus", "wavlm-large",
                            "xlsr-300m", "data2vec-large", "wav2vec2-robust"],
                   help="SSL backbone key into BACKBONE_REGISTRY.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed; campaign uses 42/123/456.")
    p.add_argument("--phase1_epochs", type=int, default=3,
                   help="Frozen-backbone warmup epochs.  Paper pg-VIB run "
                        "uses 3; phase_pgvib_g0only baseline.")
    p.add_argument("--phase2_epochs", type=int, default=17,
                   help="Fine-tuning epochs.  Paper pg-VIB run uses 17 for "
                        "a 20-epoch total at batch_size=64.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Per-domain batch (source + target each).  A100 fits "
                        "64 for WavLM-Base-Plus; drop to 24 for WavLM-Large.")
    p.add_argument("--effective_batch_size", type=int, default=384,
                   help="Effective batch via grad accumulation; computed in "
                        "__post_init__ as effective / (batch * 2).")

    # Per-group VIB controls.
    p.add_argument("--model", choices=["GLADAModel", "PerGroupVIBModel"],
                   default="PerGroupVIBModel",
                   help="Model class.  Default PerGroupVIBModel matches the "
                        "11.67% paper headline.")
    p.add_argument("--vib_disabled_groups", default="1,2",
                   help="Comma-sep group indices where z_k = mu_k and KL_k=0. "
                        "Default '1,2' = g0-only VIB (WavLM-Base-Plus G=3).  "
                        "Empty string = all groups active.")
    p.add_argument("--vib_beta", type=float, default=0.01,
                   help="VIB KL weight.  Effective per-group beta is "
                        "vib_beta / G (uniform-mean aggregation).")

    # Future-Work (b): number-of-groups.
    p.add_argument("--num_groups_override", type=int, default=0,
                   help="Override BACKBONE_REGISTRY's default num_groups.  "
                        "0 = registry default (3 for WavLM-BP, 4 for "
                        "WavLM-L).  Paper Future Work (b) nominates G=2.")

    # Future-Work (c): MUSAN + RIR.
    p.add_argument("--musan_path", default="",
                   help="Path to MUSAN corpus root; empty = disabled.")
    p.add_argument("--musan_prob", type=float, default=0.5)
    p.add_argument("--rir_path", default="",
                   help="Path to RIR_NOISES/simulated_rirs root; empty = "
                        "disabled.")
    p.add_argument("--rir_prob", type=float, default=0.5)

    # Stretch: confidence-aware adversarial weighting.
    p.add_argument("--confidence_aware_adv_weight", action="store_true",
                   help="Enable IB-CAAN-style confidence-margin weighting on "
                        "the per-group GLADA loss.")

    # Knob plumbing for sanity / debug runs.
    p.add_argument("--experiment_name", default="may19",
                   help="Used by GLADAConfig.get_run_dir(); harmless when "
                        "we set drive_log_dir/run_dir explicitly via --output.")
    p.add_argument("--run_id", default=None,
                   help="Defaults to the basename of --output.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: 1 epoch each phase, batch_size=4, "
                        "effective_batch_size=8.  For wiring checks.")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.phase1_epochs = 1
        args.phase2_epochs = 1
        args.batch_size = 4
        args.effective_batch_size = 8

    history_path = output_dir / "history.json"
    expected_epochs = args.phase1_epochs + args.phase2_epochs

    # === Resume short-circuit ===========================================
    if history_path.exists():
        try:
            hist = json.loads(history_path.read_text())
            done_epochs = len(hist) if isinstance(hist, list) else hist.get(
                "num_epochs", 0)
            if done_epochs >= expected_epochs:
                print(f"[run_pgvib] {history_path} already has "
                      f"{done_epochs} epochs (>= {expected_epochs}); "
                      f"skipping run.")
                return 0
            print(f"[run_pgvib] partial history ({done_epochs} epochs); "
                  f"will resume.")
        except Exception as e:
            print(f"[run_pgvib] history.json present but unreadable "
                  f"({e}); restarting from scratch.")

    # Heavy imports deferred until after the resume short-circuit so the
    # script exits quickly when there's nothing to do.
    import torch
    from glada import GLADAConfig, set_seed
    from glada.datasets import (
        ASVspoofDatasetWithSpeaker,
        ASVspoofEvalDataset,
        InTheWildDataset,
        create_dataloaders,
    )
    from glada.augmentation import RawBoost
    from glada.augmentation.musan_rir import CombinedAugmenter
    from glada.models.glada_model import GLADAModel, PerGroupVIBModel
    from glada.trainers import GLADATrainer

    set_seed(args.seed)

    # === Dataset paths (Colab canonical) ================================
    # These are the same locations the april13 notebooks extract into; we
    # don't repeat the extraction here — the orchestrator notebook is
    # responsible for verifying they exist before launching.
    ASV_PROTO = "/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/" \
        "ASVspoof2019.LA.cm.train.trn.txt"
    ASV_AUDIO = "/content/data/asvspoof/LA/ASVspoof2019_LA_train/flac"
    ASV_EVAL_PROTO = "/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/" \
        "ASVspoof2019.LA.cm.eval.trl.txt"
    ASV_EVAL_AUDIO = "/content/data/asvspoof/LA/ASVspoof2019_LA_eval/flac"
    ITW_META = "/content/data/itw/release_in_the_wild/meta.csv"
    ITW_AUDIO = "/content/data/itw/release_in_the_wild"
    for path in [ASV_PROTO, ASV_AUDIO, ASV_EVAL_PROTO, ASV_EVAL_AUDIO,
                 ITW_META, ITW_AUDIO]:
        if not os.path.exists(path):
            print(f"[run_pgvib] FATAL: missing required dataset path {path}",
                  file=sys.stderr)
            return 2

    # === Build config ===================================================
    run_id = args.run_id or output_dir.name
    cfg = GLADAConfig(
        backbone=args.backbone,
        seed=args.seed,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        vib_beta=args.vib_beta,
        num_groups_override=args.num_groups_override,
        musan_path=args.musan_path,
        musan_prob=args.musan_prob,
        rir_path=args.rir_path,
        rir_prob=args.rir_prob,
        confidence_aware_adv_weight=args.confidence_aware_adv_weight,
        experiment_name=args.experiment_name,
        run_id=run_id,
        drive_log_dir=str(output_dir.parent),
    )

    # === Augmenter selection ===========================================
    # CombinedAugmenter detects MUSAN/RIR presence via path existence; it
    # gracefully no-ops if either is missing.  When BOTH paths are empty
    # we fall back to the legacy RawBoost class so we don't pay the FFT-
    # convolution overhead for samples that aren't being reverbed.
    if args.musan_path or args.rir_path:
        augmenter = CombinedAugmenter(cfg)
        print(f"[run_pgvib] CombinedAugmenter: rawboost={augmenter.use_rawboost}, "
              f"musan={'on' if augmenter.musan_augmenter else 'off'}, "
              f"rir={'on' if augmenter.rir_augmenter else 'off'}")
    else:
        augmenter = RawBoost if cfg.use_rawboost else None
        print(f"[run_pgvib] augmenter: "
              f"{'RawBoost-only' if augmenter else 'NONE'}")

    # === Datasets =======================================================
    source_ds = ASVspoofDatasetWithSpeaker(
        protocol_path=ASV_PROTO,
        audio_dir=ASV_AUDIO,
        config=cfg,
        augmenter=augmenter,
    )
    target_ds = InTheWildDataset(
        meta_csv_path=ITW_META, audio_dir=ITW_AUDIO,
        config=cfg, use_labels=False,
    )
    eval_ds = InTheWildDataset(
        meta_csv_path=ITW_META, audio_dir=ITW_AUDIO,
        config=cfg, use_labels=True,
    )
    extra_eval = {
        "2019_LA": ASVspoofEvalDataset(
            protocol_path=ASV_EVAL_PROTO, audio_dir=ASV_EVAL_AUDIO,
            config=cfg,
        ),
    }
    num_speakers = getattr(source_ds, "num_speakers", None)
    loaders = create_dataloaders(
        cfg, source_ds, target_ds, eval_ds, extra_eval)

    # === Model class selection =========================================
    if args.model == "PerGroupVIBModel":
        model_cls = PerGroupVIBModel
        model_kwargs = {
            "vib_disabled_groups": _parse_groups_tuple(args.vib_disabled_groups),
        }
        print(f"[run_pgvib] PerGroupVIBModel "
              f"vib_disabled_groups={model_kwargs['vib_disabled_groups']}")
    else:
        model_cls = GLADAModel
        model_kwargs = {}

    # === Train ==========================================================
    resume_from = None
    last_ckpt = sorted(output_dir.glob("ckpt_ep*.pt"))
    if last_ckpt:
        resume_from = str(last_ckpt[-1])
        print(f"[run_pgvib] resume_from={resume_from}")

    trainer = GLADATrainer(
        config=cfg,
        source_loader=loaders["source"],
        target_loader=loaders["target"],
        eval_loader=loaders["eval"],
        extra_eval_loaders={k: v for k, v in loaders.items()
                            if k not in ("source", "target", "eval")},
        num_speakers=num_speakers,
        resume_from=resume_from,
        model_class=model_cls,
        model_kwargs=model_kwargs,
    )

    # Persist config snapshot before training (so partial runs still tell
    # us how they were configured).
    config_dump = dataclasses.asdict(cfg)
    config_dump["__model_class__"] = args.model
    config_dump["__model_kwargs__"] = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in model_kwargs.items()
    }
    config_dump["__cli_args__"] = vars(args)
    (output_dir / "config.json").write_text(json.dumps(config_dump, indent=2))

    t_start = time.time()
    results = trainer.train()
    elapsed_min = (time.time() - t_start) / 60.0

    # === Save final results ============================================
    eval_summary = {
        "best_eer": results.get("best_eer"),
        "best_epoch": results.get("best_epoch"),
        "eval_metrics": results.get("eval_metrics"),
        "elapsed_min": elapsed_min,
    }
    (output_dir / "eval.json").write_text(json.dumps(eval_summary, indent=2,
                                                     default=str))
    print(f"[run_pgvib] DONE in {elapsed_min:.1f} min; "
          f"best ITW EER = {results.get('best_eer')} "
          f"at epoch {results.get('best_epoch')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
