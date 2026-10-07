"""Shared cell builders for the Phase A/B/C notebooks.

The four build scripts (build_phase_a_axes.py, build_phase_a_leaveout.py,
build_phase_b_confirm.py, build_phase_c_probe.py) each generate a fully
self-contained Colab notebook that can be Run-All'd without depending on
state from other notebooks. The boilerplate (env setup, package copy,
dataset extraction, imports, the inline PerGroupVIBModel definition with
new vib_disabled_groups / vib_betas flags) is identical across them and
lives here.

Pattern follows build_phase_vib_stagger.py: cells are JSON dicts with
"source" lists; the build scripts assemble a CELLS list and dump to
.ipynb with json.dump.
"""

from __future__ import annotations


def md(*lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" if not l.endswith("\n") else l for l in lines],
    }


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" if not l.endswith("\n") else l for l in lines],
    }


def env_setup_cells():
    """Sections 1-3: environment, GLADA package copy, dataset extraction."""
    return [
        md("## 1. Environment Setup"),
        code(
            "!pip install -q transformers soundfile librosa scikit-learn tqdm safetensors",
            "",
            "from google.colab import drive",
            "drive.mount('/content/drive')",
            "",
            "import torch",
            "if torch.cuda.is_available():",
            "    gpu = torch.cuda.get_device_properties(0)",
            "    print(f'GPU: {gpu.name} ({gpu.total_memory / 1e9:.1f} GB)')",
            "else:",
            "    raise RuntimeError('No GPU detected')",
        ),
        md("## 2. Copy GLADA Package to Local Runtime"),
        code(
            "import shutil, os, sys",
            "",
            "CANDIDATES = [",
            "    \"/content/drive/Othercomputers/Jakob's MacBook Air/Thesis/298-deliverables/glada-code/glada\",",
            "    '/content/drive/MyDrive/Jakob/glada_pkg/glada',",
            "    '/content/drive/MyDrive/Jakob/glada_pkg',",
            "    '/content/drive/MyDrive/Jakob/glada',",
            "]",
            "DRIVE_PKG = next((c for c in CANDIDATES if os.path.exists(os.path.join(c, '__init__.py'))), None)",
            "if DRIVE_PKG is None:",
            "    raise FileNotFoundError(",
            "        'No glada/__init__.py found in candidates. Run:\\n'",
            "        \"  !find /content/drive -name '__init__.py' -path '*/glada/*' 2>/dev/null | head\\n\"",
            "        'and add the parent directory to CANDIDATES above.'",
            "    )",
            "",
            "LOCAL_PKG = '/content/glada'",
            "if os.path.exists(LOCAL_PKG): shutil.rmtree(LOCAL_PKG)",
            "shutil.copytree(DRIVE_PKG, LOCAL_PKG)",
            "",
            "for mod in list(sys.modules):",
            "    if mod == 'glada' or mod.startswith('glada.'): del sys.modules[mod]",
            "sys.path = [p for p in sys.path if 'glada' not in p]",
            "if '/content' not in sys.path: sys.path.insert(0, '/content')",
            "print(f'Package source: {DRIVE_PKG}\\nCopied to:      {LOCAL_PKG}')",
        ),
        md("## 3. Datasets (ASVspoof 2019 LA + ITW)"),
        code(
            "import tarfile",
            "",
            "def extract_dataset(tar_path, extract_to, name):",
            "    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:",
            "        print(f'{name}: already extracted to {extract_to}'); return",
            "    os.makedirs(extract_to, exist_ok=True)",
            "    print(f'{name}: extracting {tar_path} -> {extract_to} ...')",
            "    with tarfile.open(tar_path) as t: t.extractall(extract_to)",
            "    print(f'{name}: done.')",
            "",
            "extract_dataset('/content/drive/MyDrive/Jakob/Datasets/LA.tar', '/content/data/asvspoof', 'ASVspoof 2019 LA')",
            "extract_dataset('/content/drive/MyDrive/Jakob/Datasets/release_in_the_wild.tar', '/content/data/itw', 'In-the-Wild')",
            "",
            "ASV_PROTO      = '/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'",
            "ASV_AUDIO      = '/content/data/asvspoof/LA/ASVspoof2019_LA_train/flac'",
            "ASV_EVAL_PROTO = '/content/data/asvspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'",
            "ASV_EVAL_AUDIO = '/content/data/asvspoof/LA/ASVspoof2019_LA_eval/flac'",
            "ITW_META       = '/content/data/itw/release_in_the_wild/meta.csv'",
            "ITW_AUDIO      = '/content/data/itw/release_in_the_wild'",
            "for p in [ASV_PROTO, ASV_AUDIO, ASV_EVAL_PROTO, ITW_META]:",
            "    assert os.path.exists(p), f'Missing: {p}'",
            "print('Datasets verified.')",
        ),
    ]


def imports_cells():
    """Section 4: imports."""
    return [
        md("## 4. Imports"),
        code(
            "from glada import GLADAConfig, BACKBONE_REGISTRY, set_seed",
            "from glada.datasets import (",
            "    ASVspoofDatasetWithSpeaker,",
            "    ASVspoofEvalDataset,",
            "    InTheWildDataset,",
            "    create_dataloaders,",
            ")",
            "from glada.augmentation import RawBoost",
            "from glada.models import GLADAModel",
            "from glada.models.vib import VIBBottleneck",
            "from glada.models.grl import GradientReversal",
            "from glada.models.discriminators import GroupedLayerDomainDiscriminators",
            "from glada.trainers import GLADATrainer",
            "",
            "import copy, gc, json, time",
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "import numpy as np",
            "",
            "print('Imports OK')",
        ),
    ]


def model_class_cells():
    """Section 5: inline PerGroupVIBModel with vib_disabled_groups + vib_betas.

    Backwards-compatible: with no extra kwargs, behavior matches the original
    phase_vib_stagger inline class. New kwargs:
      vib_disabled_groups: tuple[int] of group indices where z_k = mu_k
        (deterministic) and kl_k = 0. The encoder still runs (so dimensions
        match downstream), but no stochastic sampling and no KL pressure.
      vib_betas: tuple[float] of length K with per-group beta multipliers.
        If None, kl_total is the simple mean across groups (matching the
        original implementation, which gives effective beta/G per group).
        If provided, kl_total = sum_k vib_betas[k] * kl_k. Note that the
        trainer multiplies kl_total by config.vib_beta a second time, so
        the effective per-group beta is config.vib_beta * vib_betas[k].
    """
    return [
        md(
            "## 5. INLINE: `PerGroupVIBModel` (with leave-one-out + per-group beta flags)",
            "",
            "Subclasses `GLADAModel`. Adds K `VIBBottleneck` modules (one per group), each compressing a `(B, 2D)` group ASP output to `(B, bottleneck_dim)`. The per-group discriminator then receives the compressed `z_k` rather than the raw `g_k`.",
            "",
            "**New flags vs. the phase_vib_stagger version:**",
            "- `vib_disabled_groups: tuple = ()` — for these group indices, `z_k = mu_k` (deterministic, no reparam sampling) and `kl_k = 0`. Used by `phase_a_leaveout` to ablate one group's compression at a time.",
            "- `vib_betas: tuple = None` — per-group beta override. If provided, `kl_total = sum_k vib_betas[k] * kl_k` instead of the default uniform mean. Used by `phase_b_confirm` to apply scaled betas based on Phase A leave-one-out results.",
            "",
            "Backwards compatibility: with no extra kwargs, behavior matches the original phase_vib_stagger inline class.",
        ),
        code(
            "class PerGroupVIBModel(GLADAModel):",
            "    \"\"\"GLADAModel with one VIBBottleneck per group, applied BEFORE the",
            "    per-group discriminators. The hierarchical pool + global VIB are",
            "    replaced by a simple linear projection over the concatenated",
            "    per-group compressed representations.",
            "",
            "    Args (in addition to GLADAConfig):",
            "        vib_disabled_groups: tuple of group indices to disable VIB on",
            "            (z_k = mu_k, kl_k = 0). Used for leave-one-out ablation.",
            "        vib_betas: tuple of length K with per-group beta multipliers.",
            "            None -> uniform mean (original behavior). Tuple -> sum of",
            "            beta-weighted KLs.",
            "    \"\"\"",
            "",
            "    def __init__(self, config, vib_disabled_groups=(), vib_betas=None):",
            "        super().__init__(config)",
            "        K = config.num_groups",
            "        D = config.hidden_dim",
            "        bd = config.bottleneck_dim",
            "        self.group_vibs = nn.ModuleList([",
            "            VIBBottleneck(2 * D, bd) for _ in range(K)",
            "        ])",
            "        self.group_compressed_proj = nn.Linear(K * bd, bd)",
            "        if self.glada_disc is not None:",
            "            self.glada_disc = GroupedLayerDomainDiscriminators(",
            "                num_groups=K,",
            "                input_dim=bd,",
            "                hidden_dim=config.glada_disc_hidden,",
            "            )",
            "",
            "        # Phase A controls.",
            "        for k in vib_disabled_groups:",
            "            assert 0 <= k < K, f'vib_disabled_groups index {k} not in [0,{K})'",
            "        self._vib_disabled = set(vib_disabled_groups)",
            "        if vib_betas is not None:",
            "            assert len(vib_betas) == K, f'vib_betas must have len {K}, got {len(vib_betas)}'",
            "            self.register_buffer('_vib_betas',",
            "                torch.tensor(vib_betas, dtype=torch.float32))",
            "        else:",
            "            self._vib_betas = None",
            "",
            "    def forward(self, input_values, attention_mask=None):",
            "        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())",
            "        if backbone_frozen:",
            "            with torch.no_grad():",
            "                outputs = self.backbone(input_values, attention_mask=attention_mask, output_hidden_states=True)",
            "        else:",
            "            outputs = self.backbone(input_values, attention_mask=attention_mask, output_hidden_states=True)",
            "        hidden_states = list(outputs.hidden_states)",
            "",
            "        if attention_mask is not None:",
            "            input_lengths = attention_mask.sum(dim=1)",
            "            output_lengths = ((input_lengths - 1) // 320) + 1",
            "            T = hidden_states[0].shape[1]",
            "            frame_mask = (torch.arange(T, device=attention_mask.device).unsqueeze(0) < output_lengths.unsqueeze(1))",
            "        else:",
            "            frame_mask = None",
            "",
            "        hidden_states = self.layer_norms(hidden_states)",
            "        _, cka_loss, group_outputs = self.pooling(hidden_states, frame_mask=frame_mask)",
            "        if not self.config.use_cka_loss:",
            "            cka_loss = torch.tensor(0.0, device=group_outputs[0].device)",
            "",
            "        # Per-group VIB with leave-one-out + per-group beta support.",
            "        z_groups = []",
            "        kl_per_group = []",
            "        for k, g_k in enumerate(group_outputs):",
            "            if k in self._vib_disabled:",
            "                # Disabled: deterministic z_k = mu_k, kl_k = 0.",
            "                z_k, _ = self.group_vibs[k](g_k, training=False)",
            "                kl_k = torch.zeros((), device=g_k.device)",
            "            else:",
            "                z_k, kl_k = self.group_vibs[k](g_k, training=self.training)",
            "            z_groups.append(z_k)",
            "            kl_per_group.append(kl_k)",
            "",
            "        # Aggregate KL across groups.",
            "        if self._vib_betas is not None:",
            "            # Beta-weighted sum (per-group beta).",
            "            kl_total = sum(self._vib_betas[k] * kl_per_group[k] for k in range(len(group_outputs)))",
            "        else:",
            "            # Uniform mean (original behavior; effective per-group beta = config.vib_beta / G).",
            "            kl_total = sum(kl_per_group) / max(len(group_outputs), 1)",
            "",
            "        z_concat = torch.cat(z_groups, dim=-1)",
            "        z = self.group_compressed_proj(z_concat)",
            "",
            "        spoof_logits = self.task_classifier(z)",
            "        result = {",
            "            'spoof_logits': spoof_logits,",
            "            'kl_loss': kl_total,",
            "            'cka_loss': cka_loss,",
            "            'features': z,",
            "        }",
            "",
            "        if self.glada_disc is not None:",
            "            result['glada_domain_logits'] = self.glada_disc(z_groups, self.domain_grl_alpha)",
            "            result['group_outputs'] = z_groups",
            "",
            "        if self.domain_disc is not None:",
            "            z_domain = GradientReversal.apply(z, self.domain_grl_alpha)",
            "            result['domain_logits'] = self.domain_disc(z_domain)",
            "",
            "        if self.speaker_adversary is not None:",
            "            z_speaker = GradientReversal.apply(z, self.speaker_grl_alpha)",
            "            result['speaker_logits'] = self.speaker_adversary(z_speaker)",
            "",
            "        return result",
            "",
            "print('PerGroupVIBModel defined (with vib_disabled_groups + vib_betas).')",
        ),
    ]


def datasets_cells(use_rawboost: bool, source_var: str = "source_ds"):
    """Section 8: dataset construction. Toggle RawBoost on/off via use_rawboost."""
    if use_rawboost:
        augmenter_line = "    augmenter=RawBoost,  # RawBoost only — matches phase_vib_stagger reference"
    else:
        augmenter_line = "    augmenter=None,  # NO augmentation — Phase A1 / RawBoost-off probe"
    return [
        md(f"## 8. Datasets (RawBoost={'ON' if use_rawboost else 'OFF'})"),
        code(
            "base_for_data = make_config('_data_setup')",
            "",
            f"{source_var} = ASVspoofDatasetWithSpeaker(",
            "    protocol_path=ASV_PROTO,",
            "    audio_dir=ASV_AUDIO,",
            "    config=base_for_data,",
            augmenter_line,
            ")",
            "target_ds = InTheWildDataset(",
            "    meta_csv_path=ITW_META, audio_dir=ITW_AUDIO,",
            "    config=base_for_data, use_labels=False,",
            ")",
            "eval_ds = InTheWildDataset(",
            "    meta_csv_path=ITW_META, audio_dir=ITW_AUDIO,",
            "    config=base_for_data, use_labels=True,",
            ")",
            "extra_eval = {",
            "    '2019_LA': ASVspoofEvalDataset(",
            "        protocol_path=ASV_EVAL_PROTO, audio_dir=ASV_EVAL_AUDIO, config=base_for_data,",
            "    ),",
            "}",
            f"num_speakers = len({source_var}.speaker_to_id) if hasattr({source_var}, 'speaker_to_id') else None",
            f"print(f'Source: {{len({source_var}):,}} ({{num_speakers}} speakers)')",
            "print(f'Target: {len(target_ds):,}, Eval: {len(eval_ds):,}, 2019 LA: {len(extra_eval[\"2019_LA\"]):,}')",
        ),
    ]


def make_config_cell(default_backbone: str = "wavlm-base-plus", drive_subdir: str = "phase_a"):
    """Section 7: make_config helper + epoch budget. Drives the drive log dir name."""
    return [
        md("## 7. Configuration Helper"),
        code(
            "USE_FULL_EPOCHS = False",
            "PHASE1 = 5  if USE_FULL_EPOCHS else 3",
            "PHASE2 = 45 if USE_FULL_EPOCHS else 17",
            "",
            f"DRIVE_LOG_DIR = '/content/drive/MyDrive/Jakob/Experiments/{drive_subdir}'",
            "os.makedirs(DRIVE_LOG_DIR, exist_ok=True)",
            "",
            f"def make_config(name, backbone='{default_backbone}', seed=42):",
            "    cfg = GLADAConfig(",
            "        experiment_name=name,",
            "        backbone=backbone,",
            "        seed=seed,",
            "        batch_size=64,  # FIXED for apples-to-apples vs 11.67% reference",
            "        num_workers=0,",
            "        phase1_epochs=PHASE1,",
            "        phase2_epochs=PHASE2,",
            "        drive_log_dir=DRIVE_LOG_DIR,",
            "        run_id=time.strftime('%Y%m%d_%H%M%S'),",
            "    )",
            "    return cfg",
        ),
    ]


def variant_sweep_cells(extra_md_intro: str = ""):
    """Section 9: the custom sweep loop with PerGroupVIBModel kwargs support.

    The variant spec dict supports:
      model_class: 'GLADAModel' or 'PerGroupVIBModel'
      staggered:   bool, use StaggeredAlphaTrainer
      backbone:    optional override (defaults to make_config default)
      model_kwargs: dict of kwargs to pass to ModelCls(config, **model_kwargs)
      config_overrides: dict of GLADAConfig attrs to override per variant
    """
    intro_lines = [
        "## 9. Custom Sweep Loop",
        "",
        "Mirrors `phase_vib_stagger`'s pattern but threads `model_kwargs` through the variant trainer factory so each variant can set `vib_disabled_groups` / `vib_betas` independently. Per-variant config overrides (e.g., backbone, vib_beta) are also supported.",
    ]
    if extra_md_intro:
        intro_lines.extend(["", extra_md_intro])

    return [
        md(*intro_lines),
        code(
            "MODEL_CLASSES = {'GLADAModel': GLADAModel, 'PerGroupVIBModel': PerGroupVIBModel}",
            "",
            "all_results = {}",
            "sweep_start = time.time()",
            "results_path = os.path.join(DRIVE_LOG_DIR, RESULTS_FILENAME)",
            "",
            "for i, (name, spec) in enumerate(VARIANTS.items()):",
            "    elapsed = (time.time() - sweep_start) / 60",
            "    print(f'\\n{\"=\"*60}\\nVariant {i+1}/{len(VARIANTS)}: {name}\\n  elapsed: {elapsed:.1f} min\\n{\"=\"*60}')",
            "",
            "    backbone = spec.get('backbone', None)",
            "    cfg_kwargs = dict()",
            "    if backbone is not None:",
            "        cfg_kwargs['backbone'] = backbone",
            "    cfg = make_config(name, **cfg_kwargs)",
            "    for attr, val in spec.get('config_overrides', {}).items():",
            "        setattr(cfg, attr, val)",
            "    # If batch_size or effective_batch_size was overridden, re-derive",
            "    # grad_accum_steps (matches GLADAConfig.__post_init__ formula).",
            "    _bs_overrides = {'batch_size', 'effective_batch_size'} & set(spec.get('config_overrides', {}))",
            "    if _bs_overrides:",
            "        cfg.grad_accum_steps = max(1, cfg.effective_batch_size // (cfg.batch_size * 2))",
            "        print(f'  [config] batch_size={cfg.batch_size}, effective_batch_size={cfg.effective_batch_size}, grad_accum_steps={cfg.grad_accum_steps}')",
            "    set_seed(cfg.seed)",
            "",
            "    # If this variant changes RawBoost, rebuild source_ds. Otherwise reuse.",
            "    use_rb = spec.get('use_rawboost', RAWBOOST_DEFAULT)",
            "    if use_rb != RAWBOOST_DEFAULT:",
            "        from glada.augmentation import RawBoost as _RB",
            "        aug = _RB if use_rb else None",
            "        local_source_ds = ASVspoofDatasetWithSpeaker(",
            "            protocol_path=ASV_PROTO, audio_dir=ASV_AUDIO,",
            "            config=cfg, augmenter=aug,",
            "        )",
            "    else:",
            "        local_source_ds = source_ds",
            "",
            "    loaders = create_dataloaders(cfg, local_source_ds, target_ds, eval_ds, extra_eval)",
            "",
            "    ModelCls = MODEL_CLASSES[spec['model_class']]",
            "    model_kwargs = spec.get('model_kwargs', {})",
            "    TrainerCls = GLADATrainer  # No staggered variants in Phase A/B",
            "",
            "    class _VariantTrainer(TrainerCls):",
            "        def __init__(self, *a, **kw):",
            "            super().__init__(*a, **kw)",
            "            old_model = self.model",
            "            new_model = ModelCls(self.config, **model_kwargs).to(self.device)",
            "            old_state = old_model.state_dict()",
            "            new_state = new_model.state_dict()",
            "            for k in new_state.keys():",
            "                if k in old_state and old_state[k].shape == new_state[k].shape:",
            "                    new_state[k] = old_state[k]",
            "            new_model.load_state_dict(new_state, strict=False)",
            "            self.model = new_model",
            "            self._setup_phase1()",
            "            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)",
            "            total = sum(p.numel() for p in self.model.parameters())",
            "            print(f'  [VARIANT] {ModelCls.__name__} kwargs={model_kwargs}; '",
            "                  f'trainable={trainable:,}/{total:,}')",
            "",
            "    try:",
            "        trainer_kwargs = dict(",
            "            config=cfg,",
            "            source_loader=loaders['source'],",
            "            target_loader=loaders['target'],",
            "            eval_loader=loaders['eval'],",
            "            extra_eval_loaders={k: v for k, v in loaders.items() if k not in ('source', 'target', 'eval')},",
            "            num_speakers=num_speakers,",
            "        )",
            "        trainer = _VariantTrainer(**trainer_kwargs)",
            "        results = trainer.train()",
            "        all_results[name] = results",
            "    except Exception as e:",
            "        import traceback; traceback.print_exc()",
            "        all_results[name] = {'error': str(e)}",
            "",
            "    save = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in all_results.items()}",
            "    with open(results_path, 'w') as f:",
            "        json.dump(save, f, indent=2, default=str)",
            "",
            "    if 'trainer' in dir(): del trainer",
            "    torch.cuda.empty_cache(); gc.collect()",
            "",
            "print(f'\\nSweep complete: {len(all_results)} variants in {(time.time()-sweep_start)/60:.1f} min')",
        ),
    ]


def reference_table_cells(reference_dict_lines: list[str], extra_intro: str = ""):
    """Section 10: decision table comparing variants against the reference set.

    reference_dict_lines is a list of "    'name':         value," lines that
    fill the REFERENCE dict.
    """
    intro = "## 10. Decision Table"
    if extra_intro:
        intro = intro + "\n\n" + extra_intro
    return [
        md(intro),
        code(
            "REFERENCE = {",
            *reference_dict_lines,
            "}",
            "",
            "header = f'{\"variant\":<32s} {\"ITW EER\":>10s}' + ''.join(f'  vs {k:<18s}' for k in REFERENCE.keys())",
            "print(header); print('-' * len(header))",
            "for name, res in all_results.items():",
            "    if 'error' in res:",
            "        print(f'{name:<32s} FAILED: {res[\"error\"][:60]}'); continue",
            "    eer = res.get('best_eer') or res.get('eval_metrics', {}).get('ITW', {}).get('eer')",
            "    if eer is None:",
            "        print(f'{name:<32s} NO EER'); continue",
            "    deltas = ''.join(f'  {(eer - v):>+18.2f}' for v in REFERENCE.values())",
            "    print(f'{name:<32s} {eer:>10.2f}{deltas}')",
            "",
            "print()",
            "print('Reference (no retraining):')",
            "for name, eer in REFERENCE.items():",
            "    print(f'  {name:<32s} {eer:>10.2f}')",
        ),
    ]


def write_notebook(cells: list, path: str):
    """Render cells to .ipynb."""
    import json as _json
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "machine_shape": "hm"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w") as f:
        _json.dump(nb, f, indent=1)
    print(f"Wrote {path} ({len(cells)} cells)")
