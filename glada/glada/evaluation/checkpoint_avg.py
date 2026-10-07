"""
Checkpoint averaging utilities for GLADA.

Averages the parameters of the top-K checkpoints (ranked by validation EER)
to produce a single model with improved generalization. This is a standard
post-training technique in speech processing that yields "free" gains without
additional training cost.

Theoretical foundation:
  - Polyak & Juditsky (1992), "Acceleration of Stochastic Approximation by
    Averaging": Averaging iterates of SGD converges at the optimal O(1/N) rate,
    even when individual iterates converge more slowly. For convex objectives,
    the averaged solution concentrates around the optimum with lower variance
    than any single iterate. While deep networks are non-convex, weight averaging
    over checkpoints from the same loss basin approximates this effect.

  - Izmailov et al. (2018), "Averaging Weights Leads to Wider Optima and Better
    Generalization" (SWA): Showed that averaging weights sampled from the SGD
    trajectory (with cyclic or constant learning rate) finds solutions in flatter
    regions of the loss landscape, improving generalization. SWA averages
    cyclically throughout training; our approach selects the top-K checkpoints
    post-hoc, which is simpler but captures a similar benefit.

  - Huang et al. (2025), IB-CAAN (arxiv 2509.23618): Applied top-5 checkpoint
    averaging (by EER) to audio deepfake detection, improving ITW EER from
    4.93% to 4.61% — a meaningful gain at zero additional training cost.

Implementation:
  1. Load history.json from the run directory to identify top-K epochs by EER.
  2. Load the corresponding checkpoint files (ckpt_epNN.pt format).
  3. Average all state_dict parameters element-wise (arithmetic mean).
  4. Return a model initialized with the averaged weights.

Requirements:
  - Periodic checkpoints must be saved during training (save_every_n_epochs > 0
    in GLADAConfig). If only best_model.pt exists, only one checkpoint is
    available and averaging degenerates to the single best model.
"""

import os
import json
import warnings
from typing import List, Optional, Type

import torch
import torch.nn as nn


def average_checkpoints(
    run_dir: str,
    model_class: Type[nn.Module],
    config: object,
    top_k: int = 5,
    metric: str = "eer",
    device: str = "cpu",
) -> nn.Module:
    """
    Load the top-K checkpoints by validation metric and return a model with
    averaged weights.

    The averaging is a simple element-wise arithmetic mean across state dicts:

        theta_avg = (1/K) * sum_{i=1}^{K} theta_i

    where theta_i is the parameter vector of the i-th checkpoint. This is
    equivalent to Polyak averaging (Polyak & Juditsky, 1992) applied post-hoc
    to selected checkpoints rather than online to SGD iterates.

    Theoretical justification: For parameters in the same loss basin (which
    checkpoints from a single training run typically are), the averaged
    parameters lie closer to the basin center, which tends to be flatter and
    thus generalizes better (Izmailov et al., 2018, SWA).

    Args:
        run_dir: Path to the training run directory containing history.json
            and checkpoint files (ckpt_epNN.pt).
        model_class: The model class (e.g., GLADAModel) to instantiate. Must
            accept a single `config` argument.
        config: Configuration object (e.g., GLADAConfig) passed to model_class.
        top_k: Number of best checkpoints to average. Huang et al. (2025)
            used K=5 for IB-CAAN. If fewer than top_k checkpoints are found,
            averages all available checkpoints with a warning.
        metric: Metric name to rank checkpoints by (default: "eer"). Lower
            values are considered better (suitable for EER, loss, etc.).
        device: Device to load checkpoints onto (default: "cpu" to avoid
            GPU OOM when loading multiple checkpoints).

    Returns:
        An nn.Module instance with averaged weights, moved to `device`.

    Raises:
        FileNotFoundError: If history.json does not exist in run_dir.
        RuntimeError: If no valid checkpoint files are found.
        RuntimeError: If checkpoint state_dict keys do not match the model,
            indicating architecture mismatch between checkpoints.
    """
    # --- Step 1: Load training history and rank epochs by metric ---
    history_path: str = os.path.join(run_dir, "history.json")
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"history.json not found in {run_dir}. Cannot determine top-K "
            f"epochs without training history."
        )

    with open(history_path, "r") as f:
        history: List[dict] = json.load(f)

    # Extract (epoch, metric_value) pairs from history.
    # History format: each entry has "epoch" (int) and "eval" dict containing
    # dataset results. The primary eval metric is ITW EER, stored at
    # history[i]["eval"]["ITW"]["eer"].
    epoch_metrics: List[tuple] = []
    for entry in history:
        epoch_num: int = entry.get("epoch", -1)
        eval_data: Optional[dict] = entry.get("eval", {})

        # Try to find the metric in evaluation results.
        # Convention: metric is looked up in the first eval dataset that
        # contains it (typically ITW).
        metric_value: Optional[float] = None
        for dataset_name, dataset_results in eval_data.items():
            if isinstance(dataset_results, dict) and metric in dataset_results:
                metric_value = dataset_results[metric]
                break

        if metric_value is not None:
            epoch_metrics.append((epoch_num, metric_value))

    if not epoch_metrics:
        raise RuntimeError(
            f"No entries in history.json contain metric '{metric}' in eval "
            f"results. Available history keys: {list(history[0].keys()) if history else '(empty)'}."
        )

    # Sort by metric value (ascending — lower EER is better)
    epoch_metrics.sort(key=lambda x: x[1])

    # Select top-K epochs
    selected_epochs: List[tuple] = epoch_metrics[:top_k]
    print(f"Checkpoint averaging: top-{top_k} epochs by {metric}:")
    for epoch_num, val in selected_epochs:
        print(f"  Epoch {epoch_num}: {metric}={val:.4f}")

    # --- Step 2: Load checkpoint files for selected epochs ---
    state_dicts: List[dict] = []
    loaded_epochs: List[int] = []

    for epoch_num, metric_val in selected_epochs:
        # Checkpoint filename convention from BaseTrainer._save_checkpoint:
        # periodic checkpoints are named "ckpt_epNN.pt"
        ckpt_filename: str = f"ckpt_ep{epoch_num:02d}.pt"
        ckpt_path: str = os.path.join(run_dir, ckpt_filename)

        # Also check best_model.pt if the epoch matches the best epoch
        if not os.path.exists(ckpt_path):
            # Try without zero-padding for epochs >= 100
            ckpt_filename = f"ckpt_ep{epoch_num}.pt"
            ckpt_path = os.path.join(run_dir, ckpt_filename)

        if not os.path.exists(ckpt_path):
            # Check if this epoch corresponds to the best model
            best_path: str = os.path.join(run_dir, "best_model.pt")
            if os.path.exists(best_path):
                best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
                if best_ckpt.get("epoch") == epoch_num:
                    ckpt_path = best_path
                else:
                    warnings.warn(
                        f"Checkpoint for epoch {epoch_num} not found "
                        f"({metric}={metric_val:.4f}). Skipping. "
                        f"Ensure save_every_n_epochs > 0 in GLADAConfig.",
                        stacklevel=2,
                    )
                    continue
            else:
                warnings.warn(
                    f"Checkpoint for epoch {epoch_num} not found at "
                    f"{ckpt_path}. Skipping.",
                    stacklevel=2,
                )
                continue

        print(f"  Loading {os.path.basename(ckpt_path)} ...")
        ckpt: dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dicts.append(ckpt["model"])
        loaded_epochs.append(epoch_num)

    if not state_dicts:
        raise RuntimeError(
            f"No valid checkpoint files found for top-{top_k} epochs "
            f"{[e for e, _ in selected_epochs]}. "
            f"Ensure save_every_n_epochs > 0 during training."
        )

    if len(state_dicts) < top_k:
        warnings.warn(
            f"Only {len(state_dicts)}/{top_k} checkpoints found. "
            f"Averaging {len(state_dicts)} checkpoints.",
            stacklevel=2,
        )

    # --- Step 3: Validate architecture consistency across checkpoints ---
    # All state_dicts must have identical keys; mismatched keys indicate that
    # checkpoints were saved from different model architectures.
    reference_keys: set = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        current_keys: set = set(sd.keys())
        if current_keys != reference_keys:
            missing: set = reference_keys - current_keys
            extra: set = current_keys - reference_keys
            raise RuntimeError(
                f"Architecture mismatch between checkpoint 0 (epoch "
                f"{loaded_epochs[0]}) and checkpoint {i} (epoch "
                f"{loaded_epochs[i]}). "
                f"Missing keys: {missing}, Extra keys: {extra}. "
                f"All checkpoints must come from the same model architecture."
            )

    # --- Step 4: Average state_dict parameters (element-wise mean) ---
    # Uses torch operations only (no numpy) for GPU compatibility and precision.
    #
    # For each parameter key, stack all checkpoint values into a single tensor
    # along a new dimension 0, then take the mean. This handles all dtypes
    # that support addition (float, half, bfloat16, etc.).
    num_ckpts: int = len(state_dicts)
    averaged_sd: dict = {}

    for key in state_dicts[0].keys():
        # Stack: [num_ckpts, *param_shape] -> mean along dim=0 -> [*param_shape]
        stacked: torch.Tensor = torch.stack(
            [sd[key].float() for sd in state_dicts], dim=0
        )
        averaged_sd[key] = (stacked.sum(dim=0) / num_ckpts).to(
            state_dicts[0][key].dtype
        )

    # --- Step 5: Build model and load averaged weights ---
    model: nn.Module = model_class(config)
    model.load_state_dict(averaged_sd)
    model.to(device)
    model.eval()

    print(
        f"\nCheckpoint averaging complete: {num_ckpts} checkpoints averaged "
        f"(epochs: {loaded_epochs})"
    )

    return model


def save_averaged_checkpoint(
    model: nn.Module,
    run_dir: str,
    config: object,
    top_k: int,
    source_epochs: List[int],
) -> str:
    """
    Save the averaged model checkpoint with metadata about its provenance.

    The saved checkpoint follows the same format as BaseTrainer._save_checkpoint
    but replaces optimizer state (not meaningful for an averaged model) with
    metadata documenting which epochs were averaged.

    Filename convention: averaged_top{K}.pt
      e.g., averaged_top5.pt for a top-5 average.

    Args:
        model: The model with averaged weights (output of average_checkpoints).
        run_dir: Path to the training run directory.
        config: Configuration object (serializable via dataclasses.asdict or
            stored directly).
        top_k: Number of checkpoints that were averaged (used in filename).
        source_epochs: List of epoch numbers that were averaged (for
            reproducibility and provenance tracking).

    Returns:
        The full path to the saved checkpoint file.
    """
    from dataclasses import asdict

    # Serialize config — handle both dataclass and dict configs gracefully
    try:
        config_dict: dict = asdict(config)
    except TypeError:
        # Config is already a dict or not a dataclass
        config_dict = dict(config) if hasattr(config, "__iter__") else {}

    ckpt: dict = {
        "model": model.state_dict(),
        "averaged": True,
        "top_k": top_k,
        "source_epochs": source_epochs,
        "num_averaged": len(source_epochs),
        "config": config_dict,
    }

    filename: str = f"averaged_top{top_k}.pt"
    save_path: str = os.path.join(run_dir, filename)
    torch.save(ckpt, save_path)

    print(f"Averaged checkpoint saved: {save_path}")
    print(f"  Source epochs: {source_epochs}")

    return save_path
