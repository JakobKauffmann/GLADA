"""
Shared training infrastructure for GLADA experiments.

This module provides the BaseTrainer class, which implements all training utilities
that are independent of the specific loss composition or architecture:

  - Checkpoint save/resume: designed around Colab disconnection recovery.
      * best_model.pt       — saved whenever validation EER improves.
      * phase_transition.pt — saved at the Phase 1 → Phase 2 boundary.
      * ckpt_epNN.pt        — periodic saves every save_every_n_epochs; oldest
                              files are pruned to keep only the most recent
                              save_top_k (default 10) to avoid filling Drive.
      * best_model.safetensors — faster-loading weights-only snapshot for inference
                                  (requires the safetensors package; falls back to
                                  torch.save only if unavailable).

  - Mixed Precision (AMP):
      * float16: uses GradScaler to prevent underflow in fp16 matmuls.
      * bfloat16: no GradScaler needed (bfloat16 has the same exponent range as
                  fp32, so underflow is not a concern). Preferred on A100/H100.
      * TF32: enabled for matmul and cuDNN on all CUDA-capable GPUs for ~3x speedup
              with negligible numerical difference (10-bit mantissa vs 23-bit fp32).

  - Evaluation: computes EER, AUC-ROC, and accuracy-at-EER-threshold.
      EER (Equal Error Rate) is the standard metric for spoofing detection:
        EER = (FPR + FNR) / 2  evaluated at the threshold where FPR ≈ FNR.
      Lower EER is better; random chance = 50%.

  - History tracking: every epoch's metrics are appended to self.history and
      serialised to history.json in the run directory so training curves survive
      Colab session restarts.

Note on Colab constraints:
  - num_workers=0 in all DataLoaders (Colab multiprocessing is unreliable).
  - Checkpoints written to Google Drive (mounted at /content/drive/MyDrive/...).
  - AMP dtype determined by GPU: bfloat16 on A100, float16 on T4.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import asdict
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_curve, roc_auc_score

# safetensors is ~2x faster than torch.save for model weights (no pickle overhead)
try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class BaseTrainer:
    """
    Abstract base class providing shared infrastructure for GLADA trainers.

    Subclasses (GLADATrainer) are responsible for:
      - Defining train_epoch() with the task-specific loss composition.
      - Calling _setup_phase1() / _setup_phase2() to configure optimizers.
      - Calling _save_checkpoint() / _load_checkpoint() as needed.

    This class is not intended to be instantiated directly.
    """

    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device

        # --- GPU Optimizations ---
        # TF32: A100/H100 tensor cores provide ~3x speedup for fp32 matmul
        # with negligible precision loss (mantissa 10 bits vs 23).
        # cudnn.benchmark: auto-tunes convolution algorithms for fixed-size inputs.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # --- Training State ---
        self.run_dir = config.get_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)

        # Save configuration
        with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)

        # AMP Setup
        self.amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        # GradScaler is recommended for float16, but optional/no-op for bfloat16 (usually)
        # However, we keep logic consistent: use scaler if enabled and float16
        self.scaler = GradScaler() if (config.use_amp and self.amp_dtype == torch.float16) else None
        
        # Tracking
        self.best_eer = float('inf')
        self.best_epoch = -1
        self.patience_counter = 0
        self.history = []
        self.start_epoch = 0
        self.current_phase = 1

    def _save_checkpoint(self, epoch, optimizer, is_best=False, phase_transition=False):
        """
        Persist model + optimizer state to disk.

        Checkpoints are written selectively to avoid filling Google Drive:
          - Always saved when is_best=True (EER improved) → best_model.pt
          - Always saved when phase_transition=True (Phase 1 complete) → phase_transition.pt
          - Saved every save_every_n_epochs epochs (if > 0) → ckpt_epNN.pt
            Periodic files are pruned to keep only the save_top_k most recent.

        For the best-model checkpoint, a weights-only safetensors file is also
        written (best_model.safetensors) when the safetensors package is available.
        This is ~2x faster to load at inference time because it avoids pickle.
        The full .pt file (with optimizer state) is always written alongside it
        for resumability.

        Args:
            epoch          (int):  Current epoch index (0-indexed).
            optimizer:             Active optimizer (Phase 1 or Phase 2 variant).
            is_best        (bool): Whether this epoch achieved a new best EER.
            phase_transition(bool): Whether this is the Phase 1 -> Phase 2 boundary.

        Returns:
            str|None: Path of the saved checkpoint, or None if nothing was saved.
        """
        should_save = is_best or phase_transition
        if self.config.save_every_n_epochs > 0:
            should_save = should_save or ((epoch + 1) % self.config.save_every_n_epochs == 0)
        if not should_save:
            return None

        ckpt = {
            'epoch': epoch + 1,       # Saved as next epoch so resume starts correctly
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_eer': self.best_eer,
            'best_epoch': self.best_epoch,
            'phase': self.current_phase,
            'config': asdict(self.config),
        }

        if is_best:
            filename = "best_model.pt"
        elif phase_transition:
            filename = "phase_transition.pt"
        else:
            filename = f"ckpt_ep{epoch+1:02d}.pt"

        path = os.path.join(self.run_dir, filename)

        if SAFETENSORS_AVAILABLE and is_best:
            # Weights-only snapshot for fast inference loading (no pickle overhead).
            # Cannot store optimizer state — the full .pt below handles that.
            st_path = os.path.join(self.run_dir, "best_model.safetensors")
            _st_save(self.model.state_dict(), st_path)

        torch.save(ckpt, path)

        # Prune old periodic checkpoints (never touches best_model.pt / phase_transition.pt)
        if not is_best and not phase_transition:
            self._prune_old_checkpoints()

        return path

    def _prune_old_checkpoints(self):
        """
        Remove the oldest periodic checkpoint(s) when the count exceeds save_top_k.

        Only ckpt_epNN.pt files are considered for pruning. best_model.pt and
        phase_transition.pt are unconditionally preserved. Files are sorted by
        modification time (oldest first), so the most recently written periodic
        checkpoints survive.

        Disabled when save_top_k <= 0 (keep all).
        """
        save_top_k = getattr(self.config, 'save_top_k', 10)
        if save_top_k <= 0:
            return
        import glob
        pattern = os.path.join(self.run_dir, "ckpt_ep*.pt")
        ckpt_files = sorted(glob.glob(pattern), key=os.path.getmtime)
        while len(ckpt_files) > save_top_k:
            oldest = ckpt_files.pop(0)
            os.remove(oldest)

    def _load_checkpoint(self, path):
        """
        Restore model weights and training state from a checkpoint file.

        Loads model weights, start epoch, best EER, best epoch, and training history
        (from history.json in the same directory). The optimizer state dict is
        returned in the checkpoint dict but NOT applied here — the caller (GLADATrainer
        ._resume) is responsible for loading it after building the correct optimizer.

        Args:
            path (str): Absolute path to the .pt checkpoint file.

        Returns:
            dict: Raw checkpoint dict (includes 'optimizer' key for the caller).
        """
        print(f"  Resuming from {path}")
        # weights_only=False required because the checkpoint contains non-tensor
        # objects (config dataclass dict, Python ints, etc.).
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.start_epoch = ckpt.get('epoch', 0)
        self.best_eer = ckpt.get('best_eer', float('inf'))
        self.best_epoch = ckpt.get('best_epoch', -1)

        # Restore per-epoch history for uninterrupted logging on resume
        hist_path = os.path.join(os.path.dirname(path), "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                self.history = json.load(f)

        return ckpt

    def _save_history(self):
        """
        Append the current epoch log to history.json in the run directory.

        Called at the end of every epoch so that training curves are recoverable
        after a Colab disconnection without needing to re-run epochs.
        """
        with open(os.path.join(self.run_dir, "history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

    @torch.no_grad()
    def evaluate(self, loader, name="Eval"):
        """
        Compute EER, AUC-ROC, and accuracy at the EER threshold on a single loader.

        Metric definitions:
          EER (Equal Error Rate): threshold where FPR == FNR; standard for speaker
              verification / anti-spoofing. Lower is better (0% = perfect, 50% = chance).
              EER = (FPR[idx] + FNR[idx]) / 2  where idx = argmin |FPR - FNR|.
          AUC: area under the ROC curve; threshold-free discriminability measure.
          Acc: binary accuracy (bonafide vs spoof) at the EER decision threshold.

        Label convention: task_label=1 means bonafide (pos_label=1 in roc_curve).
        Score: softmax probability of class 1 (bonafide).

        Edge case: if only one class is present in the batch (e.g., a tiny debug
        loader), sklearn roc_curve raises ValueError. We return chance-level metrics.

        Args:
            loader: DataLoader yielding batches with 'input_values', 'attention_mask',
                    and 'task_label' keys.
            name   (str): Dataset name for display in logs.

        Returns:
            dict: {"eer": float (%), "auc": float, "acc": float (%), "name": str}
        """
        self.model.eval()
        scores, labels = [], []

        for batch in loader:
            inp = batch['input_values'].to(self.device, non_blocking=True)
            mask = batch['attention_mask'].to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
                out = self.model(inp, attention_mask=mask)

            # P(bonafide) = softmax score for class index 1
            probs = F.softmax(out['spoof_logits'], dim=1)[:, 1]

            # bfloat16 tensors cannot be converted to numpy directly; cast first.
            scores.extend(probs.float().cpu().numpy())
            labels.extend(batch['task_label'].numpy())

        labels = np.array(labels)
        scores = np.array(scores)

        if len(np.unique(labels)) < 2:
            # Single-class batch: metrics undefined, return chance-level defaults.
            return {"eer": 50.0, "auc": 0.5, "acc": 0.0, "name": name}

        # EER: find threshold closest to the intersection of FPR and FNR curves.
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100

        auc = roc_auc_score(labels, scores)

        # Accuracy evaluated at the EER operating point (not a fixed 0.5 threshold)
        threshold = thresholds[eer_idx]
        preds = (scores >= threshold).astype(int)
        acc = (preds == labels).mean() * 100

        return {"eer": eer, "auc": auc, "acc": acc, "name": name}
