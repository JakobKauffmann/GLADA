"""
Multi-dataset evaluator for comprehensive deepfake detection benchmarking.

Purpose
-------
Evaluates a trained GLADAModel on multiple audio deepfake detection benchmarks
in a single pass: ITW, ASVspoof 2019 LA, ASVspoof 2021 LA, ASVspoof 2021 DF,
and optionally WaveFake and CodecFake+.

Architectural context
---------------------
Cross-domain generalization is the central challenge in audio deepfake detection.
Models trained solely on ASVspoof 2019 LA (studio-quality TTS/VC attacks) often
collapse on In-the-Wild (ITW) data, which contains real-world recordings with
codec artifacts, background noise, and unseen generation methods. GLADA's grouped
adversarial domain adaptation is specifically designed to bridge this gap. This
evaluator quantifies that generalization by reporting EER on each benchmark.

Primary metric: Equal Error Rate (EER)
--------------------------------------
EER is the standard metric for audio deepfake detection, established by the
ASVspoof challenge series (Yamagishi et al., 2021). It is the threshold-independent
operating point where FAR = FRR:

    EER = (FAR(tau*) + FRR(tau*)) / 2,  tau* = argmin |FAR(tau) - FRR(tau)|

Lower EER is better; 50% = random guessing.

Bootstrap confidence intervals (Efron & Tibshirani, 1993) are required for
publication-quality results and are computed with n=1000 resamples (95% CI).
See ``glada.evaluation.metrics.compute_eer_with_ci`` for implementation.

Literature
----------
- Yamagishi et al. (2021) "ASVspoof 2021: accelerating progress in spoofed and
  deepfake speech detection" — defines the evaluation protocol used here.
- Muller et al. (2022) "Does Audio Deepfake Detection Generalize?" (Interspeech)
  — quantifies the cross-domain generalization gap motivating this evaluator.
- Kinnunen et al. (2020) "t-DCF: a Detection Cost Function for the Tandem
  Assessment of Spoofing Countermeasures and Automatic Speaker Verification"
  — t-DCF is an alternative metric; we use EER as the primary metric following
  recent SOTA work (IB-CAAN, HierCon) but t-DCF can be added if needed.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Any
from torch.amp import autocast

from glada.evaluation.metrics import compute_eer_with_ci

class MultiDatasetEvaluator:
    """
    Evaluates a GLADAModel across multiple audio deepfake detection benchmarks.

    Design rationale
    ----------------
    Decoupled from training: takes a pre-loaded model and a dict of DataLoaders,
    so the same evaluator instance can run on multiple checkpoints (e.g., for
    multi-seed or multi-backbone comparison) without redundant model instantiation.

    AMP handling
    ------------
    Inference uses the same AMP dtype as training (bfloat16 on A100, float16 on T4)
    to match training-time numerical precision. Note that bfloat16 tensors cannot
    be directly converted to numpy — they must be cast to float32 first (see the
    ``evaluate_dataset`` method). This is a known PyTorch limitation.

    Supported datasets (eval_loaders keys)
    ---------------------------------------
    - "ITW"      : In-the-Wild (Muller et al., Interspeech 2022)
    - "2019_LA"  : ASVspoof 2019 LA eval set (Wang et al., 2020)
    - "2021_LA"  : ASVspoof 2021 LA eval set (Yamagishi et al., 2021)
    - "2021_DF"  : ASVspoof 2021 DF eval set (Yamagishi et al., 2021)
    - "WaveFake" : WaveFake (Frank & Schonherr, 2021)
    - "CodecFake": CodecFake+ (optional, codec-based attacks)
    """

    def __init__(self, model, config, device, eval_loaders: Dict[str, Any]):
        """
        Args:
            model: GLADAModel instance (already loaded with weights).
            config: GLADAConfig instance. Used for AMP dtype and use_amp flag.
            device: torch.device for inference.
            eval_loaders: Dict mapping dataset name strings to DataLoaders.
                          Only datasets present in this dict will be evaluated.
        """
        self.model = model
        self.config = config
        self.device = device
        self.eval_loaders = eval_loaders

        # Select AMP dtype based on config.
        # bfloat16 is preferred on A100 (wider dynamic range than float16).
        # float16 is used on T4/V100 (bfloat16 not supported).
        self.amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16

    @torch.no_grad()
    def evaluate_dataset(self, loader, name="dataset") -> dict:
        """
        Evaluate model on a single dataset with bootstrap confidence intervals.

        Workflow:
          1. Run inference on all samples (no gradient computation)
          2. Extract bonafide probability: P(class=bonafide | x) via softmax
          3. Compute EER: threshold τ* where FPR(τ*) = FNR(τ*) (equal error rate)
          4. Bootstrap CI (n=1000): resample test set, recompute EER, report 95% percentiles

        Metrics returned:
          - eer: Equal Error Rate (%, lower is better)
          - ci_low, ci_high: 95% bootstrap confidence interval bounds
          - auc: Area Under ROC Curve (higher is better)
          - acc: Accuracy at EER threshold

        Note: BFloat16 tensors (from A100 AMP) cannot convert directly to numpy.
        Must cast to float32 first (line 59).
        """
        self.model.eval()
        scores, labels = [], []
        
        for batch in tqdm(loader, desc=f"Eval {name}"):
            inp = batch['input_values'].to(self.device, non_blocking=True)
            mask = batch['attention_mask'].to(self.device, non_blocking=True)
            
            with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
                out = self.model(inp, attention_mask=mask)
                
            # Get bonafide probability (class 1)
            probs = F.softmax(out['spoof_logits'], dim=1)[:, 1]
            
            # CRITICAL BUG FIX: 
            # BFloat16 tensors cannot be directly converted to numpy.
            # Must cast to float32 first.
            scores.extend(probs.float().cpu().numpy())
            labels.extend(batch['task_label'].numpy())
            
        labels = np.array(labels)
        scores = np.array(scores)
        
        # Compute metrics with bootstrap CI
        metrics = compute_eer_with_ci(labels, scores, n_bootstrap=1000)
        metrics["n_samples"] = len(labels)
        metrics["name"] = name
        
        return metrics

    def evaluate_all(self, n_bootstrap=1000) -> Dict[str, dict]:
        """
        Run evaluation on all datasets in ``self.eval_loaders``.

        Iterates over every (name, loader) pair and calls ``evaluate_dataset``
        on each. Results are printed in a fixed-width table and returned as a
        dict for downstream use (saving to JSON, building LaTeX tables, etc.).

        Args:
            n_bootstrap: Number of bootstrap resamples for confidence intervals.
                         Passed through to ``compute_eer_with_ci``. Default 1000
                         gives stable 95% CI estimates with negligible overhead.

        Returns:
            Dict[str, dict]: Maps dataset name to the metrics dict returned by
                ``evaluate_dataset`` (keys: eer, ci_low, ci_high, auc, acc,
                n_samples, name).
        """
        results = {}
        print(f"\n{'='*20} Multi-Dataset Evaluation {'='*20}")
        
        for name, loader in self.eval_loaders.items():
            results[name] = self.evaluate_dataset(loader, name)
            r = results[name]
            print(f"  {name:<15}: EER={r['eer']:.2f}% [{r['ci_low']:.2f}, {r['ci_high']:.2f}] "
                  f"AUC={r['auc']:.4f} Acc={r['acc']:.1f}% (n={r['n_samples']})")
                  
        print('='*64 + '\n')
        return results

    @staticmethod
    def results_to_latex(results: Dict[str, dict]) -> str:
        """
        Format evaluation results as a LaTeX table row for the thesis.

        Output format (one column per dataset, standard order):
            Method & ITW & 2019 LA & 2021 LA & 2021 DF \\\\

        Each cell reports:
            EER (±margin)
        where margin = (ci_high - ci_low) / 2 approximates the half-width of
        the 95% bootstrap confidence interval.

        The ``$\\pm$`` LaTeX command produces the ± symbol. This is written
        as a raw string escape in Python f-strings (``\\pm``); do NOT simplify
        it to ``\pm`` or the backslash will be lost in the output string.

        Args:
            results: Dict[str, dict] as returned by ``evaluate_all``.

        Returns:
            A string containing the & -separated LaTeX table cells, terminated
            with a trailing space. Paste directly into a LaTeX tabular row and
            append \\\\ for the line break.

        Example output:
            5.23 ($\\pm$0.41) & 2.11 ($\\pm$0.18) & 8.34 ($\\pm$0.62) & -
        """
        # Define standard column order; datasets not present become "-"
        order = ["ITW", "2019_LA", "2021_LA", "2021_DF"]

        cells = []
        for key in order:
            if key in results:
                r = results[key]
                # Half-width of the 95% CI approximates the ±margin
                margin = (r['ci_high'] - r['ci_low']) / 2
                cells.append(f"{r['eer']:.2f} ($\\pm${margin:.2f})")
            else:
                cells.append("-")

        return " & ".join(cells) + " "

    @staticmethod
    def results_to_markdown(results: Dict[str, dict]) -> str:
        """
        Format evaluation results as a Markdown table (for notebook display).

        Useful for quick result inspection in Colab/Jupyter. The LaTeX version
        (``results_to_latex``) is used for the actual thesis tables.

        Args:
            results: Dict[str, dict] as returned by ``evaluate_all``.

        Returns:
            A multi-line Markdown string with a header row and one data row
            per dataset. Can be printed directly or rendered in a notebook.
        """
        lines = [
            "| Dataset | EER (%) | 95% CI | AUC | Samples |",
            "|---|---|---|---|---|",
        ]
        for name, r in results.items():
            lines.append(
                f"| {name} | {r['eer']:.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['auc']:.4f} | {r['n_samples']:,} |"
            )
        return "\n".join(lines)
