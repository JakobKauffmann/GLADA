"""
DataLoader factory functions for GLADA's unsupervised domain adaptation training.

Training loop structure:
  GLADA trains on INTERLEAVED mini-batches from two separate loaders:
    source_loader  — ASVspoof 2019 LA (labeled, d=0)
    target_loader  — ITW (unlabeled; task labels suppressed, d=1)
  At each step, one batch from each loader is drawn and combined.
  The source batch drives L_spoof + L_speaker; both batches drive L_GLADA
  (grouped-layer domain adversarial loss).

  Using SEPARATE loaders (rather than a concatenated dataset) lets each
  domain maintain its own sampling strategy independently.  The source
  domain uses class-balanced sampling (WeightedRandomSampler) to correct
  its ~1:9 bonafide/spoof imbalance.  The target domain uses uniform
  random sampling (shuffle=True) to present its natural distribution —
  using balanced sampling on the target would constitute indirect label
  leakage, violating the unsupervised DA assumption.

Class imbalance:
  ASVspoof 2019 LA training set:
    Bonafide: 2,580 samples (~10% of total)
    Spoof   : 22,800 samples (~90% of total)
  Without rebalancing, the model ignores bonafide samples and achieves
  trivially low loss by predicting spoof for everything.
  WeightedRandomSampler assigns weight ∝ 1/class_count to each sample,
  producing approximately 50/50 class distribution per batch.

Colab constraint:
  num_workers=0 is the REQUIRED default in Colab.  Colab's multiprocessing
  environment does not reliably support forked DataLoader workers (CUDA
  context sharing issues, OOM on fork).  Set GLADAConfig.num_workers=0.
  persistent_workers is automatically disabled when num_workers=0.

Evaluation:
  Eval loaders use batch_size * 4 (no gradient → more GPU headroom) and
  shuffle=False (deterministic ordering required for score aggregation
  and alignment with per-sample metadata).
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Optional

from glada.config import GLADAConfig

def create_balanced_sampler(dataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that produces ~50/50 class-balanced batches.

    Motivation:
      ASVspoof 2019 LA training set is heavily skewed (~1:9 bonafide:spoof).
      Without rebalancing, the model converges to predicting 'spoof' for
      everything, which minimises cross-entropy loss trivially but leaves
      EER near 50%.  Inverse-frequency weighting corrects this.

    Weighting formula:
      w_i = 1 / count(class_i)    for sample i with label class_i
    In expectation, the probability of sampling each class is proportional
    to (1/count) * count = 1, i.e., uniform across classes.

    Note: replacement=True is required so that the minority class (bonafide)
    can be oversampled to match the majority class.  This means some bonafide
    samples appear multiple times per epoch, effectively upweighting them.

    Args:
        dataset: Dataset with a `samples` attribute — a list of dicts each
                 containing a 'label' key (0 = spoof, 1 = bonafide).

    Returns:
        WeightedRandomSampler ready to pass to DataLoader(sampler=...).
    """
    labels = [s['label'] for s in dataset.samples]
    
    # Count classes
    count0 = labels.count(0)  # Spoof
    count1 = labels.count(1)  # Bonafide
    
    # Compute weights
    # +1e-5 to avoid division by zero
    weight0 = 1.0 / (count0 + 1e-5)
    weight1 = 1.0 / (count1 + 1e-5)
    
    # Assign weight to each sample
    weights = [weight1 if l == 1 else weight0 for l in labels]
    
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def create_dataloaders(
    config: GLADAConfig, 
    source_ds, 
    target_ds, 
    eval_ds, 
    extra_eval: Optional[Dict[str, object]] = None
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training and evaluation.
    
    Args:
        config (GLADAConfig): Configuration object.
        source_ds (Dataset): Source domain training set (ASVspoof).
        target_ds (Dataset): Target domain training set (ITW).
        eval_ds (Dataset): Primary evaluation set (ITW).
        extra_eval (dict, optional): Dict of name->Dataset for additional eval sets.
        
    Returns:
        dict: Dictionary containing 'source', 'target', 'eval', and extra loaders.
    """
    loaders = {}

    # persistent_workers keeps worker processes alive between batches,
    # avoiding fork overhead. Only valid when num_workers > 0.
    pw = config.num_workers > 0

    # --- Training Loaders ---
    loaders['source'] = DataLoader(
        source_ds,
        batch_size=config.batch_size,
        sampler=create_balanced_sampler(source_ds),
        num_workers=config.num_workers,
        pin_memory=True,        # pin_memory speeds up host→GPU transfer (no-op on CPU-only)
        drop_last=True,         # drop_last=True: avoids unequal batch sizes that can cause
                                # mismatch when zipping source/target batches in the training loop
        persistent_workers=pw,
    )

    # Target domain: uniform random sampling (shuffle=True) to present the
    # natural class distribution.  Using balanced sampling here would require
    # reading target labels, constituting indirect label leakage that violates
    # the unsupervised DA assumption (target labels must not influence training).
    loaders['target'] = DataLoader(
        target_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,         # must match source batch_size so GRL discriminator inputs align
        persistent_workers=pw,
    )

    # --- Evaluation Loaders ---
    # No grad → more GPU headroom → larger eval batch for throughput.
    # Cap at 4x training batch but never exceed 192: WavLM-Large outputs
    # all 25 hidden states per sample; at bf16 each sample costs ~10 MB,
    # so eval_bs=192 already needs ~2 GB for the stacked hidden states alone
    # plus ~6-8 GB for intermediate activations — well within H100 headroom
    # but 768 (=192*4) caused fragmentation OOM in observed runs.
    eval_bs = min(config.batch_size * 4, 192)

    loaders['eval'] = DataLoader(
        eval_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=pw,
    )

    if extra_eval:
        for name, ds in extra_eval.items():
            loaders[name] = DataLoader(
                ds,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=pw,
            )
            
    return loaders
