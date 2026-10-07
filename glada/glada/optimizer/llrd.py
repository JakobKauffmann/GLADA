"""
Layer-Wise Learning Rate Decay (LLRD) optimizer builder.

LLRD assigns exponentially smaller learning rates to lower (earlier) transformer layers,
preserving pre-trained general-purpose representations while allowing higher (task-specific)
layers to adapt more aggressively.

Formula::

    lr_layer_i = lr_base * decay^(L - i)

where L is the index of the highest layer, i is the current layer index (0 = CNN feature
extractor, L = final encoder layer), and decay is typically 0.95. Head parameters
(classifier, discriminators) use a separate, much larger learning rate (head_lr).

Intuition:
  Lower SSL layers (i near 0) encode general acoustic features — phonemes, pitch,
  speaker timbre — that transfer across tasks and should change slowly to avoid
  catastrophic forgetting.
  Higher SSL layers (i near L) encode task-specific patterns and benefit from
  more aggressive updates.

References:
  - Howard & Ruder (ACL 2018), "Universal Language Model Fine-tuning for Text
    Classification" — introduced discriminative fine-tuning (different LRs per layer).
  - Clark et al. (EMNLP 2020), "Electra: Pre-training Text Encoders as Discriminators
    Rather than Generators" — popularised LLRD for large transformer fine-tuning;
    showed performance degrades without it on downstream tasks.
"""

import torch
from glada.config import GLADAConfig

def build_llrd_optimizer(model, config: GLADAConfig):
    """
    Build AdamW optimizer with layer-wise learning rate decay for the backbone.

    Parameter groups created:

    1. **Head parameters** (classifier, discriminators, pooling, VIB, layer norms)
       -- learning rate: ``config.head_lr`` (default 1e-3).
       These are randomly initialised and need fast convergence.

    2. **Backbone encoder layers** (``encoder.layers.N``)
       -- learning rate: ``config.backbone_lr * config.llrd_decay^(L - N)``
       where L = index of the deepest layer.  Lower layers get smaller LRs.

    3. **Backbone non-layer parameters** (CNN feature extractor, feature projection,
       embeddings, final layer norm)
       -- assigned layer_id = 0, receiving the smallest backbone LR:
       ``config.backbone_lr * config.llrd_decay^L``.

    All groups share ``config.weight_decay`` for AdamW regularisation.

    Args:
        model (nn.Module): The full GLADA model containing a ``.backbone`` attribute.
        config (GLADAConfig): Configuration with ``head_lr``, ``backbone_lr``,
            ``llrd_decay``, and ``weight_decay``.

    Returns:
        torch.optim.AdamW: Configured optimizer with named parameter groups.
    """
    param_groups = []

    # --- Head parameters (highest LR) ---
    # Everything EXCEPT the backbone
    head_params = []
    # Explicitly list known head modules to ensure they are captured
    head_modules = ['layer_norms', 'pooling', 'vib', 'deterministic_proj',
                    'task_classifier', 'speaker_adversary', 'domain_disc',
                    'glada_disc']
                    
    for module_name in head_modules:
        module = getattr(model, module_name, None)
        if module:
            head_params.extend(list(module.parameters()))

    # Add head group
    param_groups.append({
        "params": head_params,
        "lr": config.head_lr,
        "weight_decay": config.weight_decay,
        "name": "heads",
    })

    # --- Backbone parameters with LLRD ---
    # Parse layer IDs from parameter names
    # Regex logic: 
    #   "feature_extractor" or "feature_projection" -> layer 0 (or frozen/base)
    #   "encoder.layers.N.xxx" -> layer N + 1
    #   Everything else (embeddings, final norm) -> layer 0 (base)
    
    layer_groups = {}
    
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
            
        if "feature_extractor" in name or "feature_projection" in name:
            layer_id = 0
        elif "encoder.layers" in name:
            try:
                # Extract N from "encoder.layers.N.attr"
                # Example: "encoder.layers.11.attention..." -> 11
                layer_id = int(name.split("encoder.layers.")[1].split(".")[0]) + 1
            except (IndexError, ValueError):
                layer_id = 0
        else:
            # Embeddings, final layer norm, etc.
            layer_id = 0

        if layer_id not in layer_groups:
            layer_groups[layer_id] = []
        layer_groups[layer_id].append(param)

    # Apply LLRD decay formula: lr_i = backbone_lr * decay^(L - i)
    #   where L = max layer index, i = current layer index, decay = 0.95
    #
    # Example for WavLM-Large (24 encoder layers, L=24):
    #   Layer 0 (CNN/embeddings):  lr = 1e-5 * 0.95^24 ≈ 2.9e-7  (preserve acoustic features)
    #   Layer 12 (middle):         lr = 1e-5 * 0.95^12 ≈ 5.4e-6  (moderate adaptation)
    #   Layer 24 (final):          lr = 1e-5 * 0.95^0  = 1.0e-5  (full task adaptation)
    #   Heads (classifier, etc.):  lr = 1e-3                      (random init, fast learning)
    #
    # Intuition (Clark et al., EMNLP 2020): Pre-trained lower layers encode general
    # acoustic knowledge that transfers well; aggressive updates would destroy it.
    # Higher layers are more task-specific and benefit from larger updates.
    if layer_groups:
        max_layer = max(layer_groups.keys())
        for layer_id, params in layer_groups.items():
            # Higher layer_id (closer to output) -> smaller exponent -> higher LR
            # Lower layer_id (closer to input) -> larger exponent -> lower LR
            scale = config.llrd_decay ** (max_layer - layer_id)
            
            param_groups.append({
                "params": params,
                "lr": config.backbone_lr * scale,
                "weight_decay": config.weight_decay,
                "name": f"backbone_layer_{layer_id}",
            })

    return torch.optim.AdamW(param_groups)
