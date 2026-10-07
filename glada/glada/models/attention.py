"""
Hierarchical attention pooling with per-layer normalization, attentive statistics
pooling, and CKA diversity regularization.

This module implements the core representation-learning pipeline between the SSL
backbone and the VIB bottleneck (see ``vib.py``).  The pipeline is:

    SSL hidden states  -->  PerLayerNorm  -->  HierarchicalAttentionPooling
        (layer grouping + learnable weighted sum + ASP per group)
        -->  CKA diversity loss  -->  linear projection  -->  pooled output

Components
----------
1. **PerLayerNorm** -- Independent LayerNorm (Ba et al., 2016) per SSL hidden
   state.  Motivation: El-Kheir et al. (NAACL 2025) show SSL layers differ in
   activation magnitude by 10-100x.  Without normalization the later (larger)
   layers dominate gradients during backpropagation.

   Equation (per-layer, matching chap3.tex notation):
     h'_i = gamma_i * (h_i - E[h_i]) / sqrt(Var[h_i] + eps) + beta_i

2. **AttentiveStatisticsPooling (ASP)** -- Temporal pooling via attention-weighted
   mean and standard deviation.
   Reference: Okabe et al., "Attentive Statistics Pooling for Deep Speaker
   Embedding", Odyssey 2018.  Originally proposed for speaker verification;
   captures both the distributional center (mean) and spread (std) of frame-level
   features, which is critical for detecting synthesis artifacts that manifest as
   abnormal variance patterns in SSL features.

3. **HierarchicalAttentionPooling** -- Divide the L+1 SSL hidden states into K
   contiguous groups, apply a learnable softmax-weighted sum within each group
   followed by ASP, concatenate the K group outputs, and project.
   Inspired by HierCon (Liang et al., 2026, arxiv 2602.01032).

4. **CKA Diversity Loss** -- Penalize representational similarity between group
   outputs to prevent feature collapse and enforce multi-scale specialization.
   CKA definition: Kornblith et al., "Similarity of Neural Network
   Representations Revisited", ICML 2019.
   Application as diversity loss in deepfake detection: Tran et al., ACM
   Multimedia 2025 (arxiv 2509.03409).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerLayerNorm(nn.Module):
    """Per-layer normalization for SSL hidden states (Ba et al., 2016).

    Applies an independent ``nn.LayerNorm`` to each of the L+1 hidden states
    produced by the SSL backbone before they enter the hierarchical attention
    pooling stage.

    Equation (chap3.tex Section 3.3):
        h'_i = gamma_i * (h_i - E[h_i]) / sqrt(Var[h_i] + eps) + beta_i

    where gamma_i and beta_i are learnable affine parameters for layer i.

    Why per-layer (not shared) normalization:
        El-Kheir et al. (NAACL 2025) demonstrate that SSL layers differ in
        activation magnitude by 10-100x.  A single shared LayerNorm would
        either under-normalize large layers or over-normalize small ones.
        Per-layer norms bring every layer to unit scale so that the
        downstream learnable weighted sum in ``HierarchicalAttentionPooling``
        reflects genuine information content, not magnitude artifacts.

    Args:
        num_layers: Number of hidden states (L+1, including CNN feature
            projection output at index 0).
        hidden_dim: Feature dimension D of each hidden state.
    """

    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, hidden_states):
        """Apply per-layer normalization.

        Args:
            hidden_states: List of L+1 tensors, each (B, T, D).

        Returns:
            List of L+1 normalized tensors, each (B, T, D).
        """
        return [norm(h) for norm, h in zip(self.norms, hidden_states)]


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling (ASP).

    Reference: Okabe et al., "Attentive Statistics Pooling for Deep Speaker
    Embedding", Odyssey 2018.

    Equations (chap3.tex Section 3.3, matching notation exactly):
        Attention energy:   e_t = v^T tanh(W * h_tilde_{k,t} + b)
        Attention weights:  alpha_t = exp(e_t) / sum_t' exp(e_t')
        Weighted mean:      mu_k = sum_t alpha_t * h_tilde_{k,t}        in R^D
        Weighted std:       sigma_k = sqrt(sum_t alpha_t * (h_tilde_{k,t} - mu_k)^2 + eps)  in R^D
        Group output:       g_k = [mu_k ; sigma_k]                      in R^{2D}

    Why ASP over simple mean pooling:
        Mean pooling discards temporal variance -- yet synthesis artifacts often
        manifest as abnormal variance patterns (e.g., overly smooth vocoder
        outputs have lower frame-level variance than natural speech).  ASP
        captures both distributional center (mean) and spread (std), giving
        the downstream classifier richer statistics for discrimination.

    Architecture:
        Two-layer MLP with tanh activation computes scalar attention logits
        per frame, followed by softmax normalization.  The hidden dimension
        (128) is deliberately small to avoid overfitting to temporal patterns
        in the training domain.

    Args:
        input_dim: Feature dimension D of the input frames.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x, mask=None):
        """Compute attention-weighted mean and std pooling.

        Args:
            x: (B, T, D) frame-level features for one group.
            mask: Optional (B, T) boolean mask. True = real frame, False = padded.
                When provided, padded frames are set to -inf before softmax
                so they receive zero attention weight (Vaswani et al., 2017).

        Returns:
            (B, 2*D) concatenation of weighted mean mu_k and weighted std sigma_k.
        """
        # e_t = v^T tanh(W * h_tilde_{k,t} + b)
        e = self.attention(x)                         # (B, T, 1) attention logits
        # Mask padded frames: set to -inf so softmax gives them zero weight.
        if mask is not None:
            e = e.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        # alpha_t = softmax(e_t) over the time dimension
        w = F.softmax(e, dim=1)                       # (B, T, 1) attention weights
        # Edge case: if ALL frames are masked (corrupt file, actual_length=0),
        # softmax(-inf, ..., -inf) = NaN. Replace with zeros — the resulting
        # zero vector is harmless (corrupt file rate ~0.07%).
        w = w.nan_to_num(0.0)
        # mu_k = sum_t alpha_t * h_tilde_{k,t}
        mu = (x * w).sum(dim=1)                       # (B, D)   weighted mean
        # sigma_k = sqrt(sum_t alpha_t * (h_tilde_{k,t} - mu_k)^2 + eps)
        # eps=1e-6 prevents NaN gradients when variance is near zero
        sigma = torch.sqrt(((x - mu.unsqueeze(1))**2 * w).sum(dim=1) + 1e-6)  # (B, D) weighted std
        # g_k = [mu_k ; sigma_k] in R^{2D}
        return torch.cat([mu, sigma], dim=1)          # (B, 2*D)


def compute_cka(X, Y):
    """Compute linear CKA between two representation matrices.

    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019.

    Mathematical derivation (chap3.tex Section 3.3):
        Given X, Y in R^{B x D} (centered: X <- X - mean(X), Y <- Y - mean(Y)):

        Linear kernel matrices:
            K_X = X @ X^T,   K_Y = Y @ Y^T       (both B x B)

        HSIC (Hilbert-Schmidt Independence Criterion):
            HSIC(G_i, G_j) = (1/B^2) * tr(G_i @ G_i^T @ G_j @ G_j^T)

        CKA normalizes HSIC to [0, 1]:
            CKA(G_i, G_j) = HSIC(G_i, G_j) / sqrt(HSIC(G_i, G_i) * HSIC(G_j, G_j))

        CKA = 1 means identical representations (up to invertible linear transform).
        CKA = 0 means completely dissimilar representations.

    Usage as diversity loss (Tran et al., ACM MM 2025, arxiv 2509.03409):
        L_CKA = (1 / C(K,2)) * sum_{i<j} CKA(G_i, G_j)

        Minimizing L_CKA penalizes high similarity between group outputs,
        preventing feature collapse where all K groups learn redundant
        representations.  This encourages each group to specialize on
        different aspects of the SSL hierarchy (acoustic, phonetic,
        semantic, artifact-specific features).

    Args:
        X: (B, D) feature matrix -- one group's ASP output g_i = [mu_i; sigma_i].
        Y: (B, D) feature matrix -- another group's ASP output g_j.

    Returns:
        Scalar CKA similarity in [0, 1].
    """
    # Center columns (subtract feature-wise mean across batch)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    # HSIC(X,Y) = (1/B^2) * tr(K_X @ K_Y).  The 1/B^2 factor cancels in the
    # CKA ratio, so we omit it for efficiency.
    # Implementation note: element-wise product + sum is equivalent to tr(K_X @ K_Y)
    # but avoids materializing the full B x B matrix product.
    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
    # eps=1e-8 prevents division by zero when a group collapses to constant output
    return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)


class HierarchicalAttentionPooling(nn.Module):
    """Hierarchical grouped-layer attention pooling with CKA diversity.

    Divides the L+1 SSL hidden states into K contiguous groups, computes a
    learnable softmax-weighted sum within each group, applies ASP (Okabe et al.,
    Odyssey 2018) per group, and projects the concatenated group outputs to the
    final pooled representation.

    Equations (chap3.tex Section 3.3):
        Layer grouping:
            L+1 hidden states divided into K contiguous groups G_1, ..., G_K.

        Per-group weighted sum:
            w_k = softmax(theta_k)
            h_tilde_k = sum_{i in G_k} w_{k,i} * h'_i        (B, T, D)

        Per-group ASP (temporal pooling):
            e_t = v^T tanh(W * h_tilde_{k,t} + b)
            alpha_t = exp(e_t) / sum_t' exp(e_t')
            mu_k = sum_t alpha_t * h_tilde_{k,t}              in R^D
            sigma_k = sqrt(sum_t alpha_t * (h_tilde_{k,t} - mu_k)^2 + eps)
            g_k = [mu_k ; sigma_k]                            in R^{2D}

        Group projection:
            pooled = W_proj * [g_1 ; ... ; g_K] + b_proj      in R^D

        CKA diversity loss:
            L_CKA = (1/C(K,2)) * sum_{i<j} CKA(G_i, G_j)

    Hierarchy of attention levels:
        1. Intra-group layer attention (learnable weighted sum theta_k)
        2. Intra-group temporal attention (ASP: frame-level alpha_t)
        3. Inter-group fusion (concatenation + linear projection)

    References:
        - Liang et al. (2026, arxiv 2602.01032) -- HierCon: grouped layer
          attention on XLS-R for audio deepfake detection.
        - Pan et al. (Interspeech 2024) -- Attentive merging of WavLM hidden
          embeddings for speaker verification.
        - Kornblith et al. (ICML 2019) -- CKA similarity metric.
        - Tran et al. (ACM MM 2025) -- CKA diversity loss for deepfake detection.

    Args:
        num_hidden_states: Total number of SSL hidden states (L+1).
        hidden_dim: Feature dimension D of each hidden state.
        num_groups: Number of layer groups K.
        output_dim: Dimension of the final projected output.
    """

    def __init__(self, num_hidden_states, hidden_dim, num_groups, output_dim):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim

        # Assign layers to groups (evenly, last group gets remainder).
        # Intuition (El-Kheir et al., NAACL 2025; HierCon, Liang et al. 2026):
        #   SSL encoder layers form a hierarchy of abstraction levels:
        #     Group 0 (layers 0-5):   Acoustic features (pitch, formants, spectral shape)
        #     Group 1 (layers 6-11):  Phonetic/phonological features
        #     Group 2 (layers 12-17): Word-level / semantic features
        #     Group 3 (layers 18-23): Task-specific patterns (synthesis artifacts)
        #   Grouping enables GLADA to apply separate domain adaptation at each level,
        #   and CKA diversity loss to enforce distinct group specializations.
        layers_per_group = num_hidden_states // num_groups
        self.group_assignments = []
        for g in range(num_groups):
            start = g * layers_per_group
            end = start + layers_per_group if g < num_groups - 1 else num_hidden_states
            self.group_assignments.append(list(range(start, end)))

        # Per-group learnable layer weights
        self.group_weights = nn.ParameterList([
            nn.Parameter(torch.ones(len(layers)) / len(layers))
            for layers in self.group_assignments
        ])

        # Per-group ASP (Okabe et al., 2018)
        self.group_asp = nn.ModuleList([
            AttentiveStatisticsPooling(hidden_dim)
            for _ in range(num_groups)
        ])

        # Projection: concat of K groups * 2*D -> output_dim
        self.projection = nn.Linear(num_groups * 2 * hidden_dim, output_dim)

    def forward(self, hidden_states, frame_mask=None):
        """Compute hierarchical pooling, CKA loss, and per-group outputs.

        Args:
            hidden_states: List of L+1 tensors, each (B, T, D), already
                passed through ``PerLayerNorm``.
            frame_mask: Optional (B, T) boolean mask from sample-to-frame
                conversion in ``GLADAModel.forward()``. True = real frame,
                False = zero-padded. Passed to ASP to exclude padded frames
                from the attention softmax.

        Returns:
            pooled: (B, output_dim) -- projected representation for VIB/classifier.
            cka_loss: Scalar -- mean pairwise CKA between group outputs
                (L_CKA in the total loss).
            group_outputs: List of K tensors, each (B, 2*D) -- per-group ASP
                outputs g_k, used by GLADA domain discriminators.
        """
        # Optional mask diagnostics (set by trainer for first N batches)
        if getattr(self, '_debug_mask_batches', 0) > 0 and frame_mask is not None:
            self._debug_mask_batches -= 1
            frac_padded = 1.0 - frame_mask.float().mean().item()
            frames_per_sample = frame_mask.sum(1)
            print(f"  [MASK-POOL] frame_mask: shape={frame_mask.shape}, "
                  f"padded={frac_padded:.1%}, "
                  f"frames=[{frames_per_sample.min().item()}-{frames_per_sample.max().item()}]")

        group_outputs = []

        for g, (layer_ids, weights) in enumerate(
                zip(self.group_assignments, self.group_weights)):
            # Stack layers belonging to group G_k: (|G_k|, B, T, D)
            layers = torch.stack([hidden_states[i] for i in layer_ids])
            # w_k = softmax(theta_k), reshaped for broadcasting over (B, T, D)
            w = F.softmax(weights, dim=0).view(-1, 1, 1, 1)
            # h_tilde_k = sum_{i in G_k} w_{k,i} * h'_i
            group_feat = (layers * w).sum(dim=0)       # (B, T, D)
            # g_k = ASP(h_tilde_k) = [mu_k ; sigma_k] in R^{2D}
            group_pooled = self.group_asp[g](group_feat, mask=frame_mask)  # (B, 2*D)
            group_outputs.append(group_pooled)

        # pooled = W_proj * [g_1 ; ... ; g_K] + b_proj
        concat = torch.cat(group_outputs, dim=1)       # (B, K * 2*D)
        pooled = self.projection(concat)                # (B, output_dim)

        # L_CKA = (1/C(K,2)) * sum_{i<j} CKA(g_i, g_j)
        # Encourages diverse group representations (Tran et al., ACM MM 2025)
        cka_loss = torch.tensor(0.0, device=pooled.device)
        n_pairs = 0
        for i in range(len(group_outputs)):
            for j in range(i + 1, len(group_outputs)):
                cka_loss = cka_loss + compute_cka(group_outputs[i], group_outputs[j])
                n_pairs += 1
        if n_pairs > 0:
            cka_loss = cka_loss / n_pairs  # Average over C(K,2) pairs

        return pooled, cka_loss, group_outputs
