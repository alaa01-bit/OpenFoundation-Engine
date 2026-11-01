"""
OpenFoundation-Engine: Core Model Architecture
Implements modern Decoder-Only Foundation Model architectures:
- Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)
- Rotary Position Embeddings (RoPE) with frequency scaling
- SwiGLU Gated Feed-Forward Networks
- RMSNorm with Pre-normalization residuals
- Sparse Mixture-of-Experts (MoE) with Top-K Gating and Auxiliary Load Balancing Loss
- KV Cache integration for low-latency inference
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2          # GQA ratio: 8 / 2 = 4 queries share 1 KV head
    d_ffn: int = 688             # 8/3 * d_model (SwiGLU standard)
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    use_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 2
    moe_aux_loss_coeff: float = 0.01
    dropout: float = 0.0

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def gqa_groups(self) -> int:
        return self.n_heads // self.n_kv_heads


class RMSNorm:
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x shape: (..., dim)
        variance = np.mean(np.square(x), axis=-1, keepdims=True)
        norm_x = x / np.sqrt(variance + self.eps)
        return norm_x * self.weight


def precompute_rope_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Precomputes cosine and sine matrices for Rotary Position Embedding (RoPE)"""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)  # (max_seq_len, dim // 2)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos, sin


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Applies Rotary Position Embedding to Query or Key tensors.
    x: shape (batch, seq_len, n_heads, d_head)
    cos, sin: shape (seq_len, d_head // 2)
    """
    batch_size, seq_len, n_heads, d_head = x.shape
    x_half1 = x[..., 0::2]
    x_half2 = x[..., 1::2]

    cos_expanded = cos[:seq_len, np.newaxis, :]  # (seq_len, 1, d_head // 2)
    sin_expanded = sin[:seq_len, np.newaxis, :]

    # RoPE rotation: [x1, x2] * [cos -sin; sin cos]
    rot1 = x_half1 * cos_expanded - x_half2 * sin_expanded
    rot2 = x_half1 * sin_expanded + x_half2 * cos_expanded

    out = np.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


class SwiGLUFFN:
    """SwiGLU Gated Feed-Forward Network (Shazeer, 2020)"""
    def __init__(self, d_model: int, d_ffn: int, rng: np.random.Generator):
        scale = 1.0 / math.sqrt(d_model)
        self.w_gate = rng.normal(0, scale, (d_model, d_ffn)).astype(np.float32)
        self.w_up = rng.normal(0, scale, (d_model, d_ffn)).astype(np.float32)
        self.w_down = rng.normal(0, 1.0 / math.sqrt(d_ffn), (d_ffn, d_model)).astype(np.float32)

    def silu(self, x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        gate = self.silu(np.dot(x, self.w_gate))
        up = np.dot(x, self.w_up)
        activated = gate * up
        return np.dot(activated, self.w_down)


class MoELayer:
    """
    Sparse Mixture of Experts with Top-K Softmax Gating & Load Balancing Auxiliary Loss
    Matches modern foundation models (Mixtral, DeepSeek-V2/V3).
    """
    def __init__(self, config: ModelConfig, rng: np.random.Generator):
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.d_model = config.d_model
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        # Gate router weights
        scale = 1.0 / math.sqrt(config.d_model)
        self.router_weights = rng.normal(0, scale, (config.d_model, self.num_experts)).astype(np.float32)

        # Expert FFNs
        self.experts = [SwiGLUFFN(config.d_model, config.d_ffn, rng) for _ in range(self.num_experts)]
        self.last_aux_loss = 0.0
        self.last_expert_counts = np.zeros(self.num_experts, dtype=np.int32)

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.d_model)
        num_tokens = flat_x.shape[0]

        # 1. Compute router logits
        router_logits = np.dot(flat_x, self.router_weights)  # (tokens, num_experts)
        # Numerical stability
        exp_logits = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
        router_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # 2. Top-K routing
        top_k_indices = np.argsort(-router_probs, axis=-1)[:, :self.top_k]  # (tokens, top_k)
        top_k_weights = np.take_along_axis(router_probs, top_k_indices, axis=-1)
        # Normalize weights over top-k
        top_k_weights = top_k_weights / np.sum(top_k_weights, axis=-1, keepdims=True)

        # 3. Compute auxiliary load balancing loss (switch transformer / Mixtral formulation)
        # P_i = average routing probability for expert i
        # f_i = fraction of tokens routed to expert i
        P_i = np.mean(router_probs, axis=0)
        expert_assignment_mask = np.zeros((num_tokens, self.num_experts), dtype=np.float32)
        np.put_along_axis(expert_assignment_mask, top_k_indices, 1.0, axis=-1)
        f_i = np.mean(expert_assignment_mask, axis=0)
        aux_loss = float(self.num_experts * np.sum(P_i * f_i) * self.aux_loss_coeff)
        self.last_aux_loss = aux_loss
        self.last_expert_counts = np.sum(expert_assignment_mask, axis=0).astype(np.int32)

        # 4. Dispatch tokens to experts and aggregate outputs
        output = np.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            assigned_mask = np.any(top_k_indices == expert_id, axis=-1)
            if not np.any(assigned_mask):
                continue

            tokens_for_expert = flat_x[assigned_mask]
            expert_out = expert(tokens_for_expert)

            # Get weight for this expert in the top-k for each assigned token
            expert_k_mask = (top_k_indices[assigned_mask] == expert_id)
            weights = np.sum(top_k_weights[assigned_mask] * expert_k_mask, axis=-1, keepdims=True)

            output[assigned_mask] += expert_out * weights

        return output.reshape(orig_shape), aux_loss


class GroupedQueryAttention:
    """
    Grouped-Query Attention (GQA) with RoPE and KV Caching (Ainslie et al., 2023)
    Supports LLaMA-style Multi-Query and Grouped-Query attention mechanisms.
    """
    def __init__(self, config: ModelConfig, rng: np.random.Generator):
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.gqa_groups = config.gqa_groups
        self.scale = 1.0 / math.sqrt(self.d_head)

        proj_scale = 1.0 / math.sqrt(self.d_model)
        self.w_q = rng.normal(0, proj_scale, (self.d_model, self.n_heads * self.d_head)).astype(np.float32)
        self.w_k = rng.normal(0, proj_scale, (self.d_model, self.n_kv_heads * self.d_head)).astype(np.float32)
        self.w_v = rng.normal(0, proj_scale, (self.d_model, self.n_kv_heads * self.d_head)).astype(np.float32)
        self.w_o = rng.normal(0, proj_scale, (self.n_heads * self.d_head, self.d_model)).astype(np.float32)

    def __call__(
        self,
        x: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
        kv_cache: Optional[Dict[str, np.ndarray]] = None,
        start_pos: int = 0
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = np.dot(x, self.w_q).reshape(batch_size, seq_len, self.n_heads, self.d_head)
        k = np.dot(x, self.w_k).reshape(batch_size, seq_len, self.n_kv_heads, self.d_head)
        v = np.dot(x, self.w_v).reshape(batch_size, seq_len, self.n_kv_heads, self.d_head)

        # Apply RoPE
        q = apply_rope(q, cos[start_pos:start_pos + seq_len], sin[start_pos:start_pos + seq_len])
        k = apply_rope(k, cos[start_pos:start_pos + seq_len], sin[start_pos:start_pos + seq_len])

        # Manage KV Cache
        if kv_cache is not None:
            if "k" in kv_cache:
                k = np.concatenate([kv_cache["k"], k], axis=1)
                v = np.concatenate([kv_cache["v"], v], axis=1)
            kv_cache = {"k": k, "v": v}

        total_kv_len = k.shape[1]

        # Expand KV heads to match Q heads for GQA
        # q shape: (B, seq_len, n_heads, d_head) -> transpose to (B, n_heads, seq_len, d_head)
        q = np.transpose(q, (0, 2, 1, 3))
        # k shape: (B, total_kv_len, n_kv_heads, d_head) -> expand to n_heads
        k_expanded = np.repeat(k, self.gqa_groups, axis=2)
        v_expanded = np.repeat(v, self.gqa_groups, axis=2)
        k_expanded = np.transpose(k_expanded, (0, 2, 1, 3))
        v_expanded = np.transpose(v_expanded, (0, 2, 1, 3))

        # Compute Scaled Dot-Product Attention: Q @ K.T / sqrt(d)
        scores = np.matmul(q, np.transpose(k_expanded, (0, 1, 3, 2))) * self.scale

        # Causal mask if seq_len > 1
        if seq_len > 1:
            causal_mask = np.triu(np.full((seq_len, total_kv_len), -np.inf, dtype=np.float32), k=total_kv_len - seq_len + 1)
            scores = scores + causal_mask[np.newaxis, np.newaxis, :, :]

        # Softmax over key length
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Context output: Attn @ V
        context = np.matmul(attn_probs, v_expanded)  # (B, n_heads, seq_len, d_head)
        context = np.transpose(context, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)

        # Final projection
        out = np.dot(context, self.w_o)
        return out, kv_cache


class TransformerBlock:
    """Pre-Norm Transformer Block with GQA and SwiGLU / MoE"""
    def __init__(self, config: ModelConfig, rng: np.random.Generator):
        self.config = config
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config, rng)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        if config.use_moe:
            self.ffn = MoELayer(config, rng)
        else:
            self.ffn = SwiGLUFFN(config.d_model, config.d_ffn, rng)

    def __call__(
        self,
        x: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
        kv_cache: Optional[Dict[str, np.ndarray]] = None,
        start_pos: int = 0
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]], float]:
        # 1. Pre-norm Attention with residual
        normed_x = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(normed_x, cos, sin, kv_cache=kv_cache, start_pos=start_pos)
        h = x + attn_out

        # 2. Pre-norm FFN / MoE with residual
        normed_h = self.ffn_norm(h)
        if self.config.use_moe:
            ffn_out, aux_loss = self.ffn(normed_h)
        else:
            ffn_out = self.ffn(normed_h)
            aux_loss = 0.0

        out = h + ffn_out
        return out, new_kv_cache, aux_loss


class FoundationModel:
    """
    OpenFoundation Decoder-Only Model Architecture
    Integrates all modern innovations required by top foundation model labs.
    """
    def __init__(self, config: ModelConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Token embeddings
        embed_scale = 1.0 / math.sqrt(config.d_model)
        self.token_embeddings = self.rng.normal(0, embed_scale, (config.vocab_size, config.d_model)).astype(np.float32)

        # Transformer blocks
        self.layers = [TransformerBlock(config, self.rng) for _ in range(config.n_layers)]

        # Final RMSNorm & LM Head projection
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = self.rng.normal(0, embed_scale, (config.d_model, config.vocab_size)).astype(np.float32)

        # Precompute RoPE frequencies
        self.cos, self.sin = precompute_rope_freqs_cis(config.d_head, config.max_seq_len, config.rope_theta)

    def forward(
        self,
        token_ids: np.ndarray,
        kv_caches: Optional[List[Dict[str, np.ndarray]]] = None,
        start_pos: int = 0
    ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], float]:
        """
        Forward pass over token sequence.
        token_ids: shape (batch_size, seq_len)
        Returns: logits (batch, seq_len, vocab_size), updated kv_caches, total_aux_loss
        """
        batch_size, seq_len = token_ids.shape
        x = self.token_embeddings[token_ids]  # (batch_size, seq_len, d_model)

        new_kv_caches = []
        total_aux_loss = 0.0

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches else None
            x, updated_cache, aux_loss = layer(
                x, self.cos, self.sin, kv_cache=layer_cache, start_pos=start_pos
            )
            new_kv_caches.append(updated_cache)
            total_aux_loss += aux_loss

        normed_out = self.final_norm(x)
        logits = np.dot(normed_out, self.lm_head)  # (batch, seq_len, vocab_size)
        return logits, new_kv_caches, total_aux_loss

    def count_parameters(self) -> Dict[str, int]:
        """Calculates precise parameter count across architectural components."""
        cfg = self.config
        embed_params = cfg.vocab_size * cfg.d_model
        lm_head_params = cfg.d_model * cfg.vocab_size
        norm_params = cfg.d_model * (2 * cfg.n_layers + 1)

        # Attention: W_q + W_k + W_v + W_o
        attn_per_layer = (
            cfg.d_model * (cfg.n_heads * cfg.d_head) +
            2 * cfg.d_model * (cfg.n_kv_heads * cfg.d_head) +
            (cfg.n_heads * cfg.d_head) * cfg.d_model
        )

        # FFN / MoE
        if cfg.use_moe:
            ffn_per_expert = 3 * (cfg.d_model * cfg.d_ffn)
            ffn_per_layer = cfg.d_model * cfg.num_experts + cfg.num_experts * ffn_per_expert
        else:
            ffn_per_layer = 3 * (cfg.d_model * cfg.d_ffn)

        total_layer_params = cfg.n_layers * (attn_per_layer + ffn_per_layer)
        total_params = embed_params + lm_head_params + norm_params + total_layer_params

        return {
            "embedding_params": embed_params,
            "attention_params": attn_per_layer * cfg.n_layers,
            "ffn_params": ffn_per_layer * cfg.n_layers,
            "lm_head_params": lm_head_params,
            "total_params": total_params,
            "active_params_per_token": (
                total_params if not cfg.use_moe else
                embed_params + lm_head_params + norm_params +
                cfg.n_layers * (attn_per_layer + cfg.d_model * cfg.num_experts + cfg.top_k_experts * (3 * cfg.d_model * cfg.d_ffn))
            )
        }

    def estimate_flops_per_token(self) -> int:
        """
        Estimates FLOPs required per token in forward pass:
        Standard heuristic: ~ 2 * Active_Parameters
        """
        param_counts = self.count_parameters()
        active_params = param_counts["active_params_per_token"]
        return 2 * active_params
