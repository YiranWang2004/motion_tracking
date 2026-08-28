"""Inference-only network matching the released ScaleBFM checkpoint names."""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn


class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        position = torch.arange(
            value.shape[1], device=value.device, dtype=torch.float32
        )
        frequencies = torch.outer(position, self.inv_freq)
        cosine = frequencies.cos().unsqueeze(0)
        sine = frequencies.sin().unsqueeze(0)
        even = value[..., 0::2]
        odd = value[..., 1::2]
        output = torch.empty_like(value)
        output[..., 0::2] = even * cosine - odd * sine
        output[..., 1::2] = even * sine + odd * cosine
        return output


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        norm = value.norm(dim=-1, keepdim=True) / value.shape[-1] ** 0.5
        return self.scale * value / (norm + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.silu(self.w(value)) * self.v(value))


class HumanoidTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.feed_forward = SwiGLU(embed_dim, ff_dim)
        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)
        self.rmsnorm3 = RMSNorm(embed_dim)
        self.cond_norm = RMSNorm(embed_dim)
        self.rope = RoPEPositionalEncoding(embed_dim)

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.rmsnorm1(value)
        rotated = self.rope(normalized)
        attention = self.self_attention(
            rotated, rotated, normalized, attn_mask=self_attn_mask
        )[0]
        value = value + attention
        normalized = self.rmsnorm2(value)
        condition = self.cond_norm(condition)
        value = value + self.cross_attention(normalized, condition, condition)[0]
        return value + self.feed_forward(self.rmsnorm3(value))


class TaskEmbedder(nn.Module):
    def __init__(
        self,
        task_obs_dim: int,
        embedding_dim: int,
        reduced_task_dim: int | None = None,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        output_dim = reduced_task_dim or embedding_dim
        dimensions = [task_obs_dim, *(hidden_dims or []), output_dim]
        layers: list[nn.Module] = []
        for index, (input_dim, next_dim) in enumerate(pairwise(dimensions)):
            layers.append(nn.Linear(input_dim, next_dim))
            if index < len(dimensions) - 2:
                layers.append(nn.ELU())
        self.task_projection = layers[0] if len(layers) == 1 else nn.Sequential(*layers)
        self.reduced_task_dim = reduced_task_dim
        if reduced_task_dim is not None:
            value = torch.randn(embedding_dim, reduced_task_dim, dtype=torch.float32)
            orthogonal, upper = torch.linalg.qr(value, mode="reduced")
            sign = torch.sign(torch.diag(upper))
            sign[sign == 0] = 1.0
            self.register_buffer("W", orthogonal * sign)

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        embedding = self.task_projection(task)
        if self.reduced_task_dim is None:
            return embedding
        embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1.0e-8)
        return torch.matmul(embedding, self.W.T)


class HumanoidTransformer(nn.Module):
    def __init__(
        self,
        prop_obs_dim: int,
        action_dim: int,
        output_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.prop_projection = nn.Linear(prop_obs_dim, embed_dim)
        self.action_projection = nn.Linear(action_dim, embed_dim)
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer_blocks = nn.ModuleList(
            HumanoidTransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(embed_dim)
        self.projection_head = nn.Linear(embed_dim, output_dim)

    def forward(
        self,
        prop_obs: torch.Tensor,
        action_obs: torch.Tensor,
        task_tokens: torch.Tensor,
    ) -> torch.Tensor:
        prop = self.prop_projection(prop_obs)
        action = self.action_projection(action_obs)
        context = prop.new_empty(prop.shape[0], 2 * prop.shape[1] - 1, self.embed_dim)
        context[:, 0::2] = prop
        context[:, 1::2] = action[:, 1:]
        value = torch.cat(
            (context, self.empty_embedding.expand(prop.shape[0], -1, -1)), dim=1
        )
        mask = torch.zeros(
            value.shape[1], value.shape[1], dtype=torch.bool, device=value.device
        )
        mask[:-1, -1] = True
        for block in self.transformer_blocks:
            value = block(value, task_tokens, self_attn_mask=mask)
        return self.projection_head(self.final_norm(value)[:, -1])
