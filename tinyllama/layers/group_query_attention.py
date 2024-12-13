from ..config import Config
import torch
from torch import nn, Tensor
from einops import rearrange
from tinyllama.layers.rotary_embedding import apply_rotary_pos_emb


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()

        self.dtype = config.dtype
        self.dim = config.n_embd
        self.query_heads = config.n_head
        self.queries_per_kv = config.n_head // config.n_query_groups
        self.key_value_heads = config.n_head // self.queries_per_kv

        self.kv_dim = self.dim // self.query_heads * self.key_value_heads

        self.q_proj = nn.Linear(self.dim, self.dim,
                                bias=False, dtype=self.dtype)
        self.k_proj = nn.Linear(self.dim, self.kv_dim,
                                bias=False, dtype=self.dtype)
        self.v_proj = nn.Linear(self.dim, self.kv_dim,
                                bias=False, dtype=self.dtype)

        self.o_proj = nn.Linear(self.dim, self.dim,
                                bias=False, dtype=self.dtype)
        self.config = config

    def scaled_dot_product_gqa(self, query: Tensor, key: Tensor, value: Tensor):
        scale_factor = 1 / query.size(-1) ** 0.5

        L, S = query.size(-2), key.size(-2)

        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias = attn_bias.to(query.dtype).to(query.device)

        key = torch.repeat_interleave(key, self.queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, self.queries_per_kv, dim=1)

        attn_weight = query @ key.transpose(-1, -2) * scale_factor

        attn_weight += attn_bias

        attn_weight = torch.softmax(attn_weight, dim=-1)
        y = attn_weight @ value

        return y

    def forward(self,
                x: Tensor,
                position_embeddings: Tensor) -> Tensor:

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.key_value_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.key_value_heads)

        cos, sin = position_embeddings

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        y = self.scaled_dot_product_gqa(
            query=q,
            key=k,
            value=v
        )

        y = rearrange(y, "b h n d -> b n (h d)")

        y = self.o_proj(y)

        return y
