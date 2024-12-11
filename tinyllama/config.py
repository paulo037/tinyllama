from dataclasses import dataclass
import torch


@dataclass
class Config:
    name: str = "tiny_LLaMA_1b"

    seq_length: int = 2048
    vocab_size: int = 32000
    
    n_layer: int = 22
    n_head: int = 32
    n_embd: int = 2048
    hidden_dim: int = 5632
    n_query_groups: int = 4

    base=10000
    rotary_percentage: float = 1.0
    
    device: str = "cpu"
    dtype=torch.float16
    
    stop_token_id = 2

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head
      
    @property
    def dim(self) -> int:
        return int(self.rotary_percentage * self.head_size)

configs = [
    dict(
        name="tiny_LLaMA_1b",
        seq_length=2048,
        vocab_size=32000,
        n_layer=22,
        n_head=32,
        n_embd=2048,
        rotary_percentage=1.0,
        hidden_dim=5632,
        n_query_groups=4,
    ),
    dict(
        name="tiny_tiny_LLaMA",
        seq_length=256,
        vocab_size=32000,
        n_layer=12,
        n_head=8,
        n_query_groups=4,
        n_embd=768,
        rotary_percentage=1.0,
        hidden_dim=512,
    )
]

name_to_config: dict[str, Config] = {
    config["name"]: Config(**config) for config in configs}