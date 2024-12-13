from .config import Config
import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional, Tuple
from .layers import Decoder, RMSNorm, RotaryEmbedding, FeedForward, GroupQueryAttention
from collections import OrderedDict
from safetensors.torch import load_file
import math

class TinyLlama(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd , dtype=config.dtype)
        self.layers       = nn.ModuleList([Decoder(config) for _ in range(config.n_layer)])
        self.norm         = RMSNorm(config)
        self.lm_head      = nn.Linear(config.n_embd, config.vocab_size, bias=False, dtype=config.dtype)
        
        self.rope = RotaryEmbedding(config)
        self.rope_cache: Optional[Tuple[Tensor, Tensor]] = None
    
    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX       
        for name, p in module.named_parameters():
            if (name == "o_proj.weight" and isinstance(module, GroupQueryAttention)) or (name == "up_proj.weight" and isinstance(module, FeedForward) ): 
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)
        
    def get_kv_head_dim(self, config: Config) -> int:
        queries_per_kv = config.n_head // config.n_query_groups
        key_value_heads = config.n_head // queries_per_kv
        kv_dim = config.hidden_dim // config.n_head * key_value_heads
        return kv_dim

    def build_rope_cache(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.arange(self.config.seq_length, device=idx.device).unsqueeze(0)
        return self.rope(idx, position_ids)
      
    def forward(
        self,
        input_ids: torch.LongTensor = None,
    ) -> Tensor:
      
        B, T = input_ids.shape
        
        assert T <= self.config.seq_length, f"Input length {T} exceeds maximum model length {self.config.seq_length}"
        
        max_seq_length = self.config.seq_length
        x = self.embed_tokens(input_ids)  
        
        if self.rope_cache is None :
            self.rope_cache = self.build_rope_cache(x)
        
        cos, sin = self.rope_cache
        
        cos = cos[:, :T]
        sin = sin[:, :T]

        for block in self.layers:
            x = block(x, (cos, sin), max_seq_length)

        x = self.norm(x)
    
        return self.lm_head(x) 

    def generate(self, input_ids: torch.LongTensor, max_length: int = Optional[100], sample: Optional[bool] = False) -> torch.LongTensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(input_ids)
                if sample:
                    next_token = torch.multinomial(logits[:, -1].softmax(dim=-1), num_samples=1)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1).unsqueeze(-1)
                    
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token.item() == self.config.stop_token_id:
                    break
        return input_ids



def remap_state_dict(state_dict):
    """
    Remaps the state dict keys from LlamaForCausalLM format to your custom model format
    """
    new_state_dict = OrderedDict()

    key_mapping = {
        'model.embed_tokens': 'embed_tokens',
        'model.norm': 'norm',
        'model.layers': 'layers',
        'self_attn': 'attn',
        'input_layernorm': 'norm1',
        'post_attention_layernorm': 'norm2',
        'mlp': 'ffn'
    }
    
    for key, value in state_dict.items():
        new_key = key
        
        for old, new in key_mapping.items():
            new_key = new_key.replace(old, new)
            
        if 'attn' in new_key:
  
            if value.shape != state_dict[key].shape:
                raise ValueError(f"Incompatible shape for attention weights: {key}")
                
        new_state_dict[new_key] = value
    
    return new_state_dict

def load_model_weights(model, safetensors_path):
    """
    Loads and remaps weights from a safetensors file to your custom model
    
    Args:
        model: Your custom model instance
        safetensors_path: Path to the safetensors file
    """
    try:
        original_state_dict = load_file(safetensors_path)
        
        new_state_dict = remap_state_dict(original_state_dict)
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        print("Model loaded successfully!")
        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)
            
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
