
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    context_window: int = 512
    batch_size: int = 2
    val_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    
    learning_rate: float = 4e-6
    min_lr: float = 4e-7
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    validation_split: float = 0.01
    validation_steps: int = 312
    max_epochs: int = 3
    seed: int = 14
    
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    model : str = "tiny_tiny_tiny_LLaMA"
    save_dir: str = "checkpoints"
    huggingface_repo_id: str = "paulo037/tinyllama"
    dataset_id: str = "paulo037/slimpajama"
    
    save_checkpoint : bool = True