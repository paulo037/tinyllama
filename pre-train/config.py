
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    context_window: int = 256
    batch_size: int = 8
    val_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 4e-4
    min_lr: float = 4e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    validation_split: float = 0.01
    validation_steps: int = 50
    max_epochs: int = 3
    seed: int = 14
    
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    model : str = "tiny_tiny_tiny_LLaMA"
    save_dir: str = "checkpoints"
    huggingface_repo_id: str = "paulo037/tinyllama"
    dataset_id: str = "paulo037/slimpajama"