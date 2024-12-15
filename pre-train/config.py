
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    context_window: int = 512
    batch_size: int = 2
    val_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    
    learning_rate: float = 4e-6
    min_lr: float = 4e-7
    warmup_steps: int = 200
    
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    validation_split: float = 0.01
    validation_steps: int = 156
    max_epochs: int = 3
    seed: int = 14
    
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    model : str = "tiny_tiny_tiny_LLaMA"
    save_dir: str = "checkpoints"
    huggingface_repo_id: str = "paulo037/tinyllama"
    dataset_id: str = "paulo037/slimpajama"
    
    warmup_dataset_id: str = None
    dataset_warmup_ratio : int = 0.018
    
    save_checkpoint : bool = True
    checkpoint: str = None


def configure_training_args(parser, training_config):
  parser.add_argument("--context_window", type=int, required=False, default=training_config.context_window, help="Size of the context window.")
  parser.add_argument("--batch_size", type=int, required=False, default=training_config.batch_size, help="Batch size for training.")
  parser.add_argument("--val_batch_size", type=int, required=False, default=training_config.val_batch_size, help="Batch size for validation.")
  parser.add_argument("--gradient_accumulation_steps", type=int, required=False, default=training_config.gradient_accumulation_steps, help="Number of gradient accumulation steps.")
  
  parser.add_argument("--learning_rate", type=float, required=False, default=training_config.learning_rate, help="Learning rate.")
  parser.add_argument("--min_lr", type=float, required=False, default=training_config.min_lr, help="Minimum learning rate.")
  parser.add_argument("--warmup_steps", type=int, required=False, default=training_config.warmup_steps, help="Number of warmup steps.")
  
  parser.add_argument("--weight_decay", type=float, required=False, default=training_config.weight_decay, help="Weight decay.")
  parser.add_argument("--grad_clip", type=float, required=False, default=training_config.grad_clip, help="Gradient clipping value.")
  
  parser.add_argument("--validation_split", type=float, required=False, default=training_config.validation_split, help="Fraction of data to be used for validation.")
  parser.add_argument("--validation_steps", type=int, required=False, default=training_config.validation_steps, help="Number of validation steps.")
  parser.add_argument("--max_epochs", type=int, required=False, default=training_config.max_epochs, help="Maximum number of epochs.")
  parser.add_argument("--seed", type=int, required=False, default=training_config.seed, help="Random seed.")
  
  parser.add_argument("--bos_token_id", type=int, required=False, default=training_config.bos_token_id, help="Beginning of sentence token ID.")
  parser.add_argument("--eos_token_id", type=int, required=False, default=training_config.eos_token_id, help="End of sentence token ID.")
  parser.add_argument("--pad_token_id", type=int, required=False, default=training_config.pad_token_id, help="Padding token ID.")
  
  parser.add_argument("--model", type=str, required=False, default=training_config.model, help="Name of the model to be used.")
  parser.add_argument("--save_dir", type=str, required=False, default=training_config.save_dir, help="Directory to save checkpoints.")
  parser.add_argument("--huggingface_repo_id", type=str, required=False, default=training_config.huggingface_repo_id, help="Huggingface repository ID.")
  parser.add_argument("--dataset_id", type=str, required=False, default=training_config.dataset_id, help="ID of the dataset to be used.")
  
  parser.add_argument("--save_checkpoint", type=bool, required=False, default=training_config.save_checkpoint, help="If set to True, will save the model checkpoint.")
  parser.add_argument("--checkpoint", type=str, required=False, help="Path to load the checkpoint.")
  parser.add_argument("--load_weights", type=str, required=False, default=None, help="Path to load the model weights.")
  
  parser.add_argument("--warmup_dataset_id", type=str, required=False, default=training_config.warmup_dataset_id, help="ID of the warmup dataset to be used.")
  parser.add_argument("--dataset_warmup_ratio", type=float, required=False, default=training_config.dataset_warmup_ratio, help="Ratio of the dataset to be used for warmup.")
  
  args = parser.parse_args()
  
  for key, value in vars(args).items():
      if hasattr(training_config, key):
          setattr(training_config, key, value)
  
  return training_config