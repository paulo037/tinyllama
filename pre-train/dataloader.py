
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from .config import TrainingConfig
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)

class WarmupDatasetWrapper(Dataset):
    def __init__(self, main_dataset, warmup_dataset, config: TrainingConfig):
        self.main_dataset = main_dataset
        self.warmup_dataset = warmup_dataset
        self.warmup_dataset_size = len(warmup_dataset)
        
        self.batch_size = config.batch_size * config.gradient_accumulation_steps
        self.max_epochs = config.max_epochs
        self.total_steps = len(self.main_dataset)  // self.batch_size * self.max_epochs
        self.warmup_steps = int(self.total_steps * config.dataset_warmup_ratio)

        self.warmup_per_batch =  np.linspace(0, 1, self.warmup_steps)
        self.warmup_per_batch = np.concatenate([self.warmup_per_batch, np.ones(self.total_steps - self.warmup_steps)])


        self.global_step = 0
        self.batch_step = 0
        self.warmup_batch = int(self.warmup_per_batch[self.global_step] * self.batch_size)
        
        

    def __len__(self):
        return len(self.main_dataset)

    def __getitem__(self, idx):
        
        if self.global_step <= self.warmup_steps and self.batch_step  >= self.warmup_batch:
            self.step()
            return torch.tensor(self.warmup_dataset[idx % self.warmup_dataset_size])
        else:
            self.step()
            return torch.tensor(self.main_dataset[idx])

    def step(self):
        if (self.batch_step +1)% self.batch_size == 0 and self.global_step < len(self.warmup_per_batch) - 1:
            self.warmup_batch = int(self.warmup_per_batch[self.global_step ] * self.batch_size)
            self.global_step += 1
            self.batch_step = 0

        self.batch_step += 1

def prepare_data(config: TrainingConfig):
    def tokenize_and_trim(example):
        input_ids = example["input_ids"]
        if len(input_ids) > config.context_window:
            input_ids = input_ids[:-1]
            input_ids = input_ids[:config.context_window] + [config.eos_token_id]
        return {"input_ids": input_ids}
      
      
    dataset = load_dataset(config.dataset_id, split="train")  
    dataset = dataset.select_columns("input_ids")
    dataset = dataset.shuffle(seed=config.seed)
    tokenized_dataset = dataset.map(tokenize_and_trim, batched=False)

    train_size = int((1 - config.validation_split) * len(tokenized_dataset))
    val_size = len(tokenized_dataset) - train_size
    
    train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, val_size])
    train_dataset, val_dataset = CustomDataset(train_dataset), CustomDataset(val_dataset)
    
    if config.warmup_dataset_id is None:
        return train_dataset, val_dataset
    
    
    warmup_dataset = load_dataset(config.warmup_dataset_id, split="train")  
    warmup_dataset = warmup_dataset.select_columns("input_ids")
    num_warmup_samples = int(len(tokenized_dataset) * config.dataset_warmup_ratio * config.max_epochs)
    num_warmup_samples = min(num_warmup_samples, len(warmup_dataset))
    warmup_dataset = warmup_dataset.shuffle(seed=config.seed).select(range(num_warmup_samples))
    tokenized_warmup_dataset = warmup_dataset.map(tokenize_and_trim, batched=False)
    
    train_size = int((1 - config.validation_split) * len(tokenized_warmup_dataset))
    val_size = len(tokenized_warmup_dataset) - train_size

    warmup_train_dataset, warmup_val_dataset = random_split(tokenized_dataset, [train_size, val_size])

    train_dataset = WarmupDatasetWrapper(
        train_dataset, 
        warmup_train_dataset, 
        config
    )

    val_dataset = WarmupDatasetWrapper(
        val_dataset, 
        warmup_val_dataset, 
        config
    )

    return train_dataset, val_dataset
          
def custom_collate_fn(batch, pad_token_id=0):
    """
    Custom collate function to pad sequences in a batch to the same length
    
    Args:
        batch (List[torch.Tensor]): List of input sequences
        pad_token_id (int): Token ID used for padding
    
    Returns:
        torch.Tensor: Padded batch tensor
    """
    
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    
    
    return padded_batch
  
def get_dataloaders(config):

    train_dataset, val_dataset = prepare_data(config)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=lambda x: custom_collate_fn(x, pad_token_id=config.pad_token_id)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.val_batch_size,
        collate_fn=lambda x: custom_collate_fn(x, pad_token_id=config.pad_token_id)
    )

    
    return train_loader, val_loader
    
