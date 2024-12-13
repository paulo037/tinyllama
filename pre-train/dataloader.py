
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from .config import TrainingConfig
from torch.nn.utils.rnn import pad_sequence

def prepare_data(config: TrainingConfig):
    dataset = load_dataset(config.dataset_id, split="train")  # Substitua pelo nome do dataset no HuggingFace
    dataset = dataset.select_columns("input_ids")
    #dataset = dataset.shuffle(seed=config.seed).select(range(1000))
    def tokenize_and_trim(example):
        input_ids = example["input_ids"]
        if len(input_ids) > config.context_window:
            input_ids = input_ids[:-1]
            input_ids = input_ids[:config.context_window] + [config.eos_token_id]
        return {"input_ids": input_ids}

    tokenized_dataset = dataset.map(tokenize_and_trim, batched=False)
    train_size = int((1 - config.validation_split) * len(tokenized_dataset))
    val_size = len(tokenized_dataset) - train_size

    train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, val_size])

    return train_dataset, val_dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)

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
    CustomDataset(train_dataset), 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=lambda x: custom_collate_fn(x, pad_token_id=config.pad_token_id)
    )
    val_loader = DataLoader(
        CustomDataset(val_dataset), 
        batch_size=config.val_batch_size,
        collate_fn=lambda x: custom_collate_fn(x, pad_token_id=config.pad_token_id)
    )

    
    return train_loader, val_loader
    
