import os
import torch
import random
import numpy as np
from transformers import get_cosine_schedule_with_warmup, AdamW
# from torch.optim import AdamW
import mlflow
from huggingface_hub import HfApi
from tinyllama import TinyLlama, name_to_config
from .config import TrainingConfig
from .dataloader import get_dataloaders
from functools import partial
import argparse
from dotenv import load_dotenv
import os
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

def set_seed(seed):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed (int): Seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(model, val_loader, step, pad_token_id):
    model.eval()
    perplexity = 0
    cross_entropy_loss = 0
    total_loss = 0.0 
    total_tokens = 0 

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            targets = targets.cuda() if torch.cuda.is_available() else targets

            outputs = model(inputs)

            logits = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
  
            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=pad_token_id)

            cross_entropy_loss += loss.item()
            total_loss += loss.item() * targets.size(0)  
            total_tokens += targets.size(0)  

    average_loss = total_loss / total_tokens 
    perplexity = torch.exp(torch.tensor(average_loss))
    avg_cross_entropy = cross_entropy_loss / len(val_loader)
    
    print(f"Validation perplexity: {perplexity}, cross entropy: {avg_cross_entropy}")
    
    mlflow.log_metric("validation_perplexity", perplexity.item(), step=step)
    mlflow.log_metric("validation_cross_entropy", avg_cross_entropy, step=step)

    model.train()

def save_checkpoint(model, optimizer, scheduler, step, epoch, config, run_id=None):
    checkpoint_path = os.path.join(config.save_dir, f"checkpoint_step_{step}.pt")
    mlruns_db_path = "mlruns.db"

    with open(mlruns_db_path, "rb") as f:
        mlruns_db_data = f.read()


    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "mlflow_run_id": run_id,
        "config": config,
        "mlruns_db": mlruns_db_data 
    }, checkpoint_path)

    api = HfApi()
    

    try:
      api.repo_info(repo_id=config.huggingface_repo_id, repo_type="model", token=HF_TOKEN)
    except Exception as e:
      if "404" in str(e):
        api.create_repo(repo_id=config.huggingface_repo_id, repo_type="model", token=HF_TOKEN)
      else:
        raise e
    
    api.upload_file(
      path_or_fileobj=checkpoint_path,
      path_in_repo=f"checkpoints/checkpoint_step_{step}.pt",
      repo_id=config.huggingface_repo_id,
      repo_type="model",
      token=HF_TOKEN
    )

def train(config, checkpoint=None, mlflow_run_id=None, load_weights=None):

    set_seed(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)
    
    train_loader, val_loader = get_dataloaders(config)
    
    config_llama = name_to_config[config.model]
    model = TinyLlama(config_llama)
    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        betas=(0.9, 0.95), 
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.max_epochs // config.gradient_accumulation_steps
    start_step = 0
    start_epoch = 0
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps, 
        num_training_steps=total_steps
    )
    
    if load_weights != None and checkpoint == None:
        model.load_state_dict(torch.load(load_weights))

    elif checkpoint != None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        
        print(f"Resuming training from checkpoint at step {start_step}, epoch {start_epoch}")
        
    else:
        model.apply(partial(model._init_weights, n_layer=config_llama.n_layer))
        
    
  
    model.train()
    model = model.cuda() if torch.cuda.is_available() else model

    step = start_step
    epoch = start_epoch
    optimizer.zero_grad()
    losses = []
    for epoch in range(start_epoch, config.max_epochs):
        for i, batch in enumerate(train_loader):
            if i / config.gradient_accumulation_steps < start_step:
                continue
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            targets = targets.cuda() if torch.cuda.is_available() else targets
            outputs = model(inputs)

            logits = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            
            
            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=config.pad_token_id)
            loss.backward()

            print(f"Epoch {epoch+1}, step {i + 1}, loss {loss.item()}")
            
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=i+1)
            
            if (i + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                losses.append((step, loss.item()))
                step += 1
                if step % config.validation_steps == 0:
                    for s, l in losses:
                        mlflow.log_metric("train_loss", l, step=s)
                        perplexity = torch.exp(torch.tensor(l))
                        mlflow.log_metric("train_perplexity", perplexity.item(), step=s)
                        
                    losses = []
                    validate(model, val_loader, step,  config.pad_token_id)
                    if config.save_checkpoint:
                        save_checkpoint(
                            model, optimizer, scheduler, step, epoch, 
                            config, mlflow_run_id
                        )

    save_checkpoint(model, optimizer, scheduler, step, epoch, config, mlflow_run_id)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train or evaluate a model.")
    training_config = TrainingConfig()
    
    parser.add_argument("--dataset_id", type=str, required=False, default=training_config.dataset_id, help="ID of the dataset to be used.")
    parser.add_argument("--model", type=str, required=False, default=training_config.model, help="Name of the model to be used.")
    
    parser.add_argument("--checkpoint", type=str, required=False,   help="Path to load the checkpoint.")
    parser.add_argument("--save_checkpoint", type=str, required=False, default=True, help="If set to True, will save the model checkpoint.")
    parser.add_argument("--load_weights", type=str, required=False, default=None, help="Path to load the model weights.")
    
    args = parser.parse_args()
    
    training_config.dataset_id = args.dataset_id
    training_config.model = args.model
    training_config.save_checkpoint = args.save_checkpoint
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_config.device = device
    
    resume_checkpoint =  args.checkpoint
    
    mlflow.set_tracking_uri(f"sqlite:///mlruns.db")

    run_id = None
    checkpoint = None
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        run_id = checkpoint.get('mlflow_run_id') 
    
    active_run = mlflow.start_run(run_id=run_id,run_name=f"{training_config.model}_{training_config.dataset_id.split('/')[-1]}" )
    current_run_id = active_run.info.run_id
    
    try:
        train(
            config=training_config, 
            checkpoint=checkpoint, 
            mlflow_run_id=current_run_id
        )
    except KeyboardInterrupt:
        print("Training interrupted. Checkpoints saved.")
    finally:
        mlflow.end_run()