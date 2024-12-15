import os
import torch
import random
import numpy as np
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup, AdamW
# from torch.optim import AdamW
import mlflow
from huggingface_hub import HfApi
from tinyllama import TinyLlama, name_to_config
from .config import TrainingConfig, configure_training_args
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
    

    for file in os.listdir(config.save_dir):
      if file.startswith("checkpoint_step_") and file != f"checkpoint_step_{step}.pt":
        os.remove(os.path.join(config.save_dir, file))

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        betas=(0.9, 0.95), 
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.max_epochs // config.gradient_accumulation_steps
    start_step = 0
    start_epoch = 0
    
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps, 
        num_training_steps=total_steps,
        min_lr=config.min_lr
    )
    
    if load_weights != None and checkpoint == None:
        model.load_state_dict(load_weights)

    elif checkpoint != None:
      
        checkpoint['model_state_dict'] = {k: v.to(device) for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(checkpoint['model_state_dict'])
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        
        print(f"Resuming training from checkpoint at step {start_step}, epoch {start_epoch+1}")
        
    else:
        model.apply(partial(model._init_weights, n_layer=config_llama.n_layer))
    

    model = model.to(device)

    model.train()

    step = start_step
    epoch = start_epoch
    optimizer.zero_grad()
    losses = []
    running_loss = []
    for epoch in range(start_epoch, config.max_epochs):
        start_step = (len(train_loader) // config.gradient_accumulation_steps) * (epoch)
        for i, batch in enumerate(train_loader):
            
            if start_step +  i / config.gradient_accumulation_steps < step:
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
            running_loss.append(loss.item())
            if (i + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                print(f"Epoch {epoch+1}, step {step}, loss {loss.item()}")
                mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=step)
                running_loss = sum(running_loss) / len(running_loss)
                
                losses.append((step, running_loss))
                running_loss = []
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
    validate(model, val_loader, step,  config.pad_token_id)
    save_checkpoint(model, optimizer, scheduler, step, epoch, config, mlflow_run_id)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train or evaluate a model.")
    training_config = TrainingConfig()
    
    training_config = configure_training_args(parser, training_config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_config.device = device
    
    

    run_id = None
    checkpoint = None
    load_weights = None
    if training_config.checkpoint:
        checkpoint = torch.load(training_config.checkpoint, weights_only=False, map_location=device)
        run_id = checkpoint.get('mlflow_run_id') 
        with open("mlruns.db", "wb") as f:
            f.write(checkpoint["mlruns_db"])
            
    elif training_config.load_weights:
        checkpoint = torch.load(training_config.load_weights, weights_only=False, map_location=device)
        if "mlflow_run_id" in checkpoint:
            load_weights = checkpoint["model_state_dict"]
        else:
            load_weights = checkpoint
        checkpoint = None
    
    mlflow.set_tracking_uri(f"sqlite:///mlruns.db")
    active_run = mlflow.start_run(run_id=run_id,run_name=f"{training_config.model}_{training_config.dataset_id.split('/')[-1]}" )
    current_run_id = active_run.info.run_id
    
    try:
        train(
            config=training_config, 
            checkpoint=checkpoint, 
            mlflow_run_id=current_run_id,
            load_weights=load_weights
        )
    except KeyboardInterrupt:
        print("Training interrupted. Checkpoints saved.")
    finally:
        mlflow.end_run()