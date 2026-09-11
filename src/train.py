import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.model import SilentWatchModel
from data_loader import SPHARDataset, get_dataloader
import yaml
import random
import numpy as np

def train():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    seed = config['training'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Dataset Parameters
    metadata_path = os.path.join(config['dataset']['processed_dir'], "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}")
        return
        
    # Dataset and Splits
    dataset = SPHARDataset(metadata_path, transform=None, seq_length=config['dataset']['seq_length'])
    if dataset.df['toa_frame'].isna().any() or not np.isfinite(dataset.df['toa_frame']).all():
        raise ValueError(
            "Training requires valid ToA labels for every sample. Add toa_frame/toa_seconds "
            "annotations via dataset.annotation_csv and re-run preprocessing."
        )
    train_size = int(config['dataset']['train_split'] * len(dataset))
    val_size = int(config['dataset']['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=split_generator
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model Initialization
    model = SilentWatchModel(
        num_classes=config['model']['num_classes'],
        temporal_module=config['model']['temporal_module'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers']
    ).to(device)
    
    # Model Compilation (PyTorch 2.0+) - Disabled due to Triton/Windows compatibility issues
    # try:
    #     print("Compiling model for faster training...")
    #     model = torch.compile(model)
    # except Exception as e:
    #     print(f"Model compilation skipped: {e}")
    
    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_toa = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    alpha = config['training']['alpha']
    beta = config['training']['beta']
    
    # AMP Scaler
    use_amp = config['training'].get('use_amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Training Loop
    best_val_toa_mae = float('inf')
    epochs_without_improvement = 0
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (frames, labels, toas) in enumerate(train_loader):
            frames, labels, toas = frames.to(device), labels.to(device), toas.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass with AMP
            with torch.amp.autocast('cuda', enabled=use_amp):
                cls_out, toa_out = model(frames)
                
                # Compute Loss
                loss_cls = criterion_cls(cls_out, labels)
                loss_toa = criterion_toa(toa_out, toas.float())
                total_loss = (alpha * loss_cls) + (beta * loss_toa)
            
            # Backward Pass with Scaler
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{config['training']['epochs']}] Batch {batch_idx+1}/{len(train_loader)} Loss: {total_loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_toa_abs = 0.0
        val_count = 0
        with torch.no_grad():
            for frames, labels, toas in val_loader:
                frames, labels, toas = frames.to(device), labels.to(device), toas.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    cls_out, toa_out = model(frames)
                    loss_cls = criterion_cls(cls_out, labels)
                    loss_toa = criterion_toa(toa_out, toas.float())
                    total_loss = (alpha * loss_cls) + (beta * loss_toa)
                val_loss += total_loss.item()
                val_toa_abs += torch.abs(toa_out - toas.float()).sum().item()
                val_count += toas.numel()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_toa_mae = val_toa_abs / max(val_count, 1)
        scheduler.step(val_toa_mae)
        print(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val ToA MAE: {val_toa_mae:.4f}")
        
        # Save model if validation loss improves
        if val_toa_mae < best_val_toa_mae:
            best_val_toa_mae = val_toa_mae
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config['training'].get(
                'checkpoint_path', 'silentwatch_best_model.pth'))
            print("Model checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['training'].get('early_stopping_patience', 10):
                print("Early stopping: validation ToA MAE did not improve.")
                break

if __name__ == "__main__":
    train()
