import wandb
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import sys
import pandas as pd
from torch.utils.data import SubsetRandomSampler
from data.config import IMTSConfig
from models.JETS import IMTS
from data.dataset import collate_triplets, EmpiricalDatasetIMTS

class IMTSTrainer:
    """Trainer for IMTS model with mixed precision support"""
    
    def __init__(self, model: IMTS, config: IMTSConfig, use_amp: bool = True):
        self.model = model
        self.config = config
        self.use_amp = use_amp and torch.cuda.is_available()  # Only use AMP if CUDA is available
        
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=config.learning_rate, 
            weight_decay=1e-5
        )
        
        print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("Training with patch size: ", config.patch_size)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs, 
            eta_min=0.001 * config.learning_rate
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")
        
        # Initialize GradScaler for mixed precision
        if self.use_amp:
            self.scaler = GradScaler()
            print("Mixed precision training enabled")
        else:
            self.scaler = None
            print("Mixed precision training disabled")
            
        self.model.to(self.device)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch with mixed precision support"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            triplets = batch[0].to(self.device, non_blocking=True)
            mask = batch[1].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # Forward pass with autocast
                with autocast():
                    outputs = self.model(triplets, mask)
                    loss = outputs['loss']
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print("NaN loss detected! Saving batch and exiting.")
                    torch.save(batch, "nan_batch.pt")
                    print("Batch saved to nan_batch.pt")
                    sys.exit(1)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Standard training without mixed precision
                outputs = self.model(triplets, mask)
                loss = outputs['loss']
                
                if torch.isnan(loss):
                    print("NaN loss detected! Saving batch and exiting.")
                    torch.save(batch, "nan_batch.pt")
                    print("Batch saved to nan_batch.pt")
                    sys.exit(1)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Update target encoder
            self.model.update_target_encoder()
            
            # Log metrics
            wandb.log({
                "train_loss": loss.item(), 
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            total_loss += loss.item()
            num_batches += 1
        
        self.scheduler.step()
        return total_loss / num_batches
    
    def train_MAE_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch with MAE (without target encoder updates)"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            triplets = batch[0].to(self.device, non_blocking=True)
            mask = batch[1].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # Forward pass with autocast
                with autocast():
                    outputs = self.model(triplets, mask)
                    loss = outputs['loss']
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print("NaN loss detected! Saving batch and exiting.")
                    torch.save(batch, "nan_batch.pt")
                    print("Batch saved to nan_batch.pt")
                    sys.exit(1)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Standard training without mixed precision
                outputs = self.model(triplets, mask)
                loss = outputs['loss']
                
                if torch.isnan(loss):
                    print("NaN loss detected! Saving batch and exiting.")
                    torch.save(batch, "nan_batch.pt")
                    print("Batch saved to nan_batch.pt")
                    sys.exit(1)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Log metrics
            wandb.log({
                "train_loss": loss.item(), 
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            total_loss += loss.item()
            num_batches += 1
        
        self.scheduler.step()
        return total_loss / num_batches

    def validate_epoch(self, dataloader: DataLoader) -> float:
        """End-of-epoch validation with mixed precision support"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                triplets = batch[0].to(self.device, non_blocking=True)
                mask = batch[1].to(self.device, non_blocking=True)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(triplets, mask)
                        loss = outputs['loss']
                else:
                    outputs = self.model(triplets, mask)
                    loss = outputs['loss']
                
                wandb.log({"val_loss": loss.item()})
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches
    
    def get_representations(self, dataloader: DataLoader) -> torch.Tensor:
        """Extract representations for downstream tasks with mixed precision support"""
        self.model.eval()
        representations = []
        
        with torch.no_grad():
            for batch in dataloader:
                triplets = batch[0].to(self.device, non_blocking=True)
                mask = batch[1].to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(triplets, mask, return_representations=True)
                        # Pool representations (mean pooling)
                        pooled = outputs['representations'].mean(dim=1)
                else:
                    outputs = self.model(triplets, mask, return_representations=True)
                    # Pool representations (mean pooling)
                    pooled = outputs['representations'].mean(dim=1)
                
                representations.append(pooled)
        
        return torch.cat(representations, dim=0)


# Updated main training loop
if __name__ == "__main__":
    # Your existing setup code...
    args = IMTSConfig()
    model = IMTS(args)

    df = pd.read_parquet(args.data_path)
    binary_df = pd.read_parquet(args.binary_data_path)
    target_column = args.target_column

    dataset = EmpiricalDatasetIMTS(
        args,
        df=df,
        timeseries_columns=args.timeseries_columns,
        is_pretrain=(args.pretrain == 1),
        min_obs_per_user=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        load_from_cache=True
    )

    train_sampler = SubsetRandomSampler(dataset.splits["train"])
    val_sampler = SubsetRandomSampler(dataset.splits["val"])

    # Enable pin_memory and non_blocking transfers for better performance
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_triplets,  
        num_workers=4,  # Increased from 2 for better data loading
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_triplets,
        num_workers=4,  # Increased from 2 for better data loading
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
    )

    # Initialize trainer with mixed precision enabled
    trainer = IMTSTrainer(model, args, use_amp=True)
    
    wandb.init(project="wearable")
    print("Starting training with mixed precision...")
    
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(val_loader)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_best.pt'))
            print(f'New best model weights saved to {args.save_dir}/model_best.pt (val_loss={val_loss:.6f})')

    print("Training complete!")