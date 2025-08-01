
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import List, Tuple, Optional, Dict
import math
from mamba_ssm import Mamba
import pandas as pd 
import os 
from data.config import IMTSConfig

    

class TimeEmbedding(nn.Module):
    """Learnable time embedding with sinusoidal positional encoding"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, times: torch.Tensor) -> torch.Tensor:
        # time is already normalized in EmpiricalDataset, should be a flat tensor, shape [total_time]
        normalized_times = times
        
        # Create sinusoidal embeddings
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * 
                            (-math.log(10000.0) / self.embed_dim)).to(times.device)
        
        pe = torch.zeros(times.size(0), self.embed_dim).to(times.device)
        pe[:, 0::2] = torch.sin(normalized_times.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(normalized_times.unsqueeze(1) * div_term)
        
        return self.time_proj(pe)


class TripletEmbedding(nn.Module):
    """Embedding layer for sparse triplets (time, variable, value)"""
    
    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config
        
        # Value embedding - projects continuous values
        self.value_embedding = nn.Sequential(
            nn.Linear(1, config.embed_dim),
            nn.GELU(),  # Changed from ReLU to GELU
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        
        # Variable embedding - discrete variable indices
        self.variable_embedding = nn.Embedding(config.num_variables, config.embed_dim)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(config.embed_dim)
        
        # Linear layer to project concatenated embeddings back to embed_dim
        self.projection = nn.Linear(3 * config.embed_dim, config.embed_dim)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        
    def forward(self, triplets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            triplets: tensor of shape (batch_size, seq_len, 3) where each triplet is [time, variable_id, value]
        Returns:
            embeddings: tensor of shape (batch_size, seq_len, embed_dim)
        """
        times = triplets[:, :, 0]  # (batch_size, seq_len)
        variables = triplets[:, :, 1].long()  # (batch_size, seq_len)
        values = triplets[:, :, 2].unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Get embeddings
        time_emb = self.time_embedding(times.flatten()).view(times.shape[0], times.shape[1], -1)
        var_emb = self.variable_embedding(variables)
        val_emb = self.value_embedding(values)
        
        # Concatenate embeddings instead of adding them
        concatenated = torch.cat([time_emb, var_emb, val_emb], dim=-1)  # (batch_size, seq_len, 3 * embed_dim)
        
        # Project back to embed_dim
        projected = self.projection(concatenated)  # (batch_size, seq_len, embed_dim)
        
        return self.layer_norm(projected)


class MambaEncoder(nn.Module):
    """Mamba-based encoder"""
    
    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Mamba(
                d_model=config.embed_dim,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            ) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(self.norm(x))
        return self.norm(x)


class MaskedTargetPredictor(nn.Module):
    """Predictor network that takes context and predicts masked targets"""
    
    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config
        
        # Predictor layers
        self.layers = nn.ModuleList([
            Mamba(
                d_model=config.embed_dim,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            ) for _ in range(config.predictor_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Linear layer to project concatenated embeddings (time + variable) back to embed_dim
        self.position_proj = nn.Linear(2 * config.embed_dim, config.embed_dim)
        
        # Output projection to predict target representations
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
    def forward(self, context_repr: torch.Tensor, target_time_emb: torch.Tensor, 
                target_var_emb: torch.Tensor, target_count: int) -> torch.Tensor:
        """
        Args:
            context_repr: Context representations (batch_size, context_len, embed_dim)
            target_time_emb: Time embeddings for target positions (batch_size, target_len, embed_dim)
            target_var_emb: Variable embeddings for target positions (batch_size, target_len, embed_dim)
            target_count: Number of target tokens to predict
        Returns:
            predicted_targets: (batch_size, target_len, embed_dim)
        """
        batch_size = context_repr.size(0)
        
        # Concatenate time and variable embeddings
        target_concat = torch.cat([target_time_emb, target_var_emb], dim=-1)  # (batch_size, target_len, 2 * embed_dim)
        
        # Project back to embed_dim
        target_positions = self.position_proj(target_concat)  # (batch_size, target_len, embed_dim)
        
        # Concatenate context and target queries -> [batch, seq_len, embed] 
        x = torch.cat([context_repr, target_positions], dim=1)
        
        # Apply predictor layers
        for layer in self.layers:
            x = x + layer(x)
        
        x = self.norm(x)
        
        # Extract only the target predictions (last target_count tokens)
        target_predictions = x[:, -int(target_count):]
        
        return self.output_proj(target_predictions)


class TransformerEncoder(nn.Module):
    """Transformer-based encoder"""
    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=8,  # You may want to make this configurable
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, embed_dim)
        x = self.encoder(x)
        return self.norm(x)

class TransformerMaskedTargetPredictor(nn.Module):
    """Predictor network that takes context and predicts masked targets using Transformer"""
    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=4,  # You may want to make this configurable
            dim_feedforward=config.embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.predictor_layers)
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Linear layer to project concatenated embeddings (time + variable) back to embed_dim
        self.position_proj = nn.Linear(2 * config.embed_dim, config.embed_dim)
        
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, context_repr: torch.Tensor, target_time_emb: torch.Tensor, 
                target_var_emb: torch.Tensor, target_count: int) -> torch.Tensor:
        """
        Args:
            context_repr: Context representations (batch_size, context_len, embed_dim)
            target_time_emb: Time embeddings for target positions (batch_size, target_len, embed_dim)
            target_var_emb: Variable embeddings for target positions (batch_size, target_len, embed_dim)
            target_count: Number of target tokens to predict
        Returns:
            predicted_targets: (batch_size, target_len, embed_dim)
        """
        # Concatenate time and variable embeddings
        target_concat = torch.cat([target_time_emb, target_var_emb], dim=-1)  # (batch_size, target_len, 2 * embed_dim)
        
        # Project back to embed_dim
        target_positions = self.position_proj(target_concat)  # (batch_size, target_len, embed_dim)
        
        # Concatenate context and target queries -> [batch, seq_len, embed_dim]
        x = torch.cat([context_repr, target_positions], dim=1)
        x = self.encoder(x)
        x = self.norm(x)
        target_predictions = x[:, -int(target_count):]
        return self.output_proj(target_predictions)  
    
class IMTS(nn.Module):
    """
    A Joint Embedding Predictive Architecture (I-JEPA) inspired model for 
    self-supervised learning on multivariate time series. 
    
    This implementation uses patch-based masking. The loss is computed only on
    non-padded tokens.
    """
    
    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config

        # trainable layers 
        self.triplet_embedding = TripletEmbedding(config)
        # self.context_encoder = MambaEncoder(config)
        # self.target_encoder = MambaEncoder(config)
        # self.predictor = MaskedTargetPredictor(config)
        self.context_encoder = TransformerEncoder(config)
        self.target_encoder = TransformerEncoder(config)
        self.predictor = TransformerMaskedTargetPredictor(config)
        # Initialize target encoder frozen
        self._copy_weights(self.context_encoder, self.target_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
    def _copy_weights(self, source: nn.Module, target: nn.Module):
        """Copy weights from source to target."""
        target.load_state_dict(source.state_dict())
    
    def update_target_encoder(self):
        """Update target encoder using Exponential Moving Average (EMA)."""
        with torch.no_grad():
            for ctx_param, tgt_param in zip(self.context_encoder.parameters(), 
                                          self.target_encoder.parameters()):
                tgt_param.data.mul_(self.config.ema_momentum).add_(
                    ctx_param.data, alpha=1.0 - self.config.ema_momentum
                )
    
    def _create_patch_indices(self, batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates patch-based indices for context and target tokens for the entire batch.

        Args:
            batch_size: The number of samples in the batch.
            seq_len: The length of the sequences.
            device: The torch device.

        Returns:
            A tuple containing:
            - context_indices (torch.Tensor): Indices of unmasked tokens.
            - target_indices (torch.Tensor): Indices of masked tokens.
        """
        patch_size = getattr(self.config, 'patch_size', 1)
        num_patches = seq_len // patch_size

        # Generate random permutation of patch indices for the whole batch
        patch_indices = torch.argsort(torch.rand(batch_size, num_patches, device=device), dim=-1)

        # Determine how many patches to mask
        num_masked_patches = int(self.config.mask_ratio * num_patches)
        num_context_patches = num_patches - num_masked_patches

        context_patch_indices = torch.sort(patch_indices[:, num_masked_patches:], dim=-1).values
        target_patch_indices = torch.sort(patch_indices[:, :num_masked_patches], dim=-1).values

    
        patch_range = torch.arange(patch_size, device=device)

        
        context_indices = context_patch_indices.unsqueeze(-1) * patch_size + patch_range
        target_indices = target_patch_indices.unsqueeze(-1) * patch_size + patch_range

        context_indices = context_indices.flatten(start_dim=1)
        target_indices = target_indices.flatten(start_dim=1)

        remaining_tokens = seq_len % patch_size
        if remaining_tokens > 0:
            remainder_start = num_patches * patch_size
            remainder_indices = torch.arange(remainder_start, seq_len, device=device).expand(batch_size, -1)
            context_indices = torch.cat([context_indices, remainder_indices], dim=1)
        num_masked = num_masked_patches * patch_size

        return context_indices, target_indices, num_masked
    
    def forward(self, triplets: torch.Tensor, padding_mask: torch.Tensor, return_representations: bool = False) -> Dict:
        """
        Args:
            triplets: (batch_size, seq_len, 3) - [time, variable_id, value]
            padding_mask: (batch_size, seq_len) - True for real tokens, False for padding.
            return_representations: If True, return full sequence representations for downstream tasks.
        """
        batch_size, seq_len, _ = triplets.shape
        device = triplets.device
        
        embeddings = self.triplet_embedding(triplets)
        if return_representations:
            # For downstream tasks, use the context encoder on the full sequence
            return {'representations': self.context_encoder(embeddings)}
        
        context_indices, target_indices, num_masked = self._create_patch_indices(batch_size, seq_len, device)
        
        # Handle potential padding in indices (where -1 indicates padding)
        context_valid_mask = context_indices != -1
        target_valid_mask = target_indices != -1
        
        # Clamp indices to valid range to avoid indexing errors
        context_indices_clamped = torch.clamp(context_indices, 0, seq_len - 1)
        target_indices_clamped = torch.clamp(target_indices, 0, seq_len - 1)
        
        # Gather embeddings
        context_indices_emb = context_indices_clamped.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
        context_emb = torch.gather(embeddings, 1, context_indices_emb)
        
        # Apply valid mask to context embeddings
        context_emb = context_emb * context_valid_mask.unsqueeze(-1).float()
        
        context_repr = self.context_encoder(context_emb)
        
        # Get the target representation for the full timeseries and keep the masked tokens for loss computation
        with torch.no_grad():
            full_sequence_repr = self.target_encoder(embeddings)
            target_indices_repr = target_indices_clamped.unsqueeze(-1).expand(-1, -1, full_sequence_repr.shape[-1])
            target_repr = torch.gather(full_sequence_repr, 1, target_indices_repr)

        # Get the positional information for the masked target positions
        target_times = torch.gather(triplets[:, :, 0], 1, target_indices_clamped)
        target_variables = torch.gather(triplets[:, :, 1], 1, target_indices_clamped).long()
        
        # Get embeddings for target positions
        target_time_emb = self.triplet_embedding.time_embedding(target_times.flatten()).view(target_times.shape[0], target_times.shape[1], -1)
        target_var_emb = self.triplet_embedding.variable_embedding(target_variables)
        
        # Predict the masked tokens using the context and target positions
        predicted_targets = self.predictor(
            context_repr, 
            target_time_emb,
            target_var_emb,
            num_masked
        )

        # target_mask will have shape (batch_size, num_masked_tokens)
        target_padding_mask = torch.gather(padding_mask, 1, target_indices_clamped)
        target_mask = target_padding_mask * target_valid_mask

        squared_error = (predicted_targets - target_repr.detach())**2
        masked_squared_error = squared_error * target_mask.unsqueeze(-1)
        
        # average the loss over all masked tokens (non-padded)
        sum_loss = masked_squared_error.sum()
        num_real_elements = target_mask.sum() * self.config.embed_dim
        
        # Compute the final mean loss with eps
        loss = sum_loss / (num_real_elements + 1e-8)
        
        return {'loss': loss}
    