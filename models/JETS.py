
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import List, Tuple, Optional, Dict
import math
from mamba_ssm import Mamba2
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
    
class BidirectionalMambaLayer(nn.Module):
    """A single bidirectional Mamba layer."""
    
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.forward_mamba = Mamba2(d_model, d_state, d_conv, expand)
        self.backward_mamba = Mamba2(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            padding_mask (torch.Tensor): Boolean tensor where True indicates a padded position.
        
        Returns:
            torch.Tensor: Combined output from forward and backward passes.
        """
        # Forward pass
        out_fwd = self.forward_mamba(x)
        
        # Backward pass requires reversing the sequence
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.backward_mamba(x_rev)
        
        # Un-reverse the output of the backward pass to align with original order
        out_rev_unflipped = torch.flip(out_rev, dims=[1])
        
        # Manually zero out any padded positions in both outputs before combining
        # This is crucial to prevent "leakage" from padded areas.
        out_fwd = out_fwd.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        out_rev_unflipped = out_rev_unflipped.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Combine the outputs by adding them
        return out_fwd + out_rev_unflipped
    
class MambaEncoder(nn.Module):
    """Bidirectional Mamba-based encoder that correctly handles padding."""
    
    def __init__(self, config: "IMTSConfig"): # Assuming IMTSConfig is defined elsewhere
        super().__init__()
        self.config = config
        
        self.layers = nn.ModuleList([
            BidirectionalMambaLayer(
                d_model=config.embed_dim,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            ) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
                                Padded positions should already be zeroed out.
            padding_mask (torch.Tensor): Boolean tensor of shape (batch_size, seq_len)
                                       where True indicates a padded position.
        """
        for layer in self.layers:
            h = self.norm(x)
            
            h = layer(h, padding_mask)
      
            x = x + h
            
        # Apply a final layer norm before returning
        final_output = self.norm(x)
        
        return final_output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class TransformerEncoder(nn.Module):
    """Transformer-based encoder"""
    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=4,  
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, embed_dim)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(x)
    
class MaskedTargetPredictor(nn.Module): 
    """
    Predictor network that uses an MLP head. It gets context by adding the 
    mean-pooled context representation to each target query.
    """
    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config
        
        # This layer will create the query vector from the target's position
        self.target_query_proj = nn.Linear(config.embed_dim * 2, config.embed_dim)

        # A standard Transformer decoder layer to perform cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=4,
            dim_feedforward=config.embed_dim * 2,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1) # Just one layer is often enough

    def forward(self, context_repr: torch.Tensor, context_valid_mask: torch.Tensor, 
                target_time_emb: torch.Tensor, target_var_emb: torch.Tensor):
        
        # Create the queries from the target positions
        target_pos_info = torch.cat([target_time_emb, target_var_emb], dim=-1)
        target_queries = self.target_query_proj(target_pos_info)

        #  attend to the context
        #    - `tgt`: The initial queries (what we want to find).
        #    - `memory`: The context representation (where to look).
        #    - `memory_key_padding_mask`: Hides padded parts of the context.
        predicted_targets = self.decoder(
            tgt=target_queries,
            memory=context_repr,
            memory_key_padding_mask=~context_valid_mask # PyTorch expects True for padded
        )

        return predicted_targets
    
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
        self.context_encoder = MambaEncoder(config)
        self.target_encoder = MambaEncoder(config)
        # self.context_encoder = TransformerEncoder(config)
        # self.target_encoder = TransformerEncoder(config)
        self.predictor = MaskedTargetPredictor(config)
        # Initialize target encoder frozen
        self._copy_weights(self.context_encoder, self.target_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
    def _copy_weights(self, source: nn.Module, target: nn.Module):
        """Copy weights from source to target."""
        target.load_state_dict(source.state_dict())
    
    def update_target_encoder(self, momentum):
        """Update target encoder using Exponential Moving Average (EMA)."""
        with torch.no_grad():
            for ctx_param, tgt_param in zip(self.context_encoder.parameters(), 
                                          self.target_encoder.parameters()):
                tgt_param.data.mul_(momentum).add_(
                    ctx_param.data, alpha=1.0 - momentum
                )
    def _create_forecasting_mask(self, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Creates a forecasting-style mask where the last `mask_ratio` portion of
        non-padded tokens are designated as targets, and the preceding tokens are context.
        Unlike `_create_patch_indices`, this method is deterministic and masks a
        contiguous block of tokens at the end of each sequence.

        Args:
            padding_mask (torch.Tensor): Mask where True indicates a real token.

        Returns:
            A tuple containing:
            - context_indices (torch.Tensor): Padded tensor of context indices.
            - target_indices (torch.Tensor): Padded tensor of target indices.
            - num_masked (int): The max number of target tokens in a batch.
        """
        batch_size = padding_mask.shape[0]
        device = padding_mask.device
        
        all_context_indices = []
        all_target_indices = []

        for i in range(batch_size):
            # Find indices of non-padded tokens for the current sequence
            real_token_indices = padding_mask[i].nonzero(as_tuple=True)[0]
            num_real_tokens = len(real_token_indices)

            # If there are not enough tokens to create a context and target,
            # assign all to context and none to target.
            if num_real_tokens < 2:
                all_context_indices.append(real_token_indices)
                all_target_indices.append(torch.tensor([], device=device, dtype=torch.long))
                continue

            # Calculate the number of tokens to mask (target)
            num_masked_tokens = int(self.config.mask_ratio * num_real_tokens)
            
            # Ensure at least one token is kept for context
            if num_masked_tokens >= num_real_tokens:
                num_masked_tokens = num_real_tokens - 1

            # Determine the split point
            split_idx = num_real_tokens - num_masked_tokens
            
            # Split the real_token_indices into context and target
            ctx_indices_sample = real_token_indices[:split_idx]
            tgt_indices_sample = real_token_indices[split_idx:]
            
            all_context_indices.append(ctx_indices_sample)
            all_target_indices.append(tgt_indices_sample)

        # Pad index lists to create rectangular tensors for batch processing
        context_indices = torch.nn.utils.rnn.pad_sequence(
            all_context_indices, batch_first=True, padding_value=-1
        )
        target_indices = torch.nn.utils.rnn.pad_sequence(
            all_target_indices, batch_first=True, padding_value=-1
        )
        
        num_masked = target_indices.shape[1]

        return context_indices, target_indices, num_masked
    
    def _create_patch_indices(self, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Creates patches by evenly dividing the non-padded tokens for each sequence.
        This function iterates over the batch dimension because the number of real
        tokens is variable, making full vectorization impractical. This loop-and-pad
        pattern is standard and efficient for handling such ragged data.

        Args:
            padding_mask (torch.Tensor): Mask where True indicates a real token.

        Returns:
            A tuple containing:
            - context_indices (torch.Tensor): Padded tensor of context indices.
            - target_indices (torch.Tensor): Padded tensor of target indices.
            - num_masked (int): The max number of target tokens in a batch.
        """
        batch_size = padding_mask.shape[0]
        device = padding_mask.device
        
        all_context_indices = []
        all_target_indices = []

        for i in range(batch_size):
            # Find indices of non-padded tokens for the current sequence
            real_token_indices = padding_mask[i].nonzero(as_tuple=True)[0]
            num_real_tokens = len(real_token_indices)

            # If no real tokens, skip
            if num_real_tokens == 0:
                all_context_indices.append(torch.tensor([], device=device, dtype=torch.long))
                all_target_indices.append(torch.tensor([], device=device, dtype=torch.long))
                continue

            # Cannot have more patches than tokens.
            num_patches_for_sample = min(num_real_tokens, self.config.num_patches)
            
            # Create evenly spaced patches over the real tokens.
            patch_boundaries = torch.linspace(0, num_real_tokens, num_patches_for_sample + 1, device=device).long()

            # Shuffle patch IDs and select ones to mask
            shuffled_patch_ids = torch.randperm(num_patches_for_sample, device=device)
            num_masked_patches = int(self.config.mask_ratio * num_patches_for_sample)
            masked_patch_ids = shuffled_patch_ids[:num_masked_patches]
            
            # Collect original sequence 
            ctx_indices_sample, tgt_indices_sample = [], []
            for patch_id in range(num_patches_for_sample):
                start, end = patch_boundaries[patch_id], patch_boundaries[patch_id+1]
                patch_original_indices = real_token_indices[start:end]
                
                if patch_id in masked_patch_ids:
                    tgt_indices_sample.append(patch_original_indices)
                else:
                    ctx_indices_sample.append(patch_original_indices)

            all_context_indices.append(torch.cat(ctx_indices_sample) if ctx_indices_sample else torch.tensor([], device=device, dtype=torch.long))
            all_target_indices.append(torch.cat(tgt_indices_sample) if tgt_indices_sample else torch.tensor([], device=device, dtype=torch.long))

        # Pad index lists to create rectangular tensors for batch processing
        context_indices = torch.nn.utils.rnn.pad_sequence(
            all_context_indices, batch_first=True, padding_value=-1
        )
        target_indices = torch.nn.utils.rnn.pad_sequence(
            all_target_indices, batch_first=True, padding_value=-1
        )
        
        num_masked = target_indices.shape[1]

        return context_indices, target_indices, num_masked
    
    def forward(self, triplets: torch.Tensor, padding_mask: torch.Tensor, return_representations: bool = False) -> Dict:
        """
        Args:
            triplets: (batch_size, seq_len, 3) - [time, variable_id, value]
            padding_mask: (batch_size, seq_len) - True for real tokens, False for padding.
            return_representations: If True, return full sequence representations for downstream tasks.
        """
       
        batch_size, seq_len, _ = triplets.shape
        embeddings = self.triplet_embedding(triplets)
        
        if return_representations:
            # For downstream tasks, zero out padding and pass the correct attention mask
            embeddings = embeddings * padding_mask.unsqueeze(-1).float()
            return {'representations': self.context_encoder(embeddings, padding_mask=~padding_mask)}
        
        # Create context/target indices based on dynamic patching
        context_indices, target_indices, num_masked = self._create_patch_indices(padding_mask)
        
        # Create masks from padded indices (-1 indicates padding)
        context_valid_mask = (context_indices != -1)
        target_valid_mask = (target_indices != -1)
        
        # Clamp indices to valid range for gathering, zero out padded positions later
        context_indices_clamped = context_indices.masked_fill(~context_valid_mask, 0)
        target_indices_clamped = target_indices.masked_fill(~target_valid_mask, 0)
        
        # Process context encoder
        context_indices_emb = context_indices_clamped.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
        context_emb = torch.gather(embeddings, 1, context_indices_emb)
        context_emb = context_emb * context_valid_mask.unsqueeze(-1).float() # Zero out padded embeddings
        context_repr = self.context_encoder(context_emb, padding_mask=~context_valid_mask) # 1 for padded
        
        # Process target encoder
        with torch.no_grad():
            embeddings_for_target = embeddings * padding_mask.unsqueeze(-1).float()
            full_sequence_repr = self.target_encoder(embeddings_for_target, padding_mask=~padding_mask)
            
            target_indices_repr = target_indices_clamped.unsqueeze(-1).expand(-1, -1, full_sequence_repr.shape[-1])
            target_repr = torch.gather(full_sequence_repr, 1, target_indices_repr)
            target_repr = target_repr * target_valid_mask.unsqueeze(-1).float() # Zero out padded targets

        # Prepare inputs for the predictor
        target_times = torch.gather(triplets[:, :, 0], 1, target_indices_clamped)
        target_variables = torch.gather(triplets[:, :, 1], 1, target_indices_clamped).long()
        
        target_time_emb = self.triplet_embedding.time_embedding(target_times.flatten()).view(target_times.shape[0], target_times.shape[1], -1)
        target_var_emb = self.triplet_embedding.variable_embedding(target_variables)
        
        # Zero out positional embeddings for padded targets
        target_time_emb = target_time_emb * target_valid_mask.unsqueeze(-1).float()
        target_var_emb = target_var_emb * target_valid_mask.unsqueeze(-1).float()


        # Handle case with no target tokens
        if num_masked == 0:
            return {'loss': torch.tensor(0.0, device=embeddings.device, requires_grad=True)}

        predicted_targets = self.predictor(
            context_repr, 
            context_valid_mask,
            target_time_emb,
            target_var_emb,
        )

        loss_per_token = (predicted_targets - target_repr)**2
        loss_per_token = loss_per_token.mean(dim=-1) 

        # Apply mask and compute mean loss over valid (non-padded) tokens
        masked_loss = loss_per_token * target_valid_mask.float()
        sum_loss = masked_loss.sum()
        num_real_elements = target_valid_mask.sum()
        loss = sum_loss / (num_real_elements + 1e-8) 

        return {'loss': loss}
    