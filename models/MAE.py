import torch
import torch.nn as nn
from typing import Tuple, Dict
from JETS import *
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler
from data.config import IMTSConfig
from data.dataset import collate_triplets, EmpiricalDatasetIMTS
import wandb
from trainer import IMTSTrainer


class MAEDecoder(nn.Module):
    """
    Decoder for Masked Autoencoder. Takes encoded context and mask tokens, predicts masked values.
    """

    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=4,  # You may want to make this configurable
            dim_feedforward=config.embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.predictor_layers
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        x = self.encoder(x)
        x = self.norm(x)
        return self.output_proj(x)


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) for self-supervised learning on time series.
    """

    def __init__(self, config: IMTSConfig):
        super().__init__()
        self.config = config
        self.triplet_embedding = TripletEmbedding(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = MAEDecoder(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

    def _create_patch_indices(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        patch_size = getattr(self.config, "patch_size", 1)
        num_patches = seq_len // patch_size

        # Generate random permutation of patch indices for the whole batch
        patch_indices = torch.argsort(
            torch.rand(batch_size, num_patches, device=device), dim=-1
        )

        # Determine how many patches to mask
        num_masked_patches = int(self.config.mask_ratio * num_patches)
        num_context_patches = num_patches - num_masked_patches

        context_patch_indices = torch.sort(
            patch_indices[:, num_masked_patches:], dim=-1
        ).values
        target_patch_indices = torch.sort(
            patch_indices[:, :num_masked_patches], dim=-1
        ).values

        patch_range = torch.arange(patch_size, device=device)

        context_indices = context_patch_indices.unsqueeze(-1) * patch_size + patch_range
        target_indices = target_patch_indices.unsqueeze(-1) * patch_size + patch_range

        context_indices = context_indices.flatten(start_dim=1)
        target_indices = target_indices.flatten(start_dim=1)

        remaining_tokens = seq_len % patch_size
        if remaining_tokens > 0:
            remainder_start = num_patches * patch_size
            remainder_indices = torch.arange(
                remainder_start, seq_len, device=device
            ).expand(batch_size, -1)
            context_indices = torch.cat([context_indices, remainder_indices], dim=1)
        num_masked = num_masked_patches * patch_size

        return context_indices, target_indices, num_masked

    def forward(
        self,
        triplets: torch.Tensor,
        padding_mask: torch.Tensor,
        return_representations: bool = False,
    ) -> Dict:
        # triplets: (batch, seq_len, 3), padding_mask: (batch, seq_len)
        batch_size, seq_len, _ = triplets.shape
        device = triplets.device
        embeddings = self.triplet_embedding(triplets)
        if return_representations:
            return {"representations": self.encoder(embeddings)}
        # Split into context and masked
        context_indices, target_indices, num_masked = self._create_patch_indices(
            batch_size, seq_len, device
        )
        context_emb = torch.gather(
            embeddings,
            1,
            context_indices.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1]),
        )
        context_repr = self.encoder(context_emb)
        # Prepare decoder input: context + mask tokens at masked positions
        target_pos_emb = torch.gather(
            embeddings,
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1]),
        )
        decoder_input = torch.cat(
            [context_repr, self.mask_token + target_pos_emb], dim=1
        )
        decoded_repr = self.decoder(decoder_input)
        predicted_values = decoded_repr[:, -num_masked:]  # (batch, num_masked, 1)
        target_values = torch.gather(triplets[:, :, 2], 1, target_indices).unsqueeze(-1)
        target_mask = torch.gather(padding_mask, 1, target_indices).unsqueeze(-1)
        # MSE loss on non-padded masked tokens
        loss = (predicted_values - target_values) ** 2
        loss = (loss * target_mask).sum() / (target_mask.sum() + 1e-8)
        return {"loss": loss}


class GRU(nn.Module):
    """
    A simple supervised baseline using a GRU for variable-length time series classification.
    """

    def __init__(self, config: IMTSConfig, gru_hidden_dim: int, gru_layers: int = 2):
        super().__init__()

        self.triplet_embedding = TripletEmbedding(config)

        self.gru = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,  # Keep it simple for a baseline
        )

        self.classifier = nn.Linear(gru_hidden_dim, 1)

    def forward(self, triplets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            triplets (torch.Tensor): Padded tensor of shape (batch_size, max_seq_len, 3).
            lengths (torch.Tensor): Tensor of original sequence lengths, shape (batch_size,).

        Returns:
            torch.Tensor: Logits for binary classification, shape (batch_size, 1).
        """
        # 1. Get embeddings
        # -> (batch_size, max_seq_len, embed_dim)
        embeddings = self.triplet_embedding(triplets)

        # 2. Pack the padded sequence
        # This tells the GRU to ignore the padded parts of the sequences.
        # We enforce cpu for pack_padded_sequence lengths arg as it's a requirement.
        packed_embeddings = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, last_hidden = self.gru(packed_embeddings)

        final_representation = last_hidden[-1]
        logits = self.classifier(final_representation)

        return logits


if __name__ == "__main__":

    # In[4]:
    args = IMTSConfig()
    model = MaskedAutoencoder(args)

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
        load_from_cache=False,
        outlier_method="zscore",
    )

    # In[12]:

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

    best_val_loss = float("inf")
    for epoch in range(args.num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(val_loader)

        print(
            f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))
            print(
                f"New best model weights saved to {args.save_dir}/model_best.pt (val_loss={val_loss:.6f})"
            )

    print("Training complete!")
