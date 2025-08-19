import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from tqdm import tqdm
from MAE import MaskedAutoencoder
from JETS import *
from data.config import IMTSConfig
from data.dataset import collate_triplets, EHDatasetIMTS


class LabelNormalizer:
    """Handles normalization and denormalization of labels for multiple targets."""

    def __init__(self):
        self.means = {}
        self.stds = {}
        self.fitted = False

    def fit(self, labels, target_names):
        """
        Fit the normalizer on training labels.

        Args:
            labels: torch.Tensor of shape [N, num_targets] or [N] for single target.
            target_names: List of target names.
        """
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        self.means = {}
        self.stds = {}

        for i, target_name in enumerate(target_names):
            target_labels = labels[:, i]
            valid_mask = ~torch.isnan(target_labels)

            if valid_mask.any():
                valid_labels = target_labels[valid_mask]
                self.means[target_name] = valid_labels.mean().item()
                self.stds[target_name] = valid_labels.std().item()

                # Prevent division by zero for constant targets
                if self.stds[target_name] == 0:
                    self.stds[target_name] = 1.0
            else:
                self.means[target_name] = 0.0
                self.stds[target_name] = 1.0

        self.fitted = True
        print("Label normalization parameters:")
        for target_name in target_names:
            print(
                f"  {target_name}: mean={self.means[target_name]:.4f}, "
                f"std={self.stds[target_name]:.4f}"
            )

    def normalize(self, labels, target_names):
        """Normalize labels using fitted parameters."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before use")

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        normalized = labels.clone()
        for i, target_name in enumerate(target_names):
            target_labels = labels[:, i]
            valid_mask = ~torch.isnan(target_labels)

            if valid_mask.any():
                normalized[valid_mask, i] = (
                    target_labels[valid_mask] - self.means[target_name]
                ) / self.stds[target_name]

        return normalized.squeeze() if len(target_names) == 1 else normalized

    def denormalize(self, normalized_labels, target_names):
        """Denormalize labels back to original scale."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before use")

        if normalized_labels.dim() == 1:
            normalized_labels = normalized_labels.unsqueeze(-1)

        denormalized = normalized_labels.clone()
        for i, target_name in enumerate(target_names):
            target_labels = normalized_labels[:, i]
            denormalized[:, i] = (
                target_labels * self.stds[target_name] + self.means[target_name]
            )

        return denormalized.squeeze() if len(target_names) == 1 else denormalized



def precompute_and_pool_embeddings_multi_target_normalized(
    model, dataloader, device, num_targets, normalizer, target_names
):
    """
    Runs the foundation model once to extract embeddings and performs corrected mean pooling.
    Handles multiple targets simultaneously with label normalization.
    """
    all_pooled_embeddings = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Extracting Embeddings", leave=False)
        for batch in pbar:
            triplets, padding_mask, labels = batch
            triplets = triplets.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            normalized_labels = normalizer.normalize(labels, target_names)

            # Get embeddings from the foundation model
            embeddings = model(triplets, padding_mask, return_representations=True)[
                "representations"
            ]

            masked_embeddings = embeddings * padding_mask.unsqueeze(-1)
            summed_embeddings = torch.sum(masked_embeddings, dim=1)

            # Count non-padded tokens for each sequence in the batch
            num_non_padded = padding_mask.sum(dim=1).unsqueeze(-1)
            num_non_padded = torch.clamp(num_non_padded, min=1e-9)

            pooled_embeddings = summed_embeddings / num_non_padded
            all_pooled_embeddings.append(pooled_embeddings.cpu())
            all_labels.append(normalized_labels.cpu())

    all_pooled_embeddings = torch.cat(all_pooled_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    embedding_dataset = TensorDataset(all_pooled_embeddings, all_labels)
    embedding_loader = DataLoader(embedding_dataset, batch_size=256, shuffle=True)
    return embedding_loader


def evaluate_model_multi_target_normalized(
    model, train_dataloader, val_dataloader, device, target_names
):
    """
    Evaluates the foundation model using linear probes for multiple continuous targets.
    Uses label normalization and reports MAE on the original scale.
    """
    model.eval()
    num_targets = len(target_names)

    # Step 1: Fit normalizer on training data
    print("Fitting label normalizer on training data...")
    normalizer = LabelNormalizer()
    all_train_labels = []
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Collecting training labels", leave=False):
            _, _, labels = batch
            all_train_labels.append(labels)
    all_train_labels = torch.cat(all_train_labels, dim=0)
    normalizer.fit(all_train_labels, target_names)

    # Step 2: Pre-compute embeddings with normalized labels
    print("Pre-computing training embeddings...")
    train_embedding_loader = precompute_and_pool_embeddings_multi_target_normalized(
        model, train_dataloader, device, num_targets, normalizer, target_names
    )
    torch.cuda.empty_cache()

    print("Pre-computing validation embeddings...")
    val_embedding_loader = precompute_and_pool_embeddings_multi_target_normalized(
        model, val_dataloader, device, num_targets, normalizer, target_names
    )

    # Step 3: Setup and train probes
    probes = torch.nn.ModuleList(
        [torch.nn.Linear(model.config.embed_dim, 1) for _ in range(num_targets)]
    ).to(device)
    optimizers = [
        torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-2)
        for probe in probes
    ]
    criterion = torch.nn.L1Loss()  # MAE Loss

    print(f"Training {num_targets} probes on pre-computed embeddings...")
    num_epochs = 50
    for epoch in range(num_epochs):
        total_losses = [0.0] * num_targets
        train_pbar = tqdm(
            train_embedding_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )

        for pooled_embeddings, normalized_labels in train_pbar:
            pooled_embeddings = pooled_embeddings.to(device)
            normalized_labels = normalized_labels.to(device)

            epoch_losses = []
            for target_idx in range(num_targets):
                target_labels = (
                    normalized_labels[:, target_idx]
                    if normalized_labels.dim() > 1
                    else normalized_labels
                )
                valid_mask = ~torch.isnan(target_labels)
                if not valid_mask.any():
                    epoch_losses.append(0.0)
                    continue

                valid_embeddings = pooled_embeddings[valid_mask]
                valid_labels = target_labels[valid_mask]
                preds = probes[target_idx](valid_embeddings).squeeze()
                loss = criterion(preds, valid_labels.float())

                optimizers[target_idx].zero_grad()
                loss.backward()
                optimizers[target_idx].step()

                total_losses[target_idx] += loss.item()
                epoch_losses.append(loss.item())

            avg_loss = np.mean([l for l in epoch_losses if l > 0])
            train_pbar.set_postfix({"avg_mae_loss": f"{avg_loss:.4f}"})

        if epoch % 25 == 0 or epoch == num_epochs - 1:
            avg_losses = [
                total_losses[i] / len(train_embedding_loader) for i in range(num_targets)
            ]
            print(
                f"Epoch {epoch+1}/{num_epochs} completed - "
                f"Average MAE Losses: {np.mean(avg_losses):.4f}"
            )

    # Step 4: Evaluate all probes
    print("Evaluating probes...")
    for probe in probes:
        probe.eval()

    all_preds = [[] for _ in range(num_targets)]
    all_true_labels = [[] for _ in range(num_targets)]

    val_pbar = tqdm(val_embedding_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for pooled_embeddings, normalized_labels in val_pbar:
            pooled_embeddings = pooled_embeddings.to(device)

            for target_idx in range(num_targets):
                # Get normalized predictions and denormalize them
                normalized_preds = probes[target_idx](pooled_embeddings).cpu()
                if len(target_names) == 1:
                    denorm_input = normalized_preds
                else:
                    denorm_input = torch.zeros(len(normalized_preds), len(target_names))
                    denorm_input[:, target_idx] = normalized_preds.squeeze()
                
                denormalized_preds = normalizer.denormalize(denorm_input, target_names)
                if len(target_names) > 1:
                    denormalized_preds = denormalized_preds[:, target_idx]
                all_preds[target_idx].append(denormalized_preds)

                # Denormalize true labels for evaluation
                normalized_target_labels = (
                    normalized_labels[:, target_idx]
                    if normalized_labels.dim() > 1
                    else normalized_labels
                )
                if len(target_names) == 1:
                    denorm_labels_input = normalized_target_labels
                else:
                    denorm_labels_input = torch.zeros(len(normalized_target_labels), len(target_names))
                    denorm_labels_input[:, target_idx] = normalized_target_labels
                
                denormalized_true_labels = normalizer.denormalize(denorm_labels_input, target_names)
                if len(target_names) > 1:
                    denormalized_true_labels = denormalized_true_labels[:, target_idx]
                all_true_labels[target_idx].append(denormalized_true_labels)


    # Calculate metrics for each target
    results = {}
    for target_idx, target_name in enumerate(target_names):
        preds = torch.cat(all_preds[target_idx], dim=0).numpy()
        labels = torch.cat(all_true_labels[target_idx], dim=0).numpy()

        valid_mask = ~np.isnan(labels)
        if valid_mask.sum() == 0 or np.var(labels[valid_mask]) == 0:
            results[target_name] = {"mae": None}
            continue

        valid_preds = preds[valid_mask]
        valid_labels = labels[valid_mask]
        
        try:
            mae = mean_absolute_error(valid_labels, valid_preds)
            results[target_name] = {"mae": mae}
        except Exception as e:
            print(f"Error calculating metrics for {target_name}: {e}")
            results[target_name] = {"mae": None}

    return results


def precompute_mean_pooled_features_multi_target_normalized(
    dataloader, device, num_targets, normalizer, target_names
):
    """
    Creates baseline features by performing masked mean-pooling directly on input variables.
    Handles multiple targets with normalized labels.
    """
    all_pooled_features = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Extracting Mean-Pooled Features", leave=False)
        for batch in pbar:
            features, padding_mask, labels = batch
            features, padding_mask, labels = (
                features.to(device),
                padding_mask.to(device),
                labels.to(device),
            )

            normalized_labels = normalizer.normalize(labels, target_names)

            if features.dim() == 3 and features.shape[-1] == 3:
                # Handle triplet format: (time, var_id, value)
                values = features[:, :, 2]
                var_ids = features[:, :, 1].long()
                batch_size, seq_len = values.shape
                max_var_id = 63 # Assuming max_var_id is known

                feature_matrix = torch.zeros(
                    batch_size, seq_len, max_var_id, device=device
                )
                batch_indices = (
                    torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len).to(device)
                )
                seq_indices = (
                    torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
                )

                valid_mask = padding_mask.bool()
                feature_matrix[
                    batch_indices[valid_mask],
                    seq_indices[valid_mask],
                    var_ids[valid_mask],
                ] = values[valid_mask]
                
                masked_features = feature_matrix * padding_mask.unsqueeze(-1)
                summed_features = torch.sum(masked_features, dim=1)
                
            else:
                # Standard feature matrix
                masked_features = features * padding_mask.unsqueeze(-1)
                summed_features = torch.sum(masked_features, dim=1)

            num_non_padded = padding_mask.sum(dim=1).unsqueeze(-1)
            num_non_padded = torch.clamp(num_non_padded, min=1e-9)
            pooled_features = summed_features / num_non_padded

            all_pooled_features.append(pooled_features.cpu())
            all_labels.append(normalized_labels.cpu())

    all_pooled_features = torch.cat(all_pooled_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    feature_dataset = TensorDataset(all_pooled_features, all_labels)
    feature_loader = DataLoader(feature_dataset, batch_size=512, shuffle=True)
    return feature_loader, all_pooled_features.shape[1]


def evaluate_mean_pooling_baseline_multi_target_normalized(
    train_dataloader, val_dataloader, device, target_names
):
    """
    Evaluates a baseline using mean-pooled features for multiple continuous targets.
    """
    num_targets = len(target_names)

    # Step 1: Fit normalizer
    print("Fitting baseline label normalizer on training data...")
    normalizer = LabelNormalizer()
    all_train_labels = []
    with torch.no_grad():
        for batch in tqdm(
            train_dataloader, desc="Collecting baseline training labels", leave=False
        ):
            _, _, labels = batch
            all_train_labels.append(labels)
    all_train_labels = torch.cat(all_train_labels, dim=0)
    normalizer.fit(all_train_labels, target_names)

    # Step 2: Pre-compute mean-pooled features
    print("Pre-computing baseline training features...")
    train_feature_loader, num_features = (
        precompute_mean_pooled_features_multi_target_normalized(
            train_dataloader, device, num_targets, normalizer, target_names
        )
    )
    print("Pre-computing baseline validation features...")
    val_feature_loader, _ = precompute_mean_pooled_features_multi_target_normalized(
        val_dataloader, device, num_targets, normalizer, target_names
    )

    # Step 3: Setup and train probes
    probes = torch.nn.ModuleList(
        [torch.nn.Linear(num_features, 1) for _ in range(num_targets)]
    ).to(device)
    optimizers = [
        torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-2)
        for probe in probes
    ]
    criterion = torch.nn.L1Loss()  # MAE Loss

    print(
        f"Training {num_targets} baseline probes on "
        f"{num_features}-dimensional mean-pooled features..."
    )
    num_epochs = 100
    for epoch in range(num_epochs):
        train_pbar = tqdm(
            train_feature_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )
        for features, normalized_labels in train_pbar:
            features, normalized_labels = features.to(device), normalized_labels.to(device)
            
            # This loop structure is similar to the main evaluation function
            # and could be refactored into a helper if desired.
            for target_idx in range(num_targets):
                target_labels = (
                    normalized_labels[:, target_idx]
                    if normalized_labels.dim() > 1
                    else normalized_labels
                )
                valid_mask = ~torch.isnan(target_labels)
                if not valid_mask.any():
                    continue

                valid_features = features[valid_mask]
                valid_labels = target_labels[valid_mask]
                preds = probes[target_idx](valid_features).squeeze()
                loss = criterion(preds, valid_labels.float())

                optimizers[target_idx].zero_grad()
                loss.backward()
                optimizers[target_idx].step()

    print("Evaluating baseline probes...")
    
    for probe in probes:
        probe.eval()
    
    all_preds = [[] for _ in range(num_targets)]
    all_true_labels = [[] for _ in range(num_targets)]
    
    with torch.no_grad():
        for features, normalized_labels in val_feature_loader:
            features = features.to(device)
            for target_idx in range(num_targets):
                normalized_preds = probes[target_idx](features).cpu()
                if len(target_names) == 1:
                    denorm_input = normalized_preds
                else:
                    denorm_input = torch.zeros(len(normalized_preds), len(target_names))
                    denorm_input[:, target_idx] = normalized_preds.squeeze()

                denormalized_preds = normalizer.denormalize(denorm_input, target_names)
                if len(target_names) > 1:
                    denormalized_preds = denormalized_preds[:, target_idx]
                all_preds[target_idx].append(denormalized_preds)

                normalized_target_labels = (
                    normalized_labels[:, target_idx]
                    if normalized_labels.dim() > 1
                    else normalized_labels
                )
                if len(target_names) == 1:
                    denorm_labels_input = normalized_target_labels
                else:
                    denorm_labels_input = torch.zeros(len(normalized_target_labels), len(target_names))
                    denorm_labels_input[:, target_idx] = normalized_target_labels
                
                denormalized_true_labels = normalizer.denormalize(denorm_labels_input, target_names)
                if len(target_names) > 1:
                    denormalized_true_labels = denormalized_true_labels[:, target_idx]
                all_true_labels[target_idx].append(denormalized_true_labels)


    # Calculate final metrics
    results = {}
    for target_idx, target_name in enumerate(target_names):
        preds = torch.cat(all_preds[target_idx], dim=0).numpy()
        labels = torch.cat(all_true_labels[target_idx], dim=0).numpy()
        valid_mask = ~np.isnan(labels)
        if valid_mask.sum() > 0 and np.var(labels[valid_mask]) > 0:
            mae = mean_absolute_error(labels[valid_mask], preds[valid_mask])
            results[target_name] = {"mae": mae}
        else:
            results[target_name] = {"mae": None}
    
    return results


def calculate_target_statistics(dataset, target_names):
    """Calculate statistics for continuous target values in training and validation sets."""
    train_indices = dataset.splits["train"]
    val_indices = dataset.splits["val"]
    all_labels = torch.tensor(dataset.y, dtype=torch.float32)

    train_stats, val_stats = {}, {}

    for i, target_name in enumerate(target_names):
        target_labels = all_labels[:, i] if all_labels.dim() > 1 else all_labels

        # Calculate for training set
        train_labels = target_labels[train_indices]
        train_valid_mask = ~torch.isnan(train_labels)
        if train_valid_mask.any():
            valid_train_labels = train_labels[train_valid_mask]
            train_stats[target_name] = {
                "mean": valid_train_labels.mean().item(),
                "std": valid_train_labels.std().item(),
                "min": valid_train_labels.min().item(),
                "max": valid_train_labels.max().item(),
                "count": train_valid_mask.sum().item(),
            }
        else:
            train_stats[target_name] = {"mean": None, "std": None, "min": None, "max": None, "count": 0}

        # Calculate for validation set
        val_labels = target_labels[val_indices]
        val_valid_mask = ~torch.isnan(val_labels)
        if val_valid_mask.any():
            valid_val_labels = val_labels[val_valid_mask]
            val_stats[target_name] = {
                "mean": valid_val_labels.mean().item(),
                "std": valid_val_labels.std().item(),
                "min": valid_val_labels.min().item(),
                "max": valid_val_labels.max().item(),
                "count": val_valid_mask.sum().item(),
            }
        else:
            val_stats[target_name] = {"mean": None, "std": None, "min": None, "max": None, "count": 0}

    return train_stats, val_stats


def create_multi_target_dataloader(args, df, biomarker_df, target_columns):
    """Create dataloaders for multiple continuous targets simultaneously."""
    dataset = EHDatasetIMTS(
        args,
        df=df,
        timeseries_columns=args.timeseries_columns,
        is_pretrain=(args.pretrain == 1),
        target_columns=target_columns,
        binary_df=biomarker_df,
        min_obs_per_user=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        # outlier_method="zscore",
        load_from_cache=False,
    )

    train_sampler = SubsetRandomSampler(dataset.splits["train"])
    val_sampler = SubsetRandomSampler(dataset.splits["val"])

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_triplets,
        num_workers=2,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_triplets,
        num_workers=2,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, dataset


if __name__ == "__main__":

    args = IMTSConfig() # this should have the data and checkpoint paths and not be in pretraining mode 
    biomarker_df = pd.read_parquet(args.biomarker_data_path)
    ckpt = torch.load(args.load_ckpt_path)
    model = IMTS(args)
    model.load_state_dict(ckpt)
    model.to("cuda")

    df = pd.read_csv(args.data_path)
    all_targets = args.target_columns

    available_targets = [t for t in all_targets if t in biomarker_df.columns]

    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    try:
        print("Creating multi-target dataloaders...")
        train_loader, val_loader, dataset = create_multi_target_dataloader(
            args, df, biomarker_df, available_targets
        )

        print(
            f"Dataset info: {dataset.get_target_info()['num_targets']} targets, "
            f"{len(dataset)} samples"
        )
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        train_stats, val_stats = calculate_target_statistics(dataset, available_targets)
        print("\nTarget Value Statistics (Train Set):")
        
        # Evaluate baseline
        print(f"\nEvaluating baseline for {len(available_targets)} targets...")
        baseline_results = evaluate_mean_pooling_baseline_multi_target_normalized(
            train_loader, val_loader, "cuda", available_targets
        )

        # Evaluate model
        print(f"\nEvaluating model for {len(available_targets)} targets...")
        model_results = evaluate_model_multi_target_normalized(
            model, train_loader, val_loader, "cuda", available_targets
        )

        # Compile and save results
        results = []
        for target in available_targets:
            baseline_mae = baseline_results.get(target, {}).get('mae')
            model_mae = model_results.get(target, {}).get('mae')
            
            improvement = None
            if model_mae is not None and baseline_mae is not None:
                improvement = baseline_mae - model_mae

            results.append({
                "target_variable": target,
                "baseline_mae": baseline_mae,
                "model_mae": model_mae,
                "mae_improvement": improvement,
                "train_mean": train_stats[target]["mean"],
                "train_std": train_stats[target]["std"],
            })

        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = os.path.join(
            output_dir, f"multi_target_continuous_evaluation_mae_{timestamp}.csv"
        )
        results_df.to_csv(final_filename, index=False)
        print("\n" + "=" * 80)
        print(f"Results saved to: {final_filename}")

        # Display summary
        print("\nSummary Statistics:")
        print("-" * 50)
        valid_results = results_df.dropna(subset=["baseline_mae", "model_mae"])
        if not valid_results.empty:
            print(f"Evaluated: {len(valid_results)}/{len(available_targets)} variables")
            print(f"Average baseline MAE: {valid_results['baseline_mae'].mean():.4f}")
            print(f"Average model MAE:    {valid_results['model_mae'].mean():.4f}")
            print(f"Average MAE improvement: {valid_results['mae_improvement'].mean():.4f}")
            
            print("\nTop 5 improvements:")
            top_5 = valid_results.nlargest(5, "mae_improvement")
            for _, row in top_5.iterrows():
                print(
                    f"  {row['target_variable']}: +{row['mae_improvement']:.4f} "
                    f"(model MAE: {row['model_mae']:.4f})"
                )
        else:
            print("No valid results were obtained.")
        
        print(f"\n🎉 Evaluation complete!")

    except Exception as e:
        import traceback
        print(f"❌ Critical error during evaluation: {str(e)}")
        traceback.print_exc()

        # Save error info
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        error_filename = os.path.join(
            output_dir, f"evaluation_error_{error_info['timestamp']}.txt"
        )
        with open(error_filename, "w") as f:
            f.write(str(error_info))
        print(f"Error details saved to: {error_filename}")