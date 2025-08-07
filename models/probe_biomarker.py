from Mamba6 import *
from sklearn.metrics import r2_score, mean_absolute_error
import torch
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
from datetime import datetime
from torch.utils.data import SubsetRandomSampler
from models.MAE import MaskedAutoencoder
from data.dataset import collate_triplets, EmpiricalDatasetIMTS
from data.config import IMTSConfig


def convert_long_to_wide_targets(
    df, user_column="user", target_column="target_name", value_column="value"
):
    """
    Convert a long-format DataFrame to wide format where each target becomes a column.
    Missing targets for users are filled with NaN.

    Args:
        df (pd.DataFrame): Input DataFrame in long format
        user_column (str): Name of the column containing user identifiers
        target_column (str): Name of the column containing target names
        value_column (str): Name of the column containing target values

    Returns:
        pd.DataFrame: Wide-format DataFrame with users as rows and targets as columns
    """

    # Validate input columns
    required_columns = [user_column, target_column, value_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Converting long format to wide format...")
    print(
        f"Input: {len(df)} rows, {df[user_column].nunique()} unique users, {df[target_column].nunique()} unique targets"
    )

    # Get unique targets and users for reporting
    unique_targets = sorted(df[target_column].unique())
    unique_users = sorted(df[user_column].unique())

    print(f"Unique targets: {unique_targets}")
    print(
        f"Sample of users: {unique_users[:5]}"
        + ("..." if len(unique_users) > 5 else "")
    )

    # Check for duplicate user-target combinations
    duplicates = df.groupby([user_column, target_column]).size()
    duplicate_pairs = duplicates[duplicates > 1]

    if len(duplicate_pairs) > 0:
        print(
            f"⚠️  Warning: Found {len(duplicate_pairs)} duplicate user-target combinations."
        )
        print("Sample duplicates:")
        for (user, target), count in duplicate_pairs.head().items():
            print(f"  User {user}, Target {target}: {count} values")
        print("Using the first occurrence of each duplicate.")

        # Keep only the first occurrence of each user-target combination
        df = df.drop_duplicates(subset=[user_column, target_column], keep="first")

    # Pivot the DataFrame
    wide_df = df.pivot(index=user_column, columns=target_column, values=value_column)

    # Reset index to make user_column a regular column again
    wide_df = wide_df.reset_index()

    # Flatten column names (remove the name from the columns index)
    wide_df.columns.name = None

    # Report statistics
    print(f"\nConversion complete:")
    print(f"Output: {len(wide_df)} users × {len(unique_targets)} targets")

    # Calculate completeness statistics
    print(f"\nTarget completeness:")
    for target in unique_targets:
        if target in wide_df.columns:
            valid_count = wide_df[target].notna().sum()
            total_count = len(wide_df)
            completeness = (valid_count / total_count) * 100
            print(f"  {target}: {valid_count}/{total_count} ({completeness:.1f}%)")

    # Overall completeness
    target_columns = [col for col in wide_df.columns if col != user_column]
    if target_columns:
        total_values = len(wide_df) * len(target_columns)
        valid_values = wide_df[target_columns].notna().sum().sum()
        overall_completeness = (valid_values / total_values) * 100
        print(
            f"\nOverall completeness: {valid_values}/{total_values} ({overall_completeness:.1f}%)"
        )

    return wide_df


def precompute_and_pool_embeddings_multi_target(model, dataloader, device, num_targets):
    """
    Runs the foundation model once to extract embeddings and performs corrected mean pooling.
    Handles multiple targets simultaneously.

    Args:
        model: The frozen foundation model.
        dataloader: The dataloader with triplets, masks, and multi-target labels.
        device: The device to run computation on ('cuda' or 'cpu').
        num_targets: Number of target variables.

    Returns:
        A new DataLoader containing (pooled_embedding, multi_target_labels) pairs.
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

            # Get embeddings from the foundation model
            embeddings = model(triplets, padding_mask, return_representations=True)[
                "representations"
            ]

            masked_embeddings = embeddings * padding_mask.unsqueeze(-1)
            summed_embeddings = torch.sum(masked_embeddings, dim=1)

            # Count non-padded tokens for each sequence in the batch
            num_non_padded = padding_mask.sum(dim=1).unsqueeze(-1)

            # Avoid division by zero for any sequences that are fully padded
            num_non_padded = torch.clamp(num_non_padded, min=1e-9)

            pooled_embeddings = summed_embeddings / num_non_padded

            all_pooled_embeddings.append(pooled_embeddings.cpu())
            all_labels.append(labels.cpu())

    # Create a new, efficient dataset and dataloader
    all_pooled_embeddings = torch.cat(all_pooled_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    embedding_dataset = TensorDataset(all_pooled_embeddings, all_labels)
    # Use a larger batch size for the probe training as it's less memory intensive
    embedding_loader = DataLoader(embedding_dataset, batch_size=256, shuffle=True)

    return embedding_loader


def evaluate_model_multi_target(
    model, train_dataloader, val_dataloader, device, target_names
):
    """
    Evaluates the foundation model using linear probes for multiple continuous targets simultaneously.
    """
    model.eval()
    num_targets = len(target_names)

    # Step 1: Pre-compute embeddings. This is the only time the large model is used.
    print("Pre-computing training embeddings...")
    train_embedding_loader = precompute_and_pool_embeddings_multi_target(
        model, train_dataloader, device, num_targets
    )
    torch.cuda.empty_cache()
    print("Pre-computing validation embeddings...")
    val_embedding_loader = precompute_and_pool_embeddings_multi_target(
        model, val_dataloader, device, num_targets
    )

    # Step 2: Setup and train probes for all targets
    probes = torch.nn.ModuleList(
        [torch.nn.Linear(model.config.embed_dim, 1) for _ in range(num_targets)]
    ).to(device)

    optimizers = [
        torch.optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-2)
        for probe in probes
    ]
    criterion = torch.nn.L1Loss()  # Mean Absolute Error loss

    print(f"Training {num_targets} probes on pre-computed embeddings...")

    num_epochs = 50
    for epoch in range(num_epochs):
        total_losses = [0.0] * num_targets
        train_pbar = tqdm(
            train_embedding_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )

        for pooled_embeddings, labels in train_pbar:
            pooled_embeddings = pooled_embeddings.to(device)
            labels = labels.to(device)

            # Train each probe
            epoch_losses = []
            for target_idx in range(num_targets):
                target_labels = labels[:, target_idx] if labels.dim() > 1 else labels

                # Skip if all labels are NaN for this target
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
            train_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            avg_losses = [
                total_losses[i] / len(train_embedding_loader)
                for i in range(num_targets)
            ]
            print(
                f"Epoch {epoch+1}/{num_epochs} completed - Average Losses: {np.mean(avg_losses):.4f}"
            )

    # Step 3: Evaluate all probes
    print("Evaluating probes...")
    for probe in probes:
        probe.eval()

    all_preds = [[] for _ in range(num_targets)]
    all_labels = [[] for _ in range(num_targets)]

    val_pbar = tqdm(val_embedding_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for pooled_embeddings, labels in val_pbar:
            pooled_embeddings = pooled_embeddings.to(device)
            labels = labels.cpu()

            for target_idx in range(num_targets):
                preds = probes[target_idx](pooled_embeddings).cpu()
                target_labels = labels[:, target_idx] if labels.dim() > 1 else labels

                all_preds[target_idx].append(preds)
                all_labels[target_idx].append(target_labels)

    # Calculate metrics for each target
    results = {}
    for target_idx, target_name in enumerate(target_names):
        preds = torch.cat(all_preds[target_idx], dim=0).numpy()
        labels = torch.cat(all_labels[target_idx], dim=0).numpy()

        # Remove NaN values
        valid_mask = ~np.isnan(labels)
        if valid_mask.sum() == 0:
            results[target_name] = {"r2": None, "mae": None}
            continue

        valid_preds = preds[valid_mask]
        valid_labels = labels[valid_mask]

        # Check if we have variation in labels
        if np.var(valid_labels) == 0:
            results[target_name] = {"r2": None, "mae": None}
            continue

        try:
            r2 = r2_score(valid_labels, valid_preds)
            mae = mean_absolute_error(valid_labels, valid_preds)
            results[target_name] = {"r2": r2, "mae": mae}
        except Exception as e:
            print(f"Error calculating metrics for {target_name}: {e}")
            results[target_name] = {"r2": None, "mae": None}

    return results


def precompute_mean_pooled_features_multi_target(dataloader, device, num_targets):
    """
    Creates baseline features by performing masked mean-pooling directly on input variables.
    Handles multiple targets.
    """
    all_pooled_features = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Extracting Mean-Pooled Features", leave=False)
        for batch in pbar:
            features, padding_mask, labels = batch
            features = features.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            # --- Masked Mean Pooling on Input Features ---
            # For triplets, we use the value (3rd column) as features
            if features.dim() == 3 and features.shape[-1] == 3:
                # Extract values from triplets and create feature matrix
                values = features[:, :, 2]  # Take the value column
                var_ids = features[:, :, 1].long()  # Variable IDs

                # Create one-hot encoding for variables
                batch_size, seq_len = values.shape
                max_var_id = 63

                # Create feature matrix: [batch, seq_len, num_variables]
                feature_matrix = torch.zeros(
                    batch_size, seq_len, max_var_id, device=device
                )

                # Fill in values for each variable
                batch_indices = (
                    torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len).to(device)
                )
                seq_indices = (
                    torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
                )

                # Only set values where padding mask is true
                valid_mask = padding_mask.bool()
                feature_matrix[
                    batch_indices[valid_mask],
                    seq_indices[valid_mask],
                    var_ids[valid_mask],
                ] = values[valid_mask]

                # Apply mean pooling
                masked_features = feature_matrix * padding_mask.unsqueeze(-1)
                summed_features = torch.sum(masked_features, dim=1)

                num_non_padded = padding_mask.sum(dim=1).unsqueeze(-1)
                num_non_padded = torch.clamp(num_non_padded, min=1e-9)

                pooled_features = summed_features / num_non_padded
            else:
                # Standard feature matrix
                masked_features = features * padding_mask.unsqueeze(-1)
                summed_features = torch.sum(masked_features, dim=1)

                num_non_padded = padding_mask.sum(dim=1).unsqueeze(-1)
                num_non_padded = torch.clamp(num_non_padded, min=1e-9)

                pooled_features = summed_features / num_non_padded

            all_pooled_features.append(pooled_features.cpu())
            all_labels.append(labels.cpu())

    all_pooled_features = torch.cat(all_pooled_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    feature_dataset = TensorDataset(all_pooled_features, all_labels)
    feature_loader = DataLoader(feature_dataset, batch_size=256, shuffle=True)

    return feature_loader, all_pooled_features.shape[1]


def evaluate_mean_pooling_baseline_multi_target(
    train_dataloader, val_dataloader, device, target_names
):
    """
    Evaluates baseline using mean-pooled features for multiple continuous targets simultaneously.
    """
    num_targets = len(target_names)

    # Step 1: Pre-compute the simple mean-pooled features.
    print("Pre-computing baseline training features...")
    train_feature_loader, num_features = precompute_mean_pooled_features_multi_target(
        train_dataloader, device, num_targets
    )

    print("Pre-computing baseline validation features...")
    val_feature_loader, _ = precompute_mean_pooled_features_multi_target(
        val_dataloader, device, num_targets
    )

    # Step 2: Setup and train probes for all targets
    probes = torch.nn.ModuleList(
        [torch.nn.Linear(num_features, 1) for _ in range(num_targets)]
    ).to(device)

    optimizers = [
        torch.optim.Adam(probe.parameters(), lr=0.001, weight_decay=1e-2)
        for probe in probes
    ]
    criterion = torch.nn.L1Loss()  # Mean Absolute Error loss

    print(
        f"Training {num_targets} baseline probes on {num_features}-dimensional mean-pooled features..."
    )

    num_epochs = 50
    for epoch in range(num_epochs):
        total_losses = [0.0] * num_targets
        train_pbar = tqdm(
            train_feature_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )

        for features, labels in train_pbar:
            features = features.to(device)
            labels = labels.to(device)

            # Train each probe
            epoch_losses = []
            for target_idx in range(num_targets):
                target_labels = labels[:, target_idx] if labels.dim() > 1 else labels

                # Skip if all labels are NaN for this target
                valid_mask = ~torch.isnan(target_labels)
                if not valid_mask.any():
                    epoch_losses.append(0.0)
                    continue

                valid_features = features[valid_mask]
                valid_labels = target_labels[valid_mask]

                preds = probes[target_idx](valid_features).squeeze()
                loss = criterion(preds, valid_labels.float())

                optimizers[target_idx].zero_grad()
                loss.backward()
                optimizers[target_idx].step()

                total_losses[target_idx] += loss.item()
                epoch_losses.append(loss.item())

            avg_loss = np.mean([l for l in epoch_losses if l > 0])
            train_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

        if epoch % 25 == 0 or epoch == num_epochs - 1:
            avg_losses = [
                total_losses[i] / len(train_feature_loader) for i in range(num_targets)
            ]
            print(
                f"Epoch {epoch+1}/{num_epochs} completed - Average Losses: {np.mean(avg_losses):.4f}"
            )

    # Step 3: Evaluate all probes
    print("Evaluating baseline probes...")
    for probe in probes:
        probe.eval()

    all_preds = [[] for _ in range(num_targets)]
    all_labels = [[] for _ in range(num_targets)]

    val_pbar = tqdm(val_feature_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for features, labels in val_pbar:
            features = features.to(device)
            labels = labels.cpu()

            for target_idx in range(num_targets):
                preds = probes[target_idx](features).cpu()
                target_labels = labels[:, target_idx] if labels.dim() > 1 else labels

                all_preds[target_idx].append(preds)
                all_labels[target_idx].append(target_labels)

    # Calculate metrics for each target
    results = {}
    for target_idx, target_name in enumerate(target_names):
        preds = torch.cat(all_preds[target_idx], dim=0).numpy()
        labels = torch.cat(all_labels[target_idx], dim=0).numpy()

        # Remove NaN values
        valid_mask = ~np.isnan(labels)
        if valid_mask.sum() == 0:
            results[target_name] = {"r2": None, "mae": None}
            continue

        valid_preds = preds[valid_mask]
        valid_labels = labels[valid_mask]

        # Check if we have variation in labels
        if np.var(valid_labels) == 0:
            results[target_name] = {"r2": None, "mae": None}
            continue

        try:
            r2 = r2_score(valid_labels, valid_preds)
            mae = mean_absolute_error(valid_labels, valid_preds)
            results[target_name] = {"r2": r2, "mae": mae}
        except Exception as e:
            print(f"Error calculating baseline metrics for {target_name}: {e}")
            results[target_name] = {"r2": None, "mae": None}

    return results


def calculate_target_statistics(dataset, target_names):
    """
    Calculate statistics for continuous target values in training and validation sets.

    Args:
        dataset: The dataset with splits information
        target_names: List of target column names

    Returns:
        Dictionary with train and val statistics for each target
    """
    train_indices = dataset.splits["train"]
    val_indices = dataset.splits["val"]

    # Get the continuous labels for all targets from the dataset's y attribute
    all_labels = torch.tensor(dataset.y, dtype=torch.float32)

    train_stats = {}
    val_stats = {}

    for i, target_name in enumerate(target_names):
        # Get labels for this target
        if all_labels.dim() > 1:
            target_labels = all_labels[:, i]
        else:
            target_labels = all_labels

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
            train_stats[target_name] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "count": 0,
            }

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
            val_stats[target_name] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "count": 0,
            }

    return train_stats, val_stats


def create_multi_target_dataloader(args, df, continuous_df, target_columns):
    """Create dataloaders for multiple continuous targets simultaneously."""
    dataset = EmpiricalDatasetIMTS(
        args,
        df=df,
        timeseries_columns=args.timeseries_columns,
        is_pretrain=(args.pretrain == 1),
        target_columns=target_columns,  # Pass list of targets
        target_df=continuous_df,  # Use target_df instead of binary_df
        min_obs_per_user=args.min_seq_len,
        max_seq_len=args.max_seq_len,
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
    args = IMTSConfig()
    args.biomarker_data_path = "biomarker.csv"
    args.pretrain = 0
    # MAE CKPT PARAMS
    # args.load_ckpt_path = "checkpoints/MAE_model_best.pt"
    # args.batch_size = 4
    # args.num_layers = 8
    # args.predictor_layers = 4

    ckpt = torch.load(args.load_ckpt_path)
    model = IMTS(args)
    # model = MaskedAutoencoder(args)
    model.load_state_dict(ckpt)
    model.to("cuda")

    df = pd.read_parquet(args.data_path)
    continuous_df = pd.read_parquet(
        args.biomarker_data_path
    )  # Changed to continuous data

    # Define continuous target columns - these should be your continuous variables
    all_targets = [
        "sleep_score",
        "activity_level",
        "heart_rate_variability",
        "stress_level",
        "recovery_score",
    ]

    # Filter targets that actually exist in continuous_df
    available_targets = [
        target for target in all_targets if target in continuous_df.columns
    ]
    missing_targets = [
        target for target in all_targets if target not in continuous_df.columns
    ]

    if missing_targets:
        print(
            f"⚠️  Warning: The following targets are not available in continuous_df: {missing_targets}"
        )

    print(
        f"📊 Evaluating {len(available_targets)} available continuous target variables simultaneously..."
    )
    print(
        "Available targets:",
        available_targets[:5],
        "..." if len(available_targets) > 5 else "",
    )
    print("=" * 80)

    # Create output directory
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create multi-target dataloaders
        print("Creating multi-target dataloaders...")
        train_loader, val_loader, dataset = create_multi_target_dataloader(
            args, df, continuous_df, available_targets
        )

        target_info = dataset.get_target_info()
        print(
            f"Dataset info: {target_info['num_targets']} targets, {len(dataset)} samples"
        )
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Calculate target statistics
        train_stats, val_stats = calculate_target_statistics(dataset, available_targets)
        print("\nTarget Value Statistics:")
        for target in available_targets:
            train_stat = train_stats[target]
            val_stat = val_stats[target]
            print(f"  {target}:")
            print(
                f"    Train: mean={train_stat['mean']:.4f}, std={train_stat['std']:.4f}, range=[{train_stat['min']:.4f}, {train_stat['max']:.4f}]"
            )
            print(
                f"    Val:   mean={val_stat['mean']:.4f}, std={val_stat['std']:.4f}, range=[{val_stat['min']:.4f}, {val_stat['max']:.4f}]"
            )

        # Evaluate baseline for all targets
        print(f"\n📊 Evaluating baseline for all {len(available_targets)} targets...")
        baseline_results = evaluate_mean_pooling_baseline_multi_target(
            train_loader, val_loader, "cuda", available_targets
        )

        # Evaluate model for all targets
        print(f"\n🧠 Evaluating model for all {len(available_targets)} targets...")
        model_results = evaluate_model_multi_target(
            model, train_loader, val_loader, "cuda", available_targets
        )

        # Compile results
        results = []
        for target in available_targets:
            baseline = baseline_results.get(target, {"r2": None, "mae": None})
            model = model_results.get(target, {"r2": None, "mae": None})

            baseline_r2 = baseline["r2"]
            baseline_mae = baseline["mae"]
            model_r2 = model["r2"]
            model_mae = model["mae"]

            train_stat = train_stats[target]
            val_stat = val_stats[target]

            results.append(
                {
                    "target_variable": target,
                    "baseline_r2": baseline_r2,
                    "baseline_mae": baseline_mae,
                    "model_r2": model_r2,
                    "model_mae": model_mae,
                    "r2_improvement": (
                        model_r2 - baseline_r2
                        if (model_r2 is not None and baseline_r2 is not None)
                        else None
                    ),
                    "mae_improvement": (
                        baseline_mae - model_mae
                        if (model_mae is not None and baseline_mae is not None)
                        else None
                    ),  # Negative improvement is better for MAE
                    "train_mean": train_stat["mean"],
                    "train_std": train_stat["std"],
                    "val_mean": val_stat["mean"],
                    "val_std": val_stat["std"],
                }
            )

        # Save results
        print("\n" + "=" * 80)
        print("💾 Saving results...")

        results_df = pd.DataFrame(results)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = os.path.join(
            output_dir, f"multi_target_continuous_evaluation_{timestamp}.csv"
        )
        results_df.to_csv(final_filename, index=False)

        print(f"Results saved to: {final_filename}")

        # Display summary statistics
        print("\n📈 Summary Statistics:")
        print("-" * 40)

        # Filter out None values for statistics
        valid_results = results_df.dropna(subset=["baseline_r2", "model_r2"])

        if len(valid_results) > 0:
            print(
                f"Successfully evaluated: {len(valid_results)}/{len(available_targets)} variables"
            )
            print(f"Average baseline R²: {valid_results['baseline_r2'].mean():.4f}")
            print(f"Average model R²: {valid_results['model_r2'].mean():.4f}")
            print(
                f"Average R² improvement: {valid_results['r2_improvement'].mean():.4f}"
            )
            print(f"Average baseline MAE: {valid_results['baseline_mae'].mean():.4f}")
            print(f"Average model MAE: {valid_results['model_mae'].mean():.4f}")
            print(
                f"Average MAE improvement: {valid_results['mae_improvement'].mean():.4f}"
            )

            print(f"\nTop 5 variables with largest R² improvements:")
            top_r2 = valid_results.nlargest(5, "r2_improvement")[
                ["target_variable", "r2_improvement", "train_mean", "val_mean"]
            ]
            for _, row in top_r2.iterrows():
                print(
                    f"  {row['target_variable']}: +{row['r2_improvement']:.4f} (Train mean: {row['train_mean']:.2f}, Val mean: {row['val_mean']:.2f})"
                )

            print(f"\nTop 5 variables with largest MAE improvements:")
            top_mae = valid_results.nlargest(5, "mae_improvement")[
                ["target_variable", "mae_improvement", "train_std", "val_std"]
            ]
            for _, row in top_mae.iterrows():
                print(
                    f"  {row['target_variable']}: +{row['mae_improvement']:.4f} (Train std: {row['train_std']:.2f}, Val std: {row['val_std']:.2f})"
                )
        else:
            print("No valid results obtained.")

        print(
            f"\n🎉 Multi-target continuous evaluation complete! Results saved to {final_filename}"
        )

    except Exception as e:
        print(f"❌ Critical error during evaluation: {str(e)}")
        import traceback

        traceback.print_exc()

        # Save error info
        error_info = {
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "available_targets": available_targets,
        }

        error_filename = os.path.join(
            output_dir,
            f"evaluation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )
        with open(error_filename, "w") as f:
            f.write(f"Error: {error_info}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")

        print(f"Error details saved to: {error_filename}")
