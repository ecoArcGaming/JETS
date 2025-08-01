from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler
from data.dataset import collate_triplets, EmpiricalDatasetIMTS
from models.JETS import IMTS
from data.config import IMTSConfig


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
            embeddings = model(triplets, padding_mask, return_representations=True)['representations']
            
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
    all_labels = torch.cat(all_labels, dim=0).clamp(max=1.0)
    
    embedding_dataset = TensorDataset(all_pooled_embeddings, all_labels)
    # Use a larger batch size for the probe training as it's less memory intensive
    embedding_loader = DataLoader(embedding_dataset, batch_size=256, shuffle=True)
    
    return embedding_loader


def evaluate_model_multi_target(model, train_dataloader, val_dataloader, device, target_names):
    """
    Evaluates the foundation model using linear probes for multiple targets simultaneously.
    """
    model.eval()
    num_targets = len(target_names)

    # Step 1: Pre-compute embeddings. This is the only time the large model is used.
    print("Pre-computing training embeddings...")
    train_embedding_loader = precompute_and_pool_embeddings_multi_target(model, train_dataloader, device, num_targets)
    
    print("Pre-computing validation embeddings...")
    val_embedding_loader = precompute_and_pool_embeddings_multi_target(model, val_dataloader, device, num_targets)
    
    # Step 2: Setup and train probes for all targets
    probes = torch.nn.ModuleList([
        torch.nn.Linear(model.config.embed_dim, 1) for _ in range(num_targets)
    ]).to(device)
    
    optimizers = [torch.optim.Adam(probe.parameters(), lr=5e-5, weight_decay=1e-2) for probe in probes]
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"Training {num_targets} probes on pre-computed embeddings...")
    
    num_epochs = 100
    for epoch in range(num_epochs):
        total_losses = [0.0] * num_targets
        train_pbar = tqdm(train_embedding_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)
        
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
            train_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
            
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            avg_losses = [total_losses[i] / len(train_embedding_loader) for i in range(num_targets)]
            print(f"Epoch {epoch+1}/{num_epochs} completed - Average Losses: {np.mean(avg_losses):.4f}")

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
        preds = torch.sigmoid(torch.cat(all_preds[target_idx], dim=0)).numpy()
        labels = torch.cat(all_labels[target_idx], dim=0).numpy()
        
        # Remove NaN values
        valid_mask = ~np.isnan(labels)
        if valid_mask.sum() == 0:
            results[target_name] = {'auroc': None, 'auprc': None}
            continue
            
        valid_preds = preds[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Check if we have both classes
        if len(np.unique(valid_labels)) < 2:
            results[target_name] = {'auroc': None, 'auprc': None}
            continue
        
        try:
            auroc = roc_auc_score(valid_labels, valid_preds)
            auprc = average_precision_score(valid_labels, valid_preds)
            results[target_name] = {'auroc': auroc, 'auprc': auprc}
        except Exception as e:
            print(f"Error calculating metrics for {target_name}: {e}")
            results[target_name] = {'auroc': None, 'auprc': None}
    
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
                feature_matrix = torch.zeros(batch_size, seq_len, max_var_id, device=device)
                
                # Fill in values for each variable
                batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len).to(device)
                seq_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
                
                # Only set values where padding mask is true
                valid_mask = padding_mask.bool()
                feature_matrix[batch_indices[valid_mask], seq_indices[valid_mask], var_ids[valid_mask]] = values[valid_mask]
                
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
    all_labels = torch.cat(all_labels, dim=0).clamp(max=1.0)
    
    feature_dataset = TensorDataset(all_pooled_features, all_labels)
    feature_loader = DataLoader(feature_dataset, batch_size=256, shuffle=True)
    
    return feature_loader, all_pooled_features.shape[1]


def evaluate_mean_pooling_baseline_multi_target(train_dataloader, val_dataloader, device, target_names):
    """
    Evaluates baseline using mean-pooled features for multiple targets simultaneously.
    """
    num_targets = len(target_names)
    
    # Step 1: Pre-compute the simple mean-pooled features.
    print("Pre-computing baseline training features...")
    train_feature_loader, num_features = precompute_mean_pooled_features_multi_target(train_dataloader, device, num_targets)
    
    print("Pre-computing baseline validation features...")
    val_feature_loader, _ = precompute_mean_pooled_features_multi_target(val_dataloader, device, num_targets)
    
    # Step 2: Setup and train probes for all targets
    probes = torch.nn.ModuleList([
        torch.nn.Linear(num_features, 1) for _ in range(num_targets)
    ]).to(device)
    
    optimizers = [torch.optim.Adam(probe.parameters(), lr=0.001, weight_decay=1e-2) for probe in probes]
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"Training {num_targets} baseline probes on {num_features}-dimensional mean-pooled features...")
    
    num_epochs = 100
    for epoch in range(num_epochs):
        total_losses = [0.0] * num_targets
        train_pbar = tqdm(train_feature_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)
        
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
            train_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
            
        if epoch % 25 == 0 or epoch == num_epochs - 1:
            avg_losses = [total_losses[i] / len(train_feature_loader) for i in range(num_targets)]
            print(f"Epoch {epoch+1}/{num_epochs} completed - Average Losses: {np.mean(avg_losses):.4f}")

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
        preds = torch.sigmoid(torch.cat(all_preds[target_idx], dim=0)).numpy()
        labels = torch.cat(all_labels[target_idx], dim=0).numpy()
        
        # Remove NaN values
        valid_mask = ~np.isnan(labels)
        if valid_mask.sum() == 0:
            results[target_name] = {'auroc': None, 'auprc': None}
            continue
            
        valid_preds = preds[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Check if we have both classes
        if len(np.unique(valid_labels)) < 2:
            results[target_name] = {'auroc': None, 'auprc': None}
            continue
        
        try:
            auroc = roc_auc_score(valid_labels, valid_preds)
            auprc = average_precision_score(valid_labels, valid_preds)
            results[target_name] = {'auroc': auroc, 'auprc': auprc}
        except Exception as e:
            print(f"Error calculating baseline metrics for {target_name}: {e}")
            results[target_name] = {'auroc': None, 'auprc': None}
    
    return results


def calculate_positive_sample_percentages(dataset, target_names):
    """
    Calculate the percentage of positive samples for each target in training and validation sets.
    
    Args:
        dataset: The dataset with splits information
        target_names: List of target column names
        
    Returns:
        Dictionary with train and val percentages for each target
    """
    train_indices = dataset.splits["train"]
    val_indices = dataset.splits["val"]
    
    # Get the binary labels for all targets from the dataset's y attribute
    all_labels = torch.tensor(dataset.y, dtype=torch.float32)
    
    train_percentages = {}
    val_percentages = {}
    
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
            train_pos_count = torch.sum(train_labels[train_valid_mask] == 1).item()
            train_total = train_valid_mask.sum().item()
            train_percentages[target_name] = (train_pos_count / train_total) * 100
        else:
            train_percentages[target_name] = 0.0
        
        # Calculate for validation set
        val_labels = target_labels[val_indices]
        val_valid_mask = ~torch.isnan(val_labels)
        if val_valid_mask.any():
            val_pos_count = torch.sum(val_labels[val_valid_mask] == 1).item()
            val_total = val_valid_mask.sum().item()
            val_percentages[target_name] = (val_pos_count / val_total) * 100
        else:
            val_percentages[target_name] = 0.0
    
    return train_percentages, val_percentages


def create_multi_target_dataloader(args, df, binary_df, target_columns):
    """Create dataloaders for multiple targets simultaneously."""
    dataset = EmpiricalDatasetIMTS(
        args,
        df=df,
        timeseries_columns=args.timeseries_columns,
        is_pretrain=(args.pretrain == 1),
        target_columns=target_columns,  # Pass list of targets
        binary_df=binary_df,
        min_obs_per_user=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        load_from_cache=False
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

def visualize_embeddings_tsne_multi_target(
    embedding_loader, 
    target_names: Optional[List[str]] = None,
    perplexity: int = 30,
    n_iter: int = 2000,
    learning_rate: float = 200.0,
    random_state: int = 42,
    standardize: bool = True,
    max_samples: Optional[int] = 5000,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Creates t-SNE visualizations of multi-target embeddings.
    
    Args:
        embedding_loader: DataLoader containing (pooled_embedding, multi_target_labels) pairs
        target_names: List of names for each target variable (for plot titles)
        perplexity: t-SNE perplexity parameter (15-50 recommended)
        n_iter: Number of t-SNE iterations
        learning_rate: t-SNE learning rate
        random_state: Random seed for reproducibility
        standardize: Whether to standardize embeddings before t-SNE
        max_samples: Maximum number of samples to use (for memory/speed)
        figsize: Figure size for the plots
        save_path: Optional path to save the figure
        
    Returns:
        embeddings_2d: The 2D t-SNE coordinates
        labels_array: The multi-target labels array
    """
    
    # Extract all embeddings and labels
    all_embeddings = []
    all_labels = []
    
    print("Extracting embeddings and labels...")
    for embeddings, labels in embedding_loader:
        all_embeddings.append(embeddings)
        all_labels.append(labels)
    
    # Concatenate all data
    embeddings_array = torch.cat(all_embeddings, dim=0).numpy()
    labels_array = torch.cat(all_labels, dim=0).numpy()
    
    num_samples, num_targets = labels_array.shape
    print(f"Total samples: {num_samples}, Number of targets: {num_targets}")
    
    # Subsample if dataset is too large
    if max_samples and num_samples > max_samples:
        indices = np.random.choice(num_samples, max_samples, replace=False)
        embeddings_array = embeddings_array[indices]
        labels_array = labels_array[indices]
        print(f"Subsampled to {max_samples} samples")
    
    # Standardize embeddings if requested
    if standardize:
        scaler = StandardScaler()
        embeddings_array = scaler.fit_transform(embeddings_array)
        print("Embeddings standardized")
    
    # Apply t-SNE
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=1
    )
    embeddings_2d = tsne.fit_transform(embeddings_array)
    
    # Create target names if not provided
    if target_names is None:
        target_names = [f"Target {i+1}" for i in range(num_targets)]
    
    # Create subplot grid
    cols = min(3, num_targets)
    rows = (num_targets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_targets == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each target
    for i in range(num_targets):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Get binary labels for this target
        target_labels = labels_array[:, i]
        
        # Create scatter plot
        scatter = ax.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            c=target_labels,
            cmap='RdYlBu_r',
            alpha=0.6,
            s=20
        )
        
        ax.set_title(f'{target_names[i]}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Label')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Negative', 'Positive'])
        
        # Print statistics
        pos_count = np.sum(target_labels == 1)
        neg_count = np.sum(target_labels == 0)
        print(f"{target_names[i]}: {pos_count} positive, {neg_count} negative samples")
    
    # Hide unused subplots
    for i in range(num_targets, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Create an overview plot showing label combinations
    create_label_combination_plot(embeddings_2d, labels_array, target_names, figsize)
    
    return embeddings_2d, labels_array


def create_label_combination_plot(embeddings_2d, labels_array, target_names, figsize):
    """
    Creates a plot showing different combinations of positive labels.
    """
    # Convert binary labels to combination strings
    num_samples, num_targets = labels_array.shape
    
    # Create combination labels
    combination_labels = []
    for i in range(num_samples):
        active_targets = [target_names[j] for j in range(num_targets) if labels_array[i, j] == 1]
        if not active_targets:
            combination_labels.append("None")
        else:
            combination_labels.append(" + ".join(active_targets))
    
    # Get unique combinations and their counts
    unique_combinations, counts = np.unique(combination_labels, return_counts=True)
    
    # Only show combinations with reasonable frequency (at least 1% of data)
    min_count = max(1, len(combination_labels) // 100)
    frequent_combinations = unique_combinations[counts >= min_count]
    
    # Create color map for combinations
    colors = plt.cm.Set3(np.linspace(0, 1, len(frequent_combinations)))
    color_map = {combo: colors[i] for i, combo in enumerate(frequent_combinations)}
    
    # Plot
    plt.figure(figsize=figsize)
    
    for combo in frequent_combinations:
        mask = np.array(combination_labels) == combo
        if np.sum(mask) > 0:
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[color_map[combo]], 
                label=f"{combo} ({np.sum(mask)})",
                alpha=0.6,
                s=20
            )
    
    plt.title('t-SNE Visualization by Label Combinations')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    print(f"\nLabel combination statistics:")
    for combo, count in zip(unique_combinations, counts):
        print(f"  {combo}: {count} samples ({count/len(combination_labels)*100:.1f}%)")
        
if __name__ == "__main__":
    args = IMTSConfig()
    args.pretrain = 0
    args.load_ckpt_path = "checkpoints/model_best.pt"
    ckpt = torch.load(args.load_ckpt_path)
    model = IMTS(args)
    model.load_state_dict(ckpt)
    model.to("cuda")

    df = pd.read_parquet(args.data_path)
    binary_df = pd.read_parquet(args.binary_data_path)

    all_targets = ['ADHD or ADD', 
                   "Alzheimer's", 
                   'Anxiety', 
                   'Arthritis', 
                   'Asthma', 
                   'Atrial fibrillation', 
                   'Atrial flutter', 
                   'Autism spectrum disorders', 
                   'Back pain', 'Bipolar', 
                   'Brain cancer', 
                   'Breast cancer', 
                   'COPD (chronic obstructive pulmonary disease)', 
                   'Chronic kidney disease', 
                   'Circadian rhythm disorders', 
                   'Colon cancer',
                   'Depression', 
                   'Diabetes', 
                   'Endometrial cancer', 
                   'Fatty liver disease', 
                   'HIV/AIDS', 
                   'Heart failure', 
                   'Hepatitis', 
                   'High cholesterol', 
                   'Hypertension', 
                   'Insomnia', 
                   'Kidney cancer', 
                   'Leukemia (cancer)', 
                   'Liver cancer', 
                   'Long covid', 
                   'Lung cancer', 
                   'ME/CFS', 
                   'Melanoma (skin cancer)', 
                   'Myocarditis', 
                   'Osteoporosis', 
                   'POTS (postural orthostatic tachycardia syndrome)', 
                   'Pancreatic cancer', 
                   "Parkinson's", 
                   'Previous stroke', 
                   'Prostate cancer', 
                   'Pulmonary fibrosis', 
                   'Restless leg syndrome', 
                   'SVT (Supraventricular tachycardia)', 
                   'Schizophrenia', 
                   'Sick Sinus Syndrome', 
                   'Sleep apnea', 
                   'Substance abuse', 
                   'Thyroid cancer', 
                   'Ventricular Arrhythmias', 
                   'WPW (Wolff-Parkinson-White Syndrome)', 
                   ]
    
    # Filter targets that actually exist in binary_df
    available_targets = [target for target in all_targets if target in binary_df.columns]
    missing_targets = [target for target in all_targets if target not in binary_df.columns]
    
    if missing_targets:
        print(f"Warning: The following targets are not available in binary_df: {missing_targets}")
    
    print(f"Evaluating {len(available_targets)} available target conditions simultaneously...")
    print("Available targets:", available_targets[:5], "..." if len(available_targets) > 5 else "")
    print("=" * 80)
    
    # Create output directory
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create multi-target dataloaders
        print("Creating multi-target dataloaders...")
        train_loader, val_loader, dataset = create_multi_target_dataloader(args, df, binary_df, available_targets)
        
        target_info = dataset.get_target_info()
        print(f"Dataset info: {target_info['num_targets']} targets, {len(dataset)} samples")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Calculate positive sample percentages
        train_percentages, val_percentages = calculate_positive_sample_percentages(dataset, available_targets)
        print("\nPositive Sample Percentages:")
        for target in available_targets:
            print(f"  {target}: Train={train_percentages[target]:.2f}%, Val={val_percentages[target]:.2f}%")

        # Evaluate baseline for all targets
        print(f"\nEvaluating baseline for all {len(available_targets)} targets...")
        baseline_results = evaluate_mean_pooling_baseline_multi_target(train_loader, val_loader, "cuda", available_targets)
        
        # Evaluate model for all targets
        print(f"\nEvaluating model for all {len(available_targets)} targets...")
        model_results = evaluate_model_multi_target(model, train_loader, val_loader, "cuda", available_targets)
        
        # Compile results
        results = []
        for target in available_targets:
            baseline = baseline_results.get(target, {'auroc': None, 'auprc': None})
            model = model_results.get(target, {'auroc': None, 'auprc': None})
            
            baseline_auroc = baseline['auroc']
            baseline_auprc = baseline['auprc']
            model_auroc = model['auroc']
            model_auprc = model['auprc']
            
            results.append({
                'target_condition': target,
                'baseline_auroc': baseline_auroc,
                'baseline_auprc': baseline_auprc,
                'model_auroc': model_auroc,
                'model_auprc': model_auprc,
                'auroc_improvement': model_auroc - baseline_auroc if (model_auroc is not None and baseline_auroc is not None) else None,
                'auprc_improvement': model_auprc - baseline_auprc if (model_auprc is not None and baseline_auprc is not None) else None,
                'train_pos_percent': train_percentages.get(target, 0.0),
                'val_pos_percent': val_percentages.get(target, 0.0)
            })
        
        # Save results
        print("\n" + "=" * 80)
        print("Saving results...")
        
        results_df = pd.DataFrame(results)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_filename = os.path.join(output_dir, f"multi_target_evaluation_results_{timestamp}.csv")
        results_df.to_csv(final_filename, index=False)
        
        print(f"Results saved to: {final_filename}")
        
        # Display summary statistics
        print("\nSummary Statistics:")
        print("-" * 40)
        
        # Filter out None values for statistics
        valid_results = results_df.dropna(subset=['baseline_auroc', 'model_auroc'])
        
        if len(valid_results) > 0:
            print(f"Successfully evaluated: {len(valid_results)}/{len(available_targets)} conditions")
            print(f"Average baseline AUROC: {valid_results['baseline_auroc'].mean():.4f}")
            print(f"Average model AUROC: {valid_results['model_auroc'].mean():.4f}")
            print(f"Average AUROC improvement: {valid_results['auroc_improvement'].mean():.4f}")
            print(f"Average baseline AUPRC: {valid_results['baseline_auprc'].mean():.4f}")
            print(f"Average model AUPRC: {valid_results['model_auprc'].mean():.4f}")
            print(f"Average AUPRC improvement: {valid_results['auprc_improvement'].mean():.4f}")
            
            # Add positive sample percentage statistics
            print(f"\nPositive Sample Statistics:")
            print(f"Average train positive %: {valid_results['train_pos_percent'].mean():.2f}%")
            print(f"Average val positive %: {valid_results['val_pos_percent'].mean():.2f}%")
            
            print(f"\nTop 5 conditions with largest AUROC improvements:")
            top_auroc = valid_results.nlargest(5, 'auroc_improvement')[['target_condition', 'auroc_improvement', 'train_pos_percent', 'val_pos_percent']]
            for _, row in top_auroc.iterrows():
                print(f"  {row['target_condition']}: +{row['auroc_improvement']:.4f} (Train: {row['train_pos_percent']:.1f}%, Val: {row['val_pos_percent']:.1f}%)")
                
            print(f"\nTop 5 conditions with largest AUPRC improvements:")
            top_auprc = valid_results.nlargest(5, 'auprc_improvement')[['target_condition', 'auprc_improvement', 'train_pos_percent', 'val_pos_percent']]
            for _, row in top_auprc.iterrows():
                print(f"  {row['target_condition']}: +{row['auprc_improvement']:.4f} (Train: {row['train_pos_percent']:.1f}%, Val: {row['val_pos_percent']:.1f}%)")
        else:
            print("No valid results obtained.")
        
        print(f"\nMulti-target evaluation complete! Results saved to {final_filename}")
        
    except Exception as e:
        print(f"Critical error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        error_info = {
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'available_targets': available_targets
        }
        
        error_filename = os.path.join(output_dir, f"evaluation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(error_filename, 'w') as f:
            f.write(f"Error: {error_info}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
        
        print(f"Error details saved to: {error_filename}")