from probe_diagnosis import * 

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
    save_path: Optional[str] = "tsne.png"
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
    plt.savefig("t_sne.png", dpi=1200)
    plt.show()
    
    print(f"\nLabel combination statistics:")
    for combo, count in zip(unique_combinations, counts):
        print(f"  {combo}: {count} samples ({count/len(combination_labels)*100:.1f}%)")

if __name__ == "__main__":

    args = IMTSConfig()
    args.pretrain = 0
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
                'COPD (chronic obstructive pulmonary disease)', 
                'Chronic kidney disease', 
                'Circadian rhythm disorders', 
                'Depression', 
                'Diabetes', 
                'Fatty liver disease', 
                'HIV/AIDS', 
                'Heart failure', 
                'Hepatitis', 
                'High cholesterol', 
                'Hypertension', 
                'Insomnia', 
                'Long covid', 
                'ME/CFS', 
                'Myocarditis', 
                'Osteoporosis', 
                'POTS (postural orthostatic tachycardia syndrome)', 
                "Parkinson's", 
                'Previous stroke', 
                'Pulmonary fibrosis', 
                'Restless leg syndrome', 
                'SVT (Supraventricular tachycardia)', 
                'Schizophrenia', 
                'Sick Sinus Syndrome', 
                'Sleep apnea', 
                'Substance abuse', 
                'Ventricular Arrhythmias', 
                'WPW (Wolff-Parkinson-White Syndrome)', 
                ]

    # Filter targets that actually exist in binary_df
    available_targets = [target for target in all_targets if target in binary_df.columns]
    missing_targets = [target for target in all_targets if target not in binary_df.columns]

    if missing_targets:
        print(f"⚠️  Warning: The following targets are not available in binary_df: {missing_targets}")

    print(f"📊 Evaluating {len(available_targets)} available target conditions simultaneously...")
    print("Available targets:", available_targets[:5], "..." if len(available_targets) > 5 else "")
            
    # Create multi-target dataloaders
    print("Creating multi-target dataloaders...")
    train_loader, val_loader, dataset = create_multi_target_dataloader(args, df, binary_df, available_targets)
    embedding_loader = precompute_and_pool_embeddings_multi_target(model, val_loader, "cuda", len(available_targets))
    print("Generating T-SNE visualization...")
    visualize_embeddings_tsne_multi_target(embedding_loader)
