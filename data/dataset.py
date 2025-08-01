
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd 
import pickle
import gc



class EmpiricalDatasetIMTS(Dataset):
    """
    Dataset subclass adapted for IMTS model that outputs triplets (time, variable_id, value).
    This version chunks long sequences into multiple samples and supports multiple target columns.
    """

    def __init__(
        self,
        args,
        df,
        binary_df=None,  # Optional for finetuning compatibility
        timeseries_columns=None,
        binary_columns=None,  # Not used in IMTS
        target_columns=None,  # Can be a single string or list of strings
        date_column="date",
        user_id_column="appUserId",
        min_obs_per_user=100,
        min_timespan_days=20,
        train_frac=1.0,
        run="1o1",
        is_pretrain=True,
        outlier_method="none",
        outlier_threshold=8.0,
        max_seq_len=2000,  # Maximum sequence length for IMTS
        load_from_cache= False,
        cache_dir = "dataset_cache",

    ):
        # Define a static path for the cache file
        if is_pretrain:
            cache_filename = "cached_imts_dataset.pkl"
        else:
            cache_filename = "finetune_cached_imts_dataset.pkl"

        cache_path = os.path.join(cache_dir, cache_filename)
        self.args = args
        self.date_column = date_column
        self.user_id_column = user_id_column
        
        # Handle target_columns as either string or list
        if target_columns is None:
            self.target_columns = None
        elif isinstance(target_columns, str):
            self.target_columns = [target_columns]
        elif isinstance(target_columns, list):
            self.target_columns = target_columns
        else:
            raise ValueError("target_columns must be None, a string, or a list of strings")
        
        # Keep backward compatibility with target_column
        self.target_column = target_columns if isinstance(target_columns, str) else None
        
        self.binary_df = binary_df
        self.binary_columns = binary_columns
        self.timeseries_columns = timeseries_columns
        self.is_pretrain = is_pretrain
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.max_seq_len = max_seq_len

        if load_from_cache:
            print(f"Attempting to load dataset from cache: {cache_path}")
            if not os.path.exists(cache_path):
                raise FileNotFoundError("")
            
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Load all necessary attributes from the cached dictionary
            self.triplets = cached_data['triplets']
            self.splits = cached_data['splits']
            self.N = cached_data['N']
            if 'y' in cached_data:
                self.y = cached_data['y']
            if 'target_columns' in cached_data:
                self.target_columns = cached_data['target_columns']
            
            print("Dataset loaded successfully from cache.")
            return 
        
        print(
            f"\nPreparing IMTS dataset with {len(df)} rows. {len(df[user_id_column].unique())} users. Pre-training mode: {self.is_pretrain}"
        )
        if self.target_columns:
            print(f"Target columns: {self.target_columns}")
            
        if binary_df is not None:
            binary_df = binary_df.drop_duplicates(subset=[self.user_id_column])

        # Filter users for finetuning if binary_df is provided
        if not self.is_pretrain and binary_df is not None:
            df = self._filter_users_by_binary_df(df, binary_df)

        self.timeseries_columns = timeseries_columns
        data, targets = self._prepare_data(
            df, binary_df, min_obs_per_user, min_timespan_days
        )

        # Chunk users with sequences longer than max_seq_len
        data, targets = self._chunk_long_sequences(data, targets)
        
        # Clean up original dataframe to free memory
        del df
        if binary_df is not None and not self.is_pretrain:
            del binary_df

        train_ids, val_ids, test_ids = self._create_splits(data, train_frac, run)

        # Handle splits based on pretraining vs finetuning
        if self.is_pretrain:
            data = data.loc[~data.ts_id.isin(test_ids)]
            train_ids = np.setdiff1d(data.ts_id.unique(), val_ids)

        # Ensure all variables are present in training set
        train_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        all_variables = data.variable.unique()
        delete_variables = np.setdiff1d(all_variables, train_variables)
        if len(delete_variables) > 0:
            print("Removing variables not in training set: " + str(delete_variables))
            data = data.loc[data.variable.isin(train_variables)]

        # Update IDs after filtering
        curr_ids = data.ts_id.unique()
        train_ids = np.intersect1d(train_ids, curr_ids)
        val_ids = np.intersect1d(val_ids, curr_ids)
        test_ids = np.intersect1d(test_ids, curr_ids)

        split_ids = (
            (train_ids, val_ids, test_ids)
            if not self.is_pretrain
            else (train_ids, val_ids)
        )
        sup_ts_ids = np.concatenate(split_ids)

        # Create mapping from ts_id to index
        ts_id_to_ind = {ts_id: i for i, ts_id in enumerate(sup_ts_ids)}
        data = data.loc[data.ts_id.isin(sup_ts_ids)]
        data["ts_ind"] = data["ts_id"].map(ts_id_to_ind)
        self.N = len(sup_ts_ids)

        # Create splits dictionary
        self.splits = {
            "train": [ts_id_to_ind[i] for i in train_ids],
            "val": [ts_id_to_ind[i] for i in val_ids],
        }
        
        if not self.is_pretrain and targets is not None:
            self.splits["test"] = [ts_id_to_ind[i] for i in test_ids]
            targets = targets.loc[targets.ts_id.isin(sup_ts_ids)]
            targets["ts_ind"] = targets["ts_id"].map(ts_id_to_ind)
            targets = targets.sort_values(by="ts_ind")

            # Safety check: ensure targets align with number of samples
            expected_targets = len(sup_ts_ids)
            actual_targets = len(targets)

            if actual_targets != expected_targets:
                print(f"WARNING: Mismatch between targets ({actual_targets}) and samples ({expected_targets})")
                print(f"Missing targets for ts_ids: {set(sup_ts_ids) - set(targets.ts_id.unique())}")
                raise ValueError(f"Target count mismatch: expected {expected_targets}, got {actual_targets}")

            # Handle multiple target columns
            if len(self.target_columns) == 1:
                # Single target - maintain backward compatibility
                self.y = np.array(targets[self.target_columns[0]], dtype=np.float32)
            else:
                # Multiple targets - create 2D array
                self.y = np.array(targets[self.target_columns].values, dtype=np.float32)

            # Verify the final array dimensions
            expected_shape = (self.N, len(self.target_columns)) if len(self.target_columns) > 1 else (self.N,)
            if self.y.shape != expected_shape:
                raise ValueError(f"Final target array shape ({self.y.shape}) doesn't match expected shape ({expected_shape})")

            print(f"Target array shape: {self.y.shape}")
            
        # Preprocess data for IMTS model
        self._preprocess_for_imts(data, args, train_ids)
        
        # Clean up the data DataFrame after preprocessing
        del data

        print(f"Dataset prepared with {self.N} samples")
        print(f"Max sequence length: {max([len(seq) for seq in self.triplets])}")
        
        # --- Caching Logic: Save ---
        print(f"Saving processed dataset to cache: {cache_path}")
        os.makedirs(cache_dir, exist_ok=True)
        
        data_to_cache = {
            'triplets': self.triplets,
            'splits': self.splits,
            'N': self.N,
            'target_columns': self.target_columns,
        }
        if hasattr(self, 'y'):
            data_to_cache['y'] = self.y
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data_to_cache, f)
        print("Caching complete. Use `load_from_cache=True` on next run to load.")
        


    def _filter_users_by_binary_df(self, df, binary_df):
        """Filter the main dataframe to only include users that exist in binary_df."""
        initial_users = df[self.user_id_column].nunique()
        available_users = set(binary_df[self.user_id_column].unique())

        # Use in-place filtering to avoid copying
        mask = df[self.user_id_column].isin(available_users)
        df_filtered = df[mask].copy()

        final_users = df_filtered[self.user_id_column].nunique()
        removed_users = initial_users - final_users

        print(
            f"Filtered users for finetuning: {final_users} users kept, {removed_users} users removed (not in binary_df)"
        )

        if final_users == 0:
            raise ValueError(
                "No users remain after filtering by binary_df. Check that user IDs match between dataframes."
            )

        return df_filtered

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.N

    def __getitem__(self, idx):
        """
        Returns a triplet tensor for the given index.
        Each triplet has shape (seq_len, 3) where columns are [time, variable_id, value]
        
        For finetuning:
        - Single target: returns (triplets, target_value)
        - Multiple targets: returns (triplets, target_vector)
        """
        triplets = torch.FloatTensor(self.triplets[idx])
        
        if not self.is_pretrain and hasattr(self, 'y'):
            if len(self.y.shape) == 1:
                # Single target - backward compatibility
                return triplets, torch.FloatTensor([self.y[idx]])
            else:
                # Multiple targets
                return triplets, torch.FloatTensor(self.y[idx])
        else:
            # Return only triplets for pretraining
            return triplets

    def get_target_info(self):
        """
        Returns information about the target columns.
        Useful for model configuration.
        """
        if self.target_columns is None:
            return {"num_targets": 0, "target_names": []}
        return {
            "num_targets": len(self.target_columns), 
            "target_names": self.target_columns
        }

    def _prepare_data(self, df, binary_df, min_obs_per_user, min_timespan_days):
        """Prepare and clean the time series data with memory optimization."""
        print("Converting data to long format...")
        
        # Convert date column in-place
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")

        # Remove invalid and outlier timestamps in-place
        df.dropna(subset=[self.date_column], inplace=True)
        
        start_date = pd.Timestamp("2023-01-01", tz=df[self.date_column].dt.tz)
        end_date = pd.Timestamp("2025-06-01", tz=df[self.date_column].dt.tz)
        
        date_mask = (df[self.date_column] >= start_date) & (df[self.date_column] <= end_date)
        df = df[date_mask].copy()  # Need copy here as we're changing the index
        print(f"After date filtering (2023-2025): {len(df)} rows")

        # Convert to days since minimum date
        min_date = df[self.date_column].min()
        df["day"] = (df[self.date_column] - min_date).dt.days

        # Handle targets for finetuning
        targets = None
        if not self.is_pretrain and binary_df is not None and self.target_columns is not None:
            
            # Extract all target columns plus user ID
            target_cols_to_extract = [self.user_id_column] + self.target_columns
            targets = binary_df[target_cols_to_extract].copy()
            targets.rename(columns={self.user_id_column: "ts_id"}, inplace=True)
            
            print(f"Extracted targets for {len(self.target_columns)} columns: {self.target_columns}")

        # Convert to long format using more memory-efficient approach
        id_vars = [self.user_id_column,  "day"]
        
        # Melt and clean in one go to avoid multiple copies
        data_long = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=self.timeseries_columns,
            var_name="variable",
            value_name="value",
        )
        print("finished melting")
        # Clean up original df
        del df
        gc.collect()
        
        # Remove NaN values in-place
        data_long.dropna(subset=["value"], inplace=True)
        data_long.rename(columns={self.user_id_column: "ts_id"}, inplace=True)
        
        data_long["value"] = data_long["value"]
        data_long["day"] = data_long["day"]

        initial_users = data_long["ts_id"].nunique()
        initial_obs = len(data_long)
        print(f"Initial: {initial_obs} observations for {initial_users} users")

        # Remove outliers
        if self.outlier_method != "none":
            print(f"Removing outliers using {self.outlier_method} method...")
            data_long = self._remove_outliers_optimized(data_long)
            after_outlier_obs = len(data_long)
            print(
                f"After outlier removal: {after_outlier_obs} observations ({initial_obs - after_outlier_obs} removed)"
            )

        # Filter by minimum observations per user using more efficient approach
        user_counts = data_long.groupby("ts_id").size()
        valid_users = user_counts[user_counts >= min_obs_per_user].index
        data_long = data_long[data_long["ts_id"].isin(valid_users)].copy()

        users_after_obs_filter = data_long["ts_id"].nunique()
        print(
            f"Users remaining after min_obs filter ({min_obs_per_user}): {users_after_obs_filter} ({initial_users - users_after_obs_filter} removed)"
        )

        print(
            f"After all cleaning: {len(data_long)} observations for {data_long.ts_id.nunique()} users"
        )
        return data_long, targets

    def _chunk_long_sequences(self, data, targets):
        """
        Chunks sequences for users with more observations than max_seq_len.
        Creates a new "user" for each chunk.
        """
        print("   - Chunking long sequences...")
        data = data.sort_values(['ts_id', 'day'])

        # Get observation counts for each user in a single pass
        counts = data.groupby('ts_id')['day'].transform('count')
        needs_chunking_mask = counts > self.max_seq_len

        if not needs_chunking_mask.any():
            print("   - No users needed chunking.")
            return data, targets

        # Store original ts_ids that need chunking
        original_chunked_users = data.loc[needs_chunking_mask, 'ts_id'].unique()

        # Create chunk numbers for rows that need it using cumcount
        data.loc[needs_chunking_mask, 'chunk_num'] = data.loc[needs_chunking_mask].groupby('ts_id').cumcount() // self.max_seq_len

        # Generate new, unique IDs for the chunks
        original_ids_for_chunking = data.loc[needs_chunking_mask, 'ts_id']
        chunk_numbers = data.loc[needs_chunking_mask, 'chunk_num'].astype(int).astype(str)
        new_chunk_ids = original_ids_for_chunking.astype(str) + '_chunk_' + chunk_numbers
        data.loc[needs_chunking_mask, 'ts_id'] = new_chunk_ids

        # Replicate targets for the newly created chunks
        if targets is not None:
            # Get all the new chunk IDs that were created
            chunk_id_mapping = pd.DataFrame({
                'original_ts_id': original_ids_for_chunking,
                'new_ts_id': new_chunk_ids
            }).drop_duplicates()

            # Get targets for users that were chunked
            targets_to_replicate = targets[targets['ts_id'].isin(original_chunked_users)].copy()

            # Create new target rows for each chunk
            new_target_rows = []
            for _, row in targets_to_replicate.iterrows():
                original_id = row['ts_id']
                # Get all chunk IDs for this original user
                chunk_ids = chunk_id_mapping[chunk_id_mapping['original_ts_id'] == original_id]['new_ts_id'].values

                for chunk_id in chunk_ids:
                    new_row = row.copy()
                    new_row['ts_id'] = chunk_id
                    new_target_rows.append(new_row)

            # Create DataFrame from new rows
            if new_target_rows:
                replicated_targets = pd.DataFrame(new_target_rows)

                # Combine untouched targets with the newly replicated ones
                targets = pd.concat([
                    targets[~targets['ts_id'].isin(original_chunked_users)],
                    replicated_targets
                ], ignore_index=True)

        data.drop(columns=['chunk_num'], inplace=True)

        print(f"   - Expanded users to {data.ts_id.nunique()} after chunking.")
        return data, targets

    def _remove_outliers_optimized(self, data_long):
        """Remove outliers using the specified method with memory optimization."""
        if self.outlier_method == "none":
            return data_long

        initial_count = len(data_long)
        
        # Use boolean indexing instead of creating separate outlier_mask
        keep_mask = pd.Series([True] * len(data_long), index=data_long.index)

        # Apply outlier detection per variable using vectorized operations
        for variable in data_long["variable"].unique():
            var_mask = data_long["variable"] == variable
            var_data = data_long.loc[var_mask, "value"]

            if self.outlier_method == "iqr":
                Q1 = var_data.quantile(0.25)
                Q3 = var_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                var_outliers = (var_data < lower_bound) | (var_data > upper_bound)

            elif self.outlier_method == "zscore":
                # Use more memory-efficient z-score calculation
                mean_val = var_data.mean()
                std_val = var_data.std()
                if std_val > 0:
                    z_scores = np.abs((var_data - mean_val) / std_val)
                    var_outliers = z_scores > self.outlier_threshold
                else:
                    var_outliers = pd.Series([False] * len(var_data), index=var_data.index)

            else:
                raise ValueError(f"Unknown outlier method: {self.outlier_method}")

            keep_mask.loc[var_mask] = keep_mask.loc[var_mask] & ~var_outliers

            outlier_count = var_outliers.sum()
            if outlier_count > 0:
                print(
                    f"   Variable {variable}: removed {outlier_count} outliers ({outlier_count/len(var_data)*100:.2f}%)"
                )

        data_cleaned = data_long[keep_mask].copy()
        total_removed = initial_count - len(data_cleaned)
        print(
            f"Total outliers removed: {total_removed} ({total_removed/initial_count*100:.2f}%)"
        )

        return data_cleaned

    def _create_splits(self, data, train_frac, run):
        """Create train/val/test splits."""
        unique_ids = data.ts_id.unique()
        np.random.shuffle(unique_ids)
        run_num, total_runs = list(map(int, run.split("o")))
        n_total = len(unique_ids)
        if self.is_pretrain:
            n_train_full = int(0.85 * n_total)
            n_val_full = int(0.15 * n_total)
        else:
            n_train_full = int(0.5 * n_total)
            n_val_full = int(0.5 * n_total)
        train_ids_full = unique_ids[:n_train_full]
        val_ids_full = unique_ids[n_train_full : n_train_full + n_val_full]
        test_ids = unique_ids[n_train_full + n_val_full :]
        
        num_train = int(np.ceil(train_frac * len(train_ids_full)))
        start_train = int(
            np.linspace(0, len(train_ids_full) - num_train, total_runs)[run_num - 1]
        )
        train_ids = train_ids_full[start_train : start_train + num_train]
        
        num_val = int(np.ceil(train_frac * len(val_ids_full)))
        start_val = int(
            np.linspace(0, len(val_ids_full) - num_val, total_runs)[run_num - 1]
        )
        val_ids = val_ids_full[start_val : start_val + num_val]
        
        print(
            f"Split sizes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}"
        )
        return train_ids, val_ids, test_ids

    def _preprocess_for_imts(self, data, args, train_ids):
        """Preprocess data specifically for IMTS model format with memory optimization."""
        
        # Check if we're finetuning (loading from pretrained model)
        args.finetune = (
            hasattr(args, "load_ckpt_path") and args.load_ckpt_path is not None
        )
        
        if args.finetune:
            # Load normalization stats from pretrained model
            pt_var_path = "normalization_stats.pkl"
            variables, normalization_stats, max_day = pickle.load(
                open(pt_var_path, "rb")
            )
            print(f"Loaded normalization stats for {len(variables)} variables")
        else:
            # Calculate normalization statistics from training data
            data.sort_values(by=["ts_id", "day"], inplace=True)
            
            # NOTE: The old truncation logic was removed from here, as the
            # new _chunk_long_sequences method handles it more effectively.

            train_data = data.loc[data.ts_id.isin(train_ids)]
            normalization_stats = {}
            print("Calculating normalization statistics per variable")

            # Vectorized normalization calculation
            for variable in train_data["variable"].unique():
                var_data = train_data[train_data["variable"] == variable]["value"]
                mean_val = float(var_data.mean())
                std_val = float(var_data.std())

                # Handle zero variance
                if std_val == 0 or np.isnan(std_val):
                    print(
                        f"Warning: Variable {variable} has zero/nan std in training data, using std=1"
                    )
                    std_val = 1.0

                normalization_stats[variable] = {"mean": mean_val, "std": std_val}
                print(f"  {variable}: mean={mean_val:.4f}, std={std_val:.4f}")

            # Calculate max_day for time normalization
            self.max_day_train = int(train_data["day"].max())
            print(f"Max day from training data: {self.max_day_train}")

            # Save normalization stats
            variables = list(normalization_stats.keys())
            save_path = getattr(
                args, "normalization_save_path", "normalization_stats.pkl"
            )
            with open(save_path, "wb") as f:
                pickle.dump((variables, normalization_stats, self.max_day_train), f)
            print(f"Saved normalization statistics to {save_path}")

        # Apply normalization
        print("Applying normalization to all data...")
        if args.finetune:
            self.max_day_train = max_day
            # Convert old format to new format if needed
            if isinstance(normalization_stats, pd.DataFrame):
                norm_dict = {}
                for var in normalization_stats.index:
                    norm_dict[var] = {
                        "mean": normalization_stats.loc[var, "mean"],
                        "std": normalization_stats.loc[var, "std"],
                    }
                normalization_stats = norm_dict

        # Vectorized normalization
        self._apply_normalization_vectorized(data, normalization_stats)

        # Create variable mapping
        if not args.finetune:
            variables = data.variable.unique()
        var_to_ind = {v: i for i, v in enumerate(variables)}
        V = len(variables)
        
        args.V = V
        print("# TS variables: " + str(V))

        # Convert to IMTS triplet format
        self._create_triplets_optimized(data, var_to_ind)

    def _apply_normalization_vectorized(self, data, normalization_stats):
        """Apply normalization using vectorized operations."""
        for variable in data.variable.unique():
            var_mask = data.variable == variable
            
            if variable in normalization_stats:
                mean_val = normalization_stats[variable]["mean"]
                std_val = normalization_stats[variable]["std"]
                data.loc[var_mask, "value"] = (data.loc[var_mask, "value"] - mean_val) / std_val
            else:
                print(
                    f"Warning: Variable {variable} not found in normalization stats, using original value"
                )

    def _create_triplets_optimized(self, data, var_to_ind):
        """Convert irregular time series data to IMTS triplet format with memory optimization."""
        print("Converting to triplet format for IMTS...")
        
        # Initialize triplets list
        triplets = [None] * self.N
        
        # Group by ts_ind for efficient processing
        print("Grouping data by time series...")
        grouped = data.groupby('ts_ind')
        
        print("Creating triplets...")
        for ts_ind, group in grouped:
            # Vectorized operations for each group
            times = group['day'].values.astype(np.float32)
            
            # Normalize time to [-1, 1] range as expected by IMTS
            normalized_times = (times / max(self.max_day_train, 1)) * 2 - 1
            normalized_times = np.clip(normalized_times, -2, 2)  # Clamp to [-2, 2]
            
            # Map variables to indices
            var_ids = group['variable'].map(var_to_ind).values.astype(np.int32)
            values = group['value'].values.astype(np.float32)
            
            # Create triplet array
            triplet_array = np.column_stack([normalized_times, var_ids, values])
            
            # Sort by time (first column)
            time_order = np.argsort(triplet_array[:, 0])
            triplet_array = triplet_array[time_order]
            
            # NOTE: Redundant truncation logic removed. Chunking is handled earlier.
            
            triplets[ts_ind] = triplet_array
        
        # Handle any empty sequences that might result from splitting
        for i in range(self.N):
            if triplets[i] is None:
                # Handle empty sequences with dummy triplet
                triplets[i] = np.array([[0.0, 0, 0.0]], dtype=np.float32)
        
        self.triplets = triplets
        
        # Print statistics
        seq_lengths = [len(seq) for seq in triplets]
        print(f"Triplet sequences created:")
        print(f"  Min length: {min(seq_lengths)}")
        print(f"  Max length: {max(seq_lengths)}")
        print(f"  Mean length: {np.mean(seq_lengths):.2f}")
        print(f"  Median length: {np.median(seq_lengths):.2f}")
        
        # Final memory cleanup
        gc.collect()


def collate_triplets(batch):
    """
    Custom collate function for IMTS triplets that handles variable-length sequences.
    Pads sequences to the same length within each batch and returns padding masks.
    
    Returns:
        For pretraining: (triplets_tensor, padding_mask)
        For finetuning: (triplets_tensor, padding_mask, labels)
    """
    if isinstance(batch[0], tuple):
        # Finetuning case: (triplets, labels)
        triplets, labels = zip(*batch)
        labels = torch.stack(labels)
    else:
        # Pretraining case: just triplets
        triplets = batch
        labels = None
    
    # Find max sequence length in this batch
    max_len = max(len(seq) for seq in triplets)
    
    # Pad sequences and create masks
    padded_triplets = []
    padding_masks = []
    
    for seq in triplets:
        seq_len = len(seq)
        seq_tensor = torch.FloatTensor(seq)
        
        # Create padding mask: True for real tokens, False for padding
        mask = torch.ones(seq_len, dtype=torch.bool)
        
        if seq_len < max_len:
            # Pad with zeros (time=0, var_id=0, value=0)
            padding = torch.zeros(max_len - seq_len, 3)
            seq_tensor = torch.cat([seq_tensor, padding], dim=0)
            
            # Extend mask with False for padded positions
            padding_mask = torch.zeros(max_len - seq_len, dtype=torch.bool)
            mask = torch.cat([mask, padding_mask], dim=0)
        
        padded_triplets.append(seq_tensor)
        padding_masks.append(mask)
    
    triplets_tensor = torch.stack(padded_triplets)
    padding_mask = torch.stack(padding_masks)  # Shape: (batch_size, seq_len)
    
    if labels is not None:
        return triplets_tensor, padding_mask, labels
    else:
        return triplets_tensor, padding_mask

def collate_triplets_forecast(batch, forecast_percent=0.7):
    """
    Custom collate function for IMTS triplets that handles variable-length sequences.
    Pads sequences to the same length within each batch and returns padding masks.
    
    Args:
        batch: List of triplets (and optionally labels)
        forecast_percent: Float between 0 and 1, percentage of sequence to mask for forecasting
    
    Returns:
        For pretraining: (triplets_tensor, forecast_values, combined_forecast_mask)
        For finetuning: (triplets_tensor, forecast_values, combined_forecast_mask, labels)
    """
    if isinstance(batch[0], tuple):
        # Finetuning case: (triplets, labels)
        triplets, labels = zip(*batch)
        labels = torch.stack(labels)
    else:
        # Pretraining case: just triplets
        triplets = batch
        labels = None
    
    # Find max sequence length in this batch
    max_len = max(len(seq) for seq in triplets)
    
    # Pad sequences and create masks
    padded_triplets = []
    padding_masks = []
    forecast_values = []
    forecast_masks = []
    
    for seq in triplets:
        seq_len = len(seq)
        seq_tensor = torch.FloatTensor(seq)
        
        # Calculate forecast length for this sequence
        forecast_len = int(seq_len * forecast_percent)
        forecast_start = seq_len - forecast_len if forecast_len > 0 else seq_len
        
        # Create padding mask: True for real tokens, False for padding
        mask = torch.ones(seq_len, dtype=torch.bool)
        
        # Create forecast mask: True for positions to forecast, False otherwise
        forecast_mask = torch.zeros(seq_len, dtype=torch.bool)
        if forecast_len > 0:
            forecast_mask[forecast_start:] = True
        
        # Extract forecast values before masking
        forecast_vals = seq_tensor.clone()
        
        # Mask the forecast portion in the input (set to zeros)
        if forecast_len > 0:
            seq_tensor[forecast_start:] = 0
        
        if seq_len < max_len:
            # Pad with zeros (time=0, var_id=0, value=0)
            padding = torch.zeros(max_len - seq_len, 3)
            seq_tensor = torch.cat([seq_tensor, padding], dim=0)
            forecast_vals = torch.cat([forecast_vals, padding], dim=0)
            
            # Extend padding mask with False for padded positions
            padding_mask = torch.zeros(max_len - seq_len, dtype=torch.bool)
            mask = torch.cat([mask, padding_mask], dim=0)
            
            # Extend forecast mask with False for padded positions
            forecast_padding = torch.zeros(max_len - seq_len, dtype=torch.bool)
            forecast_mask = torch.cat([forecast_mask, forecast_padding], dim=0)
        
        padded_triplets.append(seq_tensor)
        padding_masks.append(mask)
        forecast_values.append(forecast_vals)
        forecast_masks.append(forecast_mask)
    
    triplets_tensor = torch.stack(padded_triplets)
    padding_mask = torch.stack(padding_masks)  # Shape: (batch_size, seq_len)
    forecast_values_tensor = torch.stack(forecast_values)  # Shape: (batch_size, seq_len, 3)
    forecast_mask_tensor = torch.stack(forecast_masks)  # Shape: (batch_size, seq_len)
    
    # Combine padding and forecast masks: True if in forecast range AND not padded
    combined_forecast_mask = forecast_mask_tensor & padding_mask
    
    if labels is not None:
        return triplets_tensor, forecast_values_tensor, combined_forecast_mask, labels
    else:
        return triplets_tensor, forecast_values_tensor, combined_forecast_mask