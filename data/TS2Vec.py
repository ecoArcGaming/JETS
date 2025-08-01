import torch
import numpy as np
import argparse
import os
import time
import datetime
import pandas as pd
from ts2vec import TS2Vec
from data.config import IMTSConfig
from data.dataset import EmpiricalDatasetIMTS
from ts2vec.utils import init_dl_program

def save_checkpoint_callback(save_every=1, unit='epoch'):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback



def load_empirical_imts(args, df, binary_df=None, timeseries_columns=None, target_column=None, 
                       date_column="date", user_id_column="appUserId", min_obs_per_user=100, 
                       min_timespan_days=20, train_frac=1.0, run="1o1", is_pretrain=True, 
                       outlier_method="none", outlier_threshold=8.0, max_seq_len=5000, 
                       load_from_cache=False, cache_dir="dataset_cache"):
    """
    Load EmpiricalDataSetIMTS and convert to ts2vec format.
    Returns data in the same format as other ts2vec dataset loaders.
    """
    # Create the IMTS dataset
    dataset = EmpiricalDatasetIMTS(
        args=args, df=df, binary_df=binary_df, timeseries_columns=timeseries_columns,
        target_column=target_column, date_column=date_column, user_id_column=user_id_column,
        min_obs_per_user=min_obs_per_user, min_timespan_days=min_timespan_days,
        train_frac=train_frac, run=run, is_pretrain=is_pretrain,
        outlier_method=outlier_method, outlier_threshold=outlier_threshold,
        max_seq_len=max_seq_len, load_from_cache=load_from_cache, cache_dir=cache_dir
    )
    
    # Convert triplets to ts2vec format
    all_data = []
  
    max_day = 0
    for triplet_seq in dataset.triplets:
        if len(triplet_seq) > 0:
            # First column contains time values
            max_day_in_seq = np.max(triplet_seq[:, 0])
            max_day = max(max_day, max_day_in_seq)
    seq_len = int(max_day)
    print("MAX DAY", seq_len)
    n_vars = 63
    
    for triplet in dataset.triplets:
        # Initialize sequence array with NaN
        seq_array = np.full((seq_len, n_vars), np.nan, dtype=np.float32)
        
        # Fill in the values from triplets
        for row in triplet:
            time_idx = int(row[0])  # Use actual day index
            time_idx = min(max(time_idx, 0), seq_len - 1)  # Clamp to valid range
            var_idx = int(row[1])
            value = row[2]
            
            if var_idx < n_vars:
                seq_array[time_idx, var_idx] = value
        
        all_data.append(seq_array)
    
    # Stack into final array format: (n_samples, seq_len, n_vars)
    data = np.stack(all_data)
    
    # Create splits based on dataset splits
    train_indices = dataset.splits['train']
    val_indices = dataset.splits['val']
    test_indices = dataset.splits.get('test', [])
    
    # Return in ts2vec format
    if is_pretrain:
        # For pretraining, return the data and splits
        return data, train_indices, val_indices, None
    else:
        # For finetuning, also return labels
        labels = dataset.y if hasattr(dataset, 'y') else None
        return data, train_indices, val_indices, test_indices, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TS2Vec Pretraining Only')
    parser.add_argument('--run-name', type=str, required=True, help='Run name for saving model')
    parser.add_argument('--data-file', type=str, default='dhs_all.parquet', help='Path to data file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--repr-dims', type=int, default=320, help='Representation dimensions')
    parser.add_argument('--max-train-length', type=int, default=3000, help='Max training sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--iters', type=int, default=None, help='Number of iterations (overrides epochs)')
    parser.add_argument('--save-every', type=int, default=None, help='Save checkpoint every N epochs/iterations')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='Max threads')
    parser.add_argument('--min-obs-per-user', type=int, default=100, help='Minimum observations per user')
    parser.add_argument('--min-timespan-days', type=int, default=20, help='Minimum timespan in days')
    
    args = parser.parse_args()
    
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    
    # Load data using your function
    imts_args = IMTSConfig()
    data, train_indices, val_indices, labels = load_empirical_imts(
        args=imts_args,
        df=pd.read_parquet(imts_args.data_path),
        timeseries_columns=imts_args.timeseries_columns,
        min_obs_per_user=imts_args.min_seq_len,
        load_from_cache=True,
    )
    
    train_data = data[train_indices]
    
    print('done')
    print(f"Data shape: {train_data.shape}")
    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of validation samples: {len(val_indices)}")
    
    config = dict(
        batch_size=32,
        lr=1e-5,
        output_dims=imts_args.embed_dim,
        hidden_dims = imts_args.embed_dim * 2, 
        depth=16,
        max_train_length=imts_args.max_seq_len
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/pretrain_only__' + args.run_name
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Training model with input_dims={train_data.shape[-1]}")
    print(f"Model will be saved to: {run_dir}")
    
    t = time.time()
    
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    
    loss_log = model.fit(
        train_data,
        n_epochs=50,
        n_iters=args.iters,
        verbose=True
    )
    
    model.save(f'{run_dir}/model.pkl')
    
    t = time.time() - t
    print(f"\nTraining completed in: {datetime.timedelta(seconds=t)}")
    print(f"Final model saved to: {run_dir}/model.pkl")
    print(f"Model is ready for downstream tasks (classification, forecasting, etc.)")
    print("Finished.") 