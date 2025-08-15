# JETS
Joint Embedding Foundation Model for Behavioral Time Series


Create the python virtual environment with the following commands
```
conda create -n jets python=3.9
pip install -r requirements.txt
```

Log into `wandb` using `wandb login`, this allows you to track the loss and other statistics online. 

Place two datasets in this repo: `dhs.parquet` and `dx.parquet`. Csv files will also work.  

`dhs.parquet`: a tall dataframe with a row for each timestamp

`dx.parquet`: a wide dataframe with a row for each user 

In `data\config.py`, specify the time series to use for training and the variables to use for evaluation and change any hyper-parameters if desired. The checkpoints will be saved in its own folder `checkpoints`, 

To train the model, run `trainer.py`, to evaluate the model, run `probe_biomarker.py` or `probe_diagnosis.py`. Specify the checkpoint in the config file first. 

Results will be saved in a new folder `evaluation_results` as csv with timestamps. 