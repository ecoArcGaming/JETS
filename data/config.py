from dataclasses import dataclass

@dataclass
class IMTSConfig:
    """Configuration for IMTS model"""

    embed_dim: int = 256
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    num_layers: int = 8
    predictor_layers: int = 2
    max_seq_len: int = 5000  # 1000
    ema_momentum: float = 0.998
    mask_ratio: float = (
        0.7  # the original masked autoencoder uses larger ratios, need to tune this
    )
    num_masked_tokens: int = mask_ratio * max_seq_len
    num_epochs: int = 50
    batch_size: int = 32
    epoch_total_steps: int = 14400 // batch_size # should be the number of entries in a dataloder 
    min_seq_len: int = 100
    learning_rate: float = 1e-5
    patch_size: int = 32 # for MAE
    timeseries_columns = [  # Heart Rate
        "heartRate_avg",
        "heartRate_stdDev",
        "heartRate_max",
        "heartRate_min",
        # Resting Heart Rate
        "restingHeartRate_avg",
        "restingHeartRate_max",
        "restingHeartRate_min",
        # Blood Oxygen
        "oxygen_avg",
        "oxygen_stdDev",
        "oxygen_max",
        "oxygen_min",
        # VO2 Max
        "vo2Max_avg",
        "vo2Max_stdDev",
        "vo2Max_max",
        "vo2Max_min",
        # Blood Pressure
        "systolic_avg",
        "systolic_stdDev",
        "systolic_max",
        "systolic_min",
        "diastolic_avg",
        "diastolic_stdDev",
        "diastolic_max",
        "diastolic_min",
        # Respiratory
        "breathsMin_avg",
        "breathsMin_stdDev",
        "breathsMin_max",
        "breathsMin_min",
        # Temperature
        "temperature_avg",
        "temperature_max",
        "temperature_min",
        "wristTemperature_avg",
        "wristTemperature_stdDev",
        "wristTemperature_max",
        "wristTemperature_min",
        # Heart Rate Recovery
        "heartRateRecovery_avg",
        "heartRateRecovery_stdDev",
        "heartRateRecovery_max",
        "heartRateRecovery_min",
        # Sleep Metrics (Single Values)
        "sleepOxygen_value",
        "sleepDurationMins_value",
        "sleepOnsetTimeMins_value",
        "remSleepPercent_value",
        "deepSleepPercent_value",
        "remSleepDurationMins_value",
        "deepSleepDurationMins_value",
        "sleepQuality_value",
        "sleepHeartRate_value",
        "sleepHrv_value",
        "breathingDisturbances_value",
        "symptomSeverity_value",
        "remSleepMins_value",
        "deepSleepMins_value",
        # Activity & Exercise
        "steps_total",
        "cardioMins_total",
        "strengthMins_total",
        "workoutTimeMins_total",
        "activityCals_total",
        "highIntensityCardioMins_total",
        # Watch App Latest Values
        "watchApp_mostRecentHRV",
        "watchApp_mostRecentOxygen",
        "watchApp_mostRecentHeartRate",
        # Data Coverage & Scores
        "numHoursDataCoverage",
        "eventSummary_overallScore",
    ]
    num_variables: int = len(timeseries_columns)

    target_columns = [
        "ADHD or ADD",
        "Anxiety",
        "Asthma",
        "Atrial flutter",
        "Autism spectrum disorders",
        "COPD (chronic obstructive pulmonary disease)",
        "Circadian rhythm disorders",
        "Depression",
        "Hypertension",
        "Long covid",
        "ME/CFS",
        "Myocarditis",
        "Osteoporosis",
        "POTS (postural orthostatic tachycardia syndrome)",
        "Sick Sinus Syndrome",
        "Substance abuse",
    ]
    data_path: str = "dhs.parquet"
    binary_data_path: str = "dx.parquet" # also works for biomarker (continous) data
    pretrain: int = 1 # set to 0 for finetuning/eval
    save_dir: str = "checkpoints" # folder to save checkpoints
    load_ckpt_path: str = "checkpoints/model_best.pt" # path to load checkpoint from 

