from dataclasses import dataclass

@dataclass
class IMTSConfig:
    """Configuration for IMTS model"""

    embed_dim: int = 256
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    num_layers: int = 8
    predictor_layers: int = 4
    max_seq_len: int = 5000  # 1000
    num_variables: int = 63
    ema_momentum: float = 0.998
    mask_ratio: float = (
        0.7  # the original masked autoencoder uses larger ratios, need to tune this
    )
    num_masked_tokens: int = mask_ratio * max_seq_len
    num_epochs: int = 50
    batch_size: int = 32
    epoch_total_steps: int = 14400 // batch_size
    min_seq_len: int = 100
    learning_rate: float = 1e-5
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
    binary_columns = [
        "ADHD or ADD",
        "Alzheimer's",
        "Anxiety",
        "Arthritis",
        "Asthma",
        "Atrial fibrillation",
        "Atrial flutter",
        "Autism spectrum disorders",
        "Back pain",
        "Bipolar",
        "Brain cancer",
        "Breast cancer",
        "COPD",
        "COPD (chronic obstructive pulmonary disease)",
        "Chronic kidney disease",
        "Circadian rhythm disorders",
        "Colon cancer",
        "Depression",
        "Diabetes",
        "Endometrial cancer",
        "Fatty liver disease",
        "HIV/AIDS",
        "Heart failure",
        "Hepatitis",
        "High cholesterol",
        "Hypertension",
        "Hypertension (high blood pressure)",
        "Insomnia",
        "Kidney cancer",
        "Leukemia (cancer)",
        "Liver cancer",
        "Long covid",
        "Lung cancer",
        "ME/CFS",
        "Melanoma (skin cancer)",
        "Myocarditis",
        "Osteoporosis",
        "Other",
        "Other cancer",
        "Other heart conditions",
        "Other lung disease",
        "Other medical conditions",
        "Other mental health conditions",
        "Other skin cancers",
        "Other sleep conditions",
        "POTS",
        "POTS (postural orthostatic tachycardia syndrome)",
        "Pancreatic cancer",
        "Parkinson's",
        "Previous stroke",
        "Prostate cancer",
        "Pulmonary fibrosis",
        "Restless leg syndrome",
        "SVT",
        "SVT (Supraventricular tachycardia)",
        "Schizophrenia",
        "Sick Sinus Syndrome",
        "Sleep apnea",
        "Substance abuse",
        "Thyroid cancer",
        "Ventricular Arrhythmias",
        "WPW (Wolff-Parkinson-White Syndrome)",
        "other medical conditions",
    ]
    target_column: str = "Diabetes"
    data_path: str = "dhs.parquet"
    binary_data_path: str = "dx.parquet"
    pretrain: int = 1
    save_dir: str = "checkpoints"
    load_ckpt_path: str = "checkpoints/model_best.pt"
    patch_size = 16
