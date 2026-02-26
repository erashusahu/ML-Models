import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
from typing import Dict

def evaluate_icu_model(y_true: np.ndarray, y_prob: np.ndarray, alerts_triggered: np.ndarray, total_patient_days: float) -> Dict[str, float]:
    """
    Evaluates model performance using standard and critical ICU metrics.
    Targets:
    - AUROC: 0.85 - 0.92
    - AUPRC: 0.45 - 0.65
    - Sensitivity >= 85%
    - False alarms < 2/day/patient
    """
    metrics = {}
    
    # Core Classification ML Metrics
    metrics['auroc'] = roc_auc_score(y_true, y_prob)
    metrics['auprc'] = average_precision_score(y_true, y_prob)
    
    # Clinical Metrics (based on alert thresholding)
    metrics['sensitivity'] = recall_score(y_true, alerts_triggered)
    # Adding specificity is standard even if not primarily targeted
    metrics['precision'] = precision_score(y_true, alerts_triggered) 
    
    # ICU Specific: Alarm Fatigue
    # False positives occur when alert == 1 but true == 0
    false_positives = np.sum((alerts_triggered == 1) & (y_true == 0))
    metrics['false_alarms_per_patient_day'] = false_positives / total_patient_days if total_patient_days > 0 else 0.0
    
    return metrics

def calculate_lead_time(predictions_df: pd.DataFrame, time_col: str='timestamp', event_time_col: str='sepsis_onset_time') -> pd.Timedelta:
    """
    Calculates median time-to-detection (lead time) before actual clinical diagnosis.
    predictions_df should be pre-filtered to true positive alerts only.
    Target: 4-6 hours median lead time.
    """
    if predictions_df.empty:
        return pd.Timedelta(0)
        
    lead_times = pd.to_datetime(predictions_df[event_time_col]) - pd.to_datetime(predictions_df[time_col])
    
    # Filter to only valid lead times (events caught before they happen or at the exact time)
    valid_lead_times = lead_times[lead_times > pd.Timedelta(seconds=0)]
    
    if valid_lead_times.empty:
        return pd.Timedelta(0)
        
    median_lead_time = valid_lead_times.median()
    return median_lead_time
