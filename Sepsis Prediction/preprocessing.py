import pandas as pd
import numpy as np
import warnings

def add_missingness_indicators(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Adds boolean flags indicating whether a value was originally missing."""
    df_out = df.copy()
    for col in columns:
        if col in df_out.columns:
            df_out[f"{col}_is_missing"] = df_out[col].isna().astype(int)
    return df_out

def forward_fill_impute(df: pd.DataFrame, columns: list, group_col: str = 'patient_id') -> pd.DataFrame:
    """
    Backup Imputation: Group by patient and forward fill.
    If still missing (e.g., before first draw), fill with column median.
    """
    df_out = df.copy()
    
    # Forward fill within patient
    present_cols = [c for c in columns if c in df_out.columns]
    if present_cols:
        df_out[present_cols] = df_out.groupby(group_col)[present_cols].ffill()
    
    # Fill remaining NaNs with column median
    for col in columns:
        if col in df_out.columns and df_out[col].isna().any():
            median_val = df_out[col].median()
            # If the entire column is NaN, fill with 0 to prevent downstream failure
            if pd.isna(median_val):
                median_val = 0.0
            df_out[col] = df_out[col].fillna(median_val)
            
    return df_out

def add_time_since_last_measurement(df: pd.DataFrame, columns: list, group_col: str = 'patient_id', time_col: str = 'timestamp') -> pd.DataFrame:
    """
    Adds features indicating the time elapsed since the last recorded measurement.
    Crucial ICU feature since order frequency is a clinical signal.
    """
    df_out = df.copy()
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df_out[time_col]):
        df_out[time_col] = pd.to_datetime(df_out[time_col])
    
    for col in columns:
        if col not in df_out.columns:
            continue
            
        # Create a series of timestamps where the value is NOT missing
        meas_times = df_out[time_col].copy()
        meas_times[df_out[col].isna()] = pd.NaT
        
        # Forward fill the measurement timestamps per patient
        last_measured_time = df_out.groupby(group_col)[meas_times.name].ffill()
        
        # Calculate time difference in hours
        time_diff = (df_out[time_col] - last_measured_time).dt.total_seconds() / 3600.0
        
        # Fill missing time diffs (no prior measurement) with a large proxy value (e.g., 999 hours)
        df_out[f"{col}_time_since_last_hr"] = time_diff.fillna(999.0)
        
    return df_out

def add_decay_imputation(df: pd.DataFrame, columns: list, group_col: str = 'patient_id', time_col: str = 'timestamp') -> pd.DataFrame:
    """
    Primary Imputation: GRU-D style decay proxy for tabular data.
    Fades the last known value towards the population mean exponentially over time.
    """
    df_out = df.copy()
    
    for col in columns:
        time_diff_col = f"{col}_time_since_last_hr"
        if time_diff_col not in df_out.columns or col not in df_out.columns:
            continue
            
        pop_mean = df_out[col].mean()
        if pd.isna(pop_mean):
            pop_mean = 0.0
            
        # Time constant for decay (e.g., values return to mean after ~24-48 hours)
        decay_constant_hours = 24.0 
        
        # Decay weight: closer to 1 if recent, closer to 0 if long time ago
        decay_weight = np.exp(-df_out[time_diff_col] / decay_constant_hours)
        
        # Forward fill to get the last known baseline (ignoring NaNs for now)
        last_known_val = df_out.groupby(group_col)[col].ffill()
        last_known_val = last_known_val.fillna(pop_mean) # fallback if nan at start
        
        # Calculate decayed value
        decayed_val = (last_known_val * decay_weight) + (pop_mean * (1 - decay_weight))
        
        # Only overwrite actual NaNs in the original column
        missing_mask = df_out[col].isna()
        df_out.loc[missing_mask, col] = decayed_val.loc[missing_mask]
        
    return df_out

class ICUPreprocessor:
    """
    Robust Preprocessing Layer handling multi-strategy imputation and temporal logic.
    """
    def __init__(self, vital_cols: list, lab_cols: list, use_decay: bool = False):
        self.vital_cols = vital_cols
        self.lab_cols = lab_cols
        self.use_decay = use_decay
        self.all_cols = self.vital_cols + self.lab_cols
        
    def fit(self, df: pd.DataFrame):
        """Fit method to store population medians/means for production deployment (TODO)."""
        pass
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        if df_processed.empty:
            return df_processed
            
        # 1. Add missingness indicators (Critical signal)
        df_processed = add_missingness_indicators(df_processed, self.all_cols)
        
        # 2. Add time-since-last-measurement (Frequency = clinical suspicion)
        df_processed = add_time_since_last_measurement(df_processed, self.all_cols)
        
        # 3. Impute
        if self.use_decay:
            # Primary method: Decay-based impute
            df_processed = add_decay_imputation(df_processed, self.all_cols)
            # Catch any remaining NaNs at the beginning of records with ffill/median
            df_processed = forward_fill_impute(df_processed, self.all_cols)
        else:
            # Backup method: Forward fill
            df_processed = forward_fill_impute(df_processed, self.all_cols)
            
        return df_processed
