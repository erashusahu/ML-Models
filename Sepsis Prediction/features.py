import pandas as pd
import numpy as np

def calculate_clinical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates composite clinical scores like Shock Index."""
    df_out = df.copy()
    if 'hr' in df_out.columns and 'sbp' in df_out.columns:
        # Shock Index = HR / SBP
        # Clip SBP to avoid division by zero
        safe_sbp = df_out['sbp'].replace(0, np.nan)
        df_out['shock_index'] = df_out['hr'] / safe_sbp
        
    return df_out

def add_rolling_features(df: pd.DataFrame, columns: list, window_sizes: list = [4, 12], group_col: str = 'patient_id') -> pd.DataFrame:
    """Adds Exponential Moving Average (EMA) and rolling variability (std)."""
    df_out = df.copy()
    
    for col in columns:
        if col not in df_out.columns:
            continue
            
        for w in window_sizes:
            # EMA
            df_out[f"{col}_ema_{w}h"] = df_out.groupby(group_col)[col].transform(lambda x: x.ewm(span=w, adjust=False).mean())
            
            # Variability (rolling std)
            df_out[f"{col}_std_{w}h"] = df_out.groupby(group_col)[col].transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0))
            
    # Explicit MAP variability index
    if 'map' in df_out.columns:
        df_out['map_variability_index'] = df_out.groupby(group_col)['map'].transform(lambda x: x.rolling(window=12, min_periods=1).std().fillna(0))
        
    return df_out

def calculate_lactate_trend(df: pd.DataFrame, group_col: str = 'patient_id') -> pd.DataFrame:
    """Adds Lactate trend and acceleration."""
    df_out = df.copy()
    if 'lactate' in df_out.columns:
        df_out['lactate_delta_1h'] = df_out.groupby(group_col)['lactate'].diff(1).fillna(0)
        df_out['lactate_accel_1h'] = df_out.groupby(group_col)['lactate_delta_1h'].diff(1).fillna(0)
    return df_out

def create_clinical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backup Feature Stream: Converts continuous vitals into clinical ranges.
    XGBoost/LightGBM often perform better on explicitly binned data.
    """
    df_out = df.copy()
    
    # HR Buckets
    if 'hr' in df_out.columns:
        df_out['hr_clinical'] = pd.cut(df_out['hr'], bins=[-np.inf, 60, 100, np.inf], labels=['bradycardia', 'normal', 'tachycardia']).astype(str)
        
    # SBP Buckets
    if 'sbp' in df_out.columns:
        df_out['sbp_clinical'] = pd.cut(df_out['sbp'], bins=[-np.inf, 90, 120, 140, np.inf], labels=['hypotension', 'normal', 'prehypertension', 'hypertension']).astype(str)
        
    return df_out

class ICUFeatureEngineer:
    """
    Dual-Stream Feature Engineering logic: Primary (Z-score) and Backup (Categorical).
    """
    def __init__(self, continuous_cols: list, rolling_cols: list = None):
        self.continuous_cols = continuous_cols
        self.rolling_cols = rolling_cols if rolling_cols is not None else continuous_cols
        self.scaler_means = {}
        self.scaler_stds = {}
        
    def fit(self, df: pd.DataFrame):
        """Fit Z-score normalization params."""
        for col in self.continuous_cols:
            if col in df.columns:
                self.scaler_means[col] = df[col].mean()
                self.scaler_stds[col] = df[col].std()
                
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feats = df.copy()
        
        if df_feats.empty:
            return df_feats
            
        # 1. Derived Composites
        df_feats = calculate_clinical_scores(df_feats)
        df_feats = calculate_lactate_trend(df_feats)
        
        # 2. Rolling & Temporal
        df_feats = add_rolling_features(df_feats, columns=self.rolling_cols)
        
        # 3. Categorical/Clinical ranges (Backup Stream)
        df_feats = create_clinical_ranges(df_feats)
        
        # 4. Continuous Z-score Normalization (Primary Stream)
        for col in self.continuous_cols:
            if col in df_feats.columns:
                mean = self.scaler_means.get(col, df_feats[col].mean())
                std = self.scaler_stds.get(col, df_feats[col].std())
                if pd.isna(mean) or pd.isna(std) or std == 0:
                    df_feats[f"{col}_zscore"] = 0.0
                else:
                    # Clip extreme outliers before normalizing
                    clipped_val = df_feats[col].clip(lower=mean - 5*std, upper=mean + 5*std)
                    df_feats[f"{col}_zscore"] = (clipped_val - mean) / std
                    
        return df_feats
