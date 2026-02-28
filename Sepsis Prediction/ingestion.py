import pandas as pd
import numpy as np
from typing import Tuple

def patient_level_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, stratify_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset at the patient level to prevent "identity leakage".
    All rows for a single patient will go entirely to either Train or Test.
    Optionally stratifies based on a target column (e.g., if any of the patient's
    rows contain a positive label).
    """
    if 'patient_id' not in df.columns:
        raise ValueError("DataFrame must contain a 'patient_id' column.")
    
    unique_patients = df['patient_id'].unique()
    
    if stratify_col and stratify_col in df.columns:
        from sklearn.model_selection import train_test_split
        # Get max label per patient (if any row is 1, patient is 1)
        patient_labels = df.groupby('patient_id')[stratify_col].max()
        
        train_patients, test_patients = train_test_split(
            patient_labels.index, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=patient_labels.values
        )
    else:
        np.random.seed(random_state)
        # Using a copy prevents the Deprecation Warning about shuffling unique() output
        unique_patients_list = list(unique_patients)
        np.random.shuffle(unique_patients_list)
        
        split_idx = int(len(unique_patients_list) * (1 - test_size))
        train_patients = unique_patients_list[:split_idx]
        test_patients = unique_patients_list[split_idx:]
    
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()
    
    print(f"Patient Level Split - Train: {len(train_patients)} patients, Test: {len(test_patients)} patients.")
    
    return train_df, test_df

class ICUEdgeBuffer:
    """
    Local edge buffer on ICU server (last X min cache).
    Maintains robustness against network drops and handles out-of-order data correction.
    """
    def __init__(self, buffer_window_minutes: int = 30):
        self.buffer_window = pd.Timedelta(minutes=buffer_window_minutes)
        self.buffer = pd.DataFrame()

    def ingest_stream(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Ingests a mini-batch of streaming data.
        Corrects out-of-order timestamps and evicts old data seamlessly.
        """
        if new_data.empty:
            return self.buffer

        # Ensure timestamp is datetime
        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        
        # Append
        self.buffer = pd.concat([self.buffer, new_data], ignore_index=True)
        
        # Out-of-order correction logic (sort by patient and time)
        self.buffer = self.buffer.sort_values(by=['patient_id', 'timestamp']).reset_index(drop=True)
        
        # Evict data older than the buffer window based on the absolute latest timestamp in the buffer
        if not self.buffer.empty:
            latest_time = self.buffer['timestamp'].max()
            cutoff_time = latest_time - self.buffer_window
            self.buffer = self.buffer[self.buffer['timestamp'] >= cutoff_time].copy()
            
        return self.buffer

def sql_polling_backup(db_connection_mock, query: str, interval_minutes: int = 15) -> pd.DataFrame:
    """
    Backup data ingestion method. Polls the SQL database every X minutes.
    Used if the primary streaming layer (Kafka/Flink) drops.
    """
    print(f"Executing SQL Polling Backup fallback... INTERVAL={interval_minutes}m")
    # In a real system: return pd.read_sql(query, db_connection_mock)
    # Returning empty DataFrame as mock stub
    return pd.DataFrame()
