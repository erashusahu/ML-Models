from datetime import datetime
import pandas as pd
import numpy as np

# from schemas import ICUVitalSign, ICULabResult
from ingestion import ICUEdgeBuffer, patient_level_split
from preprocessing import ICUPreprocessor
from features import ICUFeatureEngineer
from models import Tier1LightGBM
from calibration import TimeAwareCalibrator
from alerting import ICUAlertSystem
from governance import AuditLogger, DriftDetector

class SepsisPredictionPipeline:
    """
    Central orchestration class tying together the entire multi-tier ML architecture
    for Sepsis prediction in the ICU.
    """
    def __init__(self):
        # 1. Ingestion
        self.buffer = ICUEdgeBuffer(buffer_window_minutes=30)
        
        # 2. Preprocessing
        self.preprocessor = ICUPreprocessor(
            vital_cols=['hr', 'sbp', 'dbp', 'map', 'o2sat', 'temp', 'resp_rate'],
            lab_cols=['lactate', 'wbc', 'creatinine', 'bilirubin', 'platelets'],
            static_cols=['age', 'weight'],
            use_decay=True # Primary GRU-D style approach
        )
        
        # 3. Features
        self.feat_engineer = ICUFeatureEngineer(
            continuous_cols=['hr', 'sbp', 'map', 'temp', 'lactate', 'resp_rate', 'age', 'weight']
        )
        
        # 4. Modeling & Calibration
        self.tier1_model = Tier1LightGBM()
        self.calibrator = TimeAwareCalibrator(method='isotonic')
        
        # 5. Alerting & Governance
        self.alert_system = ICUAlertSystem(low_threshold=0.4, med_threshold=0.7, high_threshold=0.85)
        self.logger = AuditLogger()
        self.drift_detector = DriftDetector()
        
    def train(self, df_train: pd.DataFrame, y_train: pd.Series):
        """Historical fit on retrospective data."""
        print("Training full Sepsis Predictor pipeline suite...")
        
        # Establish reference data for data drift
        self.drift_detector.fit_reference(df_train)
        
        # Fit logic
        self.preprocessor.fit(df_train)
        df_processed = self.preprocessor.transform(df_train)
        
        self.feat_engineer.fit(df_processed)
        df_feats = self.feat_engineer.transform(df_processed)
        
        # In production this calls the real model fitting logic
        print("Model and transformers compiled successfully.")

    def mock_stream_inference(self, incoming_mini_batch: pd.DataFrame):
        """Processes a live chunk of streaming data (Kafka/Flink equivalent)."""
        if incoming_mini_batch.empty:
            return
            
        # Layer 1: Edge Buffer (corrects out-of-order networks drops)
        df_buffered = self.buffer.ingest_stream(incoming_mini_batch)
        
        # Layer 2: Preprocess (Missing flags, Decay impute)
        df_processed = self.preprocessor.transform(df_buffered)
        
        # Layer 3: Feature Eng (Rolling EMA, Z-score, Shock Index)
        df_feats = self.feat_engineer.transform(df_processed)
        
        # Filter inference to only the newest snapshot per patient
        latest_snaps = df_feats.sort_values('timestamp').groupby('patient_id').tail(1).copy()
        
        # --- End of stream mock ---
        
        # Layer 4: Generate Risk estimates (Mocking out the raw predictive engine)
        for i, row in latest_snaps.reset_index().iterrows():
            pid = row['patient_id']
            ts = pd.to_datetime(row['timestamp'])
            
            # Simulate a calibrator-adjusted LightGBM probability + Monte Carlo uncertainty bound
            mock_risk = np.random.uniform(0.1, 0.95)
            mock_uncert = np.random.uniform(0.05, 0.4)
            
            # Layer 5: Alert System evaluation
            alert_decision = self.alert_system.process_prediction(
                patient_id=pid,
                timestamp=ts,
                risk_score=mock_risk,
                uncertainty=mock_uncert
            )
            
            # Governance Layer
            self.logger.log_inference(
                patient_id=pid, 
                timestamp=ts, 
                risk_score=alert_decision['risk_score'], 
                action=alert_decision['final_action']
            )
            
            # Emit outcome
            print(f"[ICU MONITOR] Patient: {pid} | Risk: {alert_decision['risk_score']:.2%} | Output Action: {alert_decision['final_action']}")

# Entry point simulation
if __name__ == "__main__":
    import datetime
    
    # Fake a fast multi-patient batch 
    mock_data = pd.DataFrame({
        'patient_id': ['P_01', 'P_02', 'P_03'],
        'timestamp': [datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()],
        'hr': [88, 125, 75], # P_02 has tachycardia (High Risk proxy)
        'sbp': [120, 85, 110], # P_02 is hypotensive
        'lactate': [1.2, 4.5, 0.8], # P_02 shows severe shock indicators
        'age': [45.0, 72.0, 30.0],
        'weight': [70.5, 85.0, 62.0]
    })
    
    pipeline = SepsisPredictionPipeline()
    print("Initiating Mock Streaming Batch...")
    pipeline.mock_stream_inference(mock_data)
