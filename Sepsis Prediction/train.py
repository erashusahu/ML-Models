import pandas as pd
import numpy as np

# Import our custom modules
from ingestion import patient_level_split
from pipeline import SepsisPredictionPipeline
from evaluation import evaluate_icu_model

def train_sepsis_model(data_path: str):
    """
    Orchestrates the training of the Tier 1 LightGBM model using the synthetic dataset.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Patient-Level Split (Crucial for medical data to prevent data leakage)
    # 80% Train, 20% Test
    train_df, test_df = patient_level_split(df, test_size=0.2, random_state=42, stratify_col='sepsis_label')
    
    print("\nInitializing Pipeline Components...")
    pipeline = SepsisPredictionPipeline()
    
    # Extract labels for the training set
    y_train = train_df['sepsis_label']
    y_test = test_df['sepsis_label']
    
    # 2. Fit and Transform Training Data
    # The pipeline class currently has a mock train() method, so we will manually orchestrate
    # the training using the components initialized inside the pipeline.
    print("Preprocessing Training Data...")
    pipeline.preprocessor.fit(train_df)
    train_processed = pipeline.preprocessor.transform(train_df)
    
    print("Engineering Features for Training Data...")
    pipeline.feat_engineer.fit(train_processed)
    train_features = pipeline.feat_engineer.transform(train_processed)
    
    # LightGBM cannot handle datetime or string columns directly, drop them for modeling
    drop_cols = ['patient_id', 'timestamp', 'sepsis_label', 'hr_clinical', 'sbp_clinical']
    X_train = train_features.drop(columns=[c for c in drop_cols if c in train_features.columns])
    
    # 3. Train the Model
    print("Training Tier 1 LightGBM...")
    pipeline.tier1_model.fit(X_train, y_train)
    
    # 4. Process Test Data and Evaluate
    print("\nProcessing Test Data...")
    test_processed = pipeline.preprocessor.transform(test_df)
    test_features = pipeline.feat_engineer.transform(test_processed)
    
    X_test = test_features.drop(columns=[c for c in drop_cols if c in test_features.columns])
    
    print("Evaluating Model on Test Data...")
    # Predict probabilities (using the median model)
    y_prob = pipeline.tier1_model.model_median.predict_proba(X_test)[:, 1]
    
    # Simulate the Alerting mechanism (High threshold triggers alert)
    alerts_triggered = (y_prob >= pipeline.alert_system.points['high']).astype(int)
    
    # Calculate patient days for alarm fatigue metric
    # Total distinct timestamps in test set (assuming 1 timestamp = roughly some fraction of a day)
    # For our synthetic data: 4 measurements = 1 day
    total_patient_days = len(X_test) / 4.0 
    
    metrics = evaluate_icu_model(
        y_true=y_test.values, 
        y_prob=y_prob, 
        alerts_triggered=alerts_triggered, 
        total_patient_days=total_patient_days
    )
    
    print("\n==================================")
    print("FINAL TEST SET METRICS")
    print("==================================")
    print(f"AUROC (Discrimination): {metrics['auroc']:.4f}")
    print(f"AUPRC (Precision-Recall): {metrics['auprc']:.4f}")
    print(f"Sensitivity (Recall): {metrics['sensitivity']:.2%}")
    print(f"Precision (PPV): {metrics['precision']:.2%}")
    print(f"False Alarms per Patient Day: {metrics['false_alarms_per_patient_day']:.2f}")
    print("==================================")
    
    # Note on calibration
    print("\nNote: The scores above are raw. The TimeAwareCalibrator is initialized in the pipeline")
    print("but would typically be fitted post-training on a separated validation set.")
    
if __name__ == "__main__":
    train_sepsis_model("synthetic_sepsis_data.csv")
