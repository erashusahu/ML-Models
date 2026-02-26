import numpy as np
import pandas as pd
import datetime

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) to track data drift.
    A PSI > 0.1 indicates moderate drift; PSI > 0.2 indicates significant drift.
    """
    # Defensive checks
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    expected_percents = np.histogram(expected, buckets, density=True)[0]
    expected_percents = expected_percents / np.sum(expected_percents)
    
    actual_percents = np.histogram(actual, buckets, density=True)[0]
    actual_percents = actual_percents / np.sum(actual_percents)

    def sub_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        return (e_perc - a_perc) * np.log(e_perc / a_perc)

    psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents)))
    return psi_value

class DriftDetector:
    """
    Monitors data distributions over time to trigger structural recalibrations 
    if the incoming ICU population changes characteristics or sensors drift.
    """
    def __init__(self, thresholds={"psi_warning": 0.1, "psi_critical": 0.2}):
        self.thresholds = thresholds
        self.reference_distributions = {}

    def fit_reference(self, X_train: pd.DataFrame):
        for col in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                self.reference_distributions[col] = X_train[col].dropna().values

    def detect_drift(self, X_recent: pd.DataFrame) -> dict:
        drift_report = {}
        for col in self.reference_distributions:
            if col in X_recent.columns:
                actual = X_recent[col].dropna().values
                if len(actual) > 10 and len(self.reference_distributions[col]) > 10:
                    psi = calculate_psi(self.reference_distributions[col], actual)
                    status = "STEADY"
                    if psi > self.thresholds["psi_critical"]:
                        status = "CRITICAL_DRIFT_THRESHOLD_EXCEEDED"
                    elif psi > self.thresholds["psi_warning"]:
                        status = "DRIFT_WARNING"
                    drift_report[col] = {"psi": round(psi, 3), "status": status}
        return drift_report

class AuditLogger:
    """
    Logs all inferences and clinical overrides for hospital compliance and auditing.
    """
    def __init__(self, log_file: str = "icu_audit_trail.log"):
        self.log_file = log_file

    def log_inference(self, patient_id: str, timestamp: datetime.datetime, risk_score: float, action: str):
        """Standard automated log."""
        entry = f"[{timestamp.isoformat()}] | TYPE: inference | PATIENT: {patient_id} | RISK: {risk_score:.3f} | ACTION: {action}\n"
        # In practice: Write to secure DB
        pass

    def log_override(self, override_data: dict):
        """Specialized log for Human-in-the-loop interventions."""
        entry = (f"[{override_data['timestamp'].isoformat()}] | TYPE: override | PATIENT: {override_data['patient_id']} | "
                 f"CLINICIAN_ID: {override_data['clinician_id']} | NEW_TIER: {override_data['new_risk_tier']} | "
                 f"REASON: {override_data['override_reason']}\n")
        # In practice: Write to secure DB 
        pass
