import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class TimeAwareCalibrator:
    """
    Calibration layer that supports Primary (Isotonic) and Backup (Platt).
    Designed to be retrained frequently (e.g., every 30 days) on real-world outcomes
    to prevent Calibration Decay due to hospital protocol changes.
    """
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        if self.method == 'isotonic':
            # Primary: Flexible, data-driven, prevents monotonic probability distortion
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'platt':
            # Backup: Stable, parametric (Logistic Regression), assumes sigmoid curve
            self.calibrator = LogisticRegression()
        else:
            raise ValueError("Method must be 'isotonic' or 'platt'")
            
    def fit(self, uncalibrated_probs: np.ndarray, true_labels: np.ndarray):
        """Fit the recalibrator on a recent window of data (e.g., last 30 days)."""
        X = uncalibrated_probs
        if self.method == 'platt':
            # Needs 2D array for LogisticRegression
            X = X.reshape(-1, 1)
            
        self.calibrator.fit(X, true_labels)
        
    def calibrate(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        """Adjust raw output model scores to represent real-world event probabilities."""
        X = uncalibrated_probs
        if self.method == 'platt':
            X = X.reshape(-1, 1)
            return self.calibrator.predict_proba(X)[:, 1]
        else:
            return self.calibrator.predict(X)
