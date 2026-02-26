import pandas as pd
import datetime
import shap

class ICUAlertSystem:
    """
    Multi-level alerting system with cooldowns to prevent alarm fatigue.
    Includes uncertainty-aware escalation overrides.
    """
    def __init__(self, low_threshold=0.4, med_threshold=0.7, high_threshold=0.85, cooldown_hours=2.0):
        self.points = {
            'low': low_threshold,
            'med': med_threshold,
            'high': high_threshold
        }
        self.cooldown = pd.Timedelta(hours=cooldown_hours)
        self.last_alert_time = {} # dict mapping patient_id -> timestamp (of last fired alert)
        
    def determine_tier(self, risk_score: float) -> str:
        if risk_score >= self.points['high']:
            return "high" # Trigger audible alert
        elif risk_score >= self.points['med']:
            return "medium" # Trigger nurse dashboard icon
        elif risk_score >= self.points['low']:
            return "low" # Silent monitor
        return "none" # Sub-clinical

    def process_prediction(self, patient_id: str, timestamp: datetime.datetime, risk_score: float, uncertainty: float) -> dict:
        """Processes a single prediction line and dictates the system UI action."""
        tier = self.determine_tier(risk_score)
        
        # Hard Rule Fallback: If uncertainty is massive in a borderline/high situation, request lab draw
        if uncertainty > 0.30 and tier in ["medium", "high"]:
            action = "MANUAL_LAB_VERIFICATION_REQUESTED"
        else:
            action = tier
            
        # Cooldown Logic: Prevent rapid re-firing of alerts for the same patient in 2 hours
        if action in ["medium", "high", "MANUAL_LAB_VERIFICATION_REQUESTED"]:
            last_time = self.last_alert_time.get(patient_id)
            if last_time is not None:
                if (timestamp - last_time) < self.cooldown:
                    # Suppress alert due to cooldown (alarm fatigue protection)
                    action = "suppressed_by_cooldown"
                else:
                    self.last_alert_time[patient_id] = timestamp # Update last alert
            else:
                self.last_alert_time[patient_id] = timestamp # First time alert

        return {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "risk_score": risk_score,
            "uncertainty": uncertainty,
            "raw_tier": tier,
            "final_action": action
        }

def get_shap_explanation(model, features: pd.DataFrame):
    """
    SHAP explanation stub.
    Outputs the top contributing features to explain exactly WHY the risk score is high,
    addressing the "Black Box" problem.
    """
    # Assuming model is the Tree-based Tier 1 model (LightGBM)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Code to map shap values to top features would go here
    # E.g., returning a dict of Top 3 features and their contribution direction
    return {
        "status": "SHAP extracted",
        "sample_contribution": list(shap_values[0][:3]) if len(shap_values) > 0 else []
    }
