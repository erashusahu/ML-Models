from pydantic import BaseModel, Field
from typing import Optional
import datetime

class ICUVitalSign(BaseModel):
    """Schema for incoming vital signs from bedside monitors."""
    patient_id: str
    timestamp: datetime.datetime
    hr: Optional[float] = Field(None, description="Heart Rate (bpm)")
    sbp: Optional[float] = Field(None, description="Systolic Blood Pressure (mmHg)")
    dbp: Optional[float] = Field(None, description="Diastolic Blood Pressure (mmHg)")
    map: Optional[float] = Field(None, description="Mean Arterial Pressure (mmHg)")
    o2sat: Optional[float] = Field(None, description="Oxygen Saturation (%)")
    temp: Optional[float] = Field(None, description="Temperature (C)")
    resp_rate: Optional[float] = Field(None, description="Respiratory Rate (breaths/min)")

class ICULabResult(BaseModel):
    """Schema for incoming laboratory results."""
    patient_id: str
    timestamp: datetime.datetime
    lactate: Optional[float] = Field(None, description="Lactate level (mmol/L)")
    wbc: Optional[float] = Field(None, description="White blood cell count (10^9/L)")
    creatinine: Optional[float] = Field(None, description="Creatinine level (mg/dL)")
    bilirubin: Optional[float] = Field(None, description="Bilirubin level (mg/dL)")
    platelets: Optional[float] = Field(None, description="Platelet count (10^9/L)")

class MonitorHeartbeat(BaseModel):
    """Schema to track monitor connection status and detect network drops."""
    monitor_id: str
    timestamp: datetime.datetime
    status: str = Field(..., description="e.g., 'active', 'disconnected', 'degraded'")

class ClinicalOverride(BaseModel):
    """Schema for Human-in-the-loop override logging."""
    patient_id: str
    timestamp: datetime.datetime
    clinician_id: str
    override_reason: str
    new_risk_tier: str = Field(..., description="Risk tier set by clinician override")
