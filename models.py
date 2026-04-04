# models.py - Medical Triage Environment Models
from typing import List, Optional, Dict, Any
from pydantic import Field, BaseModel
from openenv.core.env_server.types import Action, Observation, State

# ============================================================================
# PATIENT DATA MODELS
# ============================================================================

class Vitals(BaseModel):
    """Patient vital signs"""
    heart_rate: int = Field(..., description="Heart rate in bpm")
    blood_pressure: str = Field(..., description="BP as systolic/diastolic")
    spo2: int = Field(..., description="Oxygen saturation %")
    temperature: float = Field(..., description="Temperature in Celsius")
    respiratory_rate: int = Field(..., description="Breaths per minute")

class PatientCase(BaseModel):
    """Complete patient case"""
    patient_id: str
    age: int
    sex: str  # "M" or "F"
    chief_complaint: str
    vitals: Vitals
    history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    
    # Ground truth (hidden from agent)
    true_diagnosis: str
    true_urgency: int  # 1=immediate, 2=urgent, 3=non-urgent
    required_investigations: List[str]
    correct_disposition: str  # "admit" or "discharge"
    safe_medications: List[str]

# ============================================================================
# ACTION MODEL
# ============================================================================

class TriageAction(Action):
    """Actions the agent can take"""
    task_type: str = Field(..., description="easy/medium/hard")
    
    # Task 1: Triage prioritization
    urgency_assignment: Optional[int] = Field(None, description="1-3 urgency level")
    
    # Task 2: Investigation ordering
    ordered_investigations: Optional[List[str]] = Field(None, description="List of test codes")
    
    # Task 3: Discharge decision
    diagnosis: Optional[str] = Field(None, description="Diagnosis string")
    disposition: Optional[str] = Field(None, description="admit/discharge")
    prescribed_medications: Optional[List[str]] = Field(None, description="Medication list")
    follow_up_days: Optional[int] = Field(None, description="Follow-up in days")

# ============================================================================
# OBSERVATION MODEL
# ============================================================================

class TriageObservation(Observation):
    """Observation returned by environment"""
    # done, reward, metadata inherited from Observation base class
    current_patient: Optional[dict] = None  
    available_investigations: list[str] = []
    investigation_results: Optional[dict] = None
    task_instruction: str = ""
    partial_score: float = 0.0

# ============================================================================
# STATE MODEL
# ============================================================================

class TriageState(State):
    """Episode metadata"""
    current_task: str = Field("easy", description="Current task type")
    patients_processed: int = Field(0, description="Number of patients seen")
    total_score: float = Field(0.0, description="Cumulative score")