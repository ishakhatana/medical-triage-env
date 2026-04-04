# server/environment.py - Medical Triage Environment Implementation
import random
from uuid import uuid4
from typing import Optional, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import TriageAction, TriageObservation, PatientCase
from patient_cases import PATIENT_CASES, get_cases_by_urgency, get_case_by_id


class MedicalTriageEnvironment(Environment):
    """
    Medical Triage & Discharge Planning Environment
    
    Three tasks of increasing difficulty:
    - Easy: Triage prioritization (sort patients by urgency)
    - Medium: Investigation ordering (select appropriate tests)
    - Hard: Full discharge decision (diagnosis, meds, disposition)
    """
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    # Available investigations
    AVAILABLE_TESTS = [
        "ecg", "troponin", "cbc", "cxr", "ct_head", "ct_abdomen",
        "ultrasound", "urinalysis", "blood_culture", "lactate",
        "bnp", "inr", "electrolytes", "rapid_strep", "xray_ankle",
        "xray_leg", "blood_glucose", "compartment_pressure",
        "urine_culture", "bhcg", "lumbar_puncture", "endoscopy"
    ]
    
    # CLASS-LEVEL storage for task scores (shared across instances)
    _global_task_scores = {
        "easy": 0.0,
        "medium": 0.0,
        "hard": 0.0
    }
    
    def __init__(self):
        """Initialize the environment"""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = "easy"
        self._current_patient = None
        self._patients_for_triage = []
        self._score = 0.0
        self._done = False
        
        # Store scores per task (instance-level)
        self._task_scores = {
            "easy": 0.0,
            "medium": 0.0,
            "hard": 0.0
        }
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> TriageObservation:
        """Reset environment and return initial observation"""
        # Select random patient
        if seed is not None:
            random.seed(seed)
        
        self._current_patient = random.choice(PATIENT_CASES)
        
        # Get task instruction (default to easy for initial observation)
        task_instruction = self._get_task_instruction()
        
        # Convert patient to dict for observation
        patient_dict = {
            "id": self._current_patient.patient_id,
            "age": self._current_patient.age,
            "sex": self._current_patient.sex,
            "chief_complaint": self._current_patient.chief_complaint,
            "heart_rate": self._current_patient.vitals.heart_rate,
            "blood_pressure": self._current_patient.vitals.blood_pressure,
            "spo2": self._current_patient.vitals.spo2,
            "temperature": self._current_patient.vitals.temperature,
            "respiratory_rate": self._current_patient.vitals.respiratory_rate,
            "past_medical_history": self._current_patient.history,
            "allergies": self._current_patient.allergies,
        }
        
        return TriageObservation(
            done=False,
            reward=None,
            current_patient=patient_dict,
            available_investigations=self.AVAILABLE_TESTS,
            task_instruction=task_instruction,
            partial_score=0.0,
        )    

    
    def step(self, action: TriageAction) -> TriageObservation:

        """Execute action and return observation"""  # FIX: Remove extra space before """
        self._state.step_count += 1
        self._current_task = action.task_type
        
        # Route to appropriate task handler
        if action.task_type == "easy":
            reward, done = self._handle_easy_task(action)
        elif action.task_type == "medium":
            reward, done = self._handle_medium_task(action)
        elif action.task_type == "hard":
            reward, done = self._handle_hard_task(action)
        else:
            reward, done = 0.0, True
        
        self._score += reward
        self._done = done
        
        # Store the score for this task (both instance and class level)
        if action.task_type in self._task_scores:
            self._task_scores[action.task_type] = reward
            # Also store in class-level variable for grader access
            MedicalTriageEnvironment._global_task_scores[action.task_type] = reward
        
        # Convert current patient to dict (if exists)
        patient_dict = None
        if self._current_patient:
            patient_dict = {
                "id": self._current_patient.patient_id,
                "age": self._current_patient.age,
                "sex": self._current_patient.sex,
                "chief_complaint": self._current_patient.chief_complaint,
                "heart_rate": self._current_patient.vitals.heart_rate,
                "blood_pressure": self._current_patient.vitals.blood_pressure,
                "spo2": self._current_patient.vitals.spo2,
                "temperature": self._current_patient.vitals.temperature,
                "respiratory_rate": self._current_patient.vitals.respiratory_rate,
                "past_medical_history": self._current_patient.history,
                "allergies": self._current_patient.allergies,
            }
            
        return TriageObservation(
            done=done,
            reward=reward,
            current_patient=patient_dict,
            available_investigations=self.AVAILABLE_TESTS,
            task_instruction=self._get_task_instruction(),
            partial_score=self._score
        )
    
     
    
    
    def _handle_easy_task(self, action: TriageAction) -> tuple[float, bool]:
        """Task 1: Triage Prioritization"""
        if action.urgency_assignment is None:
            return 0.0, True
        
        # Simple: assign urgency to ONE patient
        # In real scenario, agent would process all 5
        # For simplicity, we test on first patient
        if not self._current_patient and self._patients_for_triage:
            self._current_patient = self._patients_for_triage[0]
        
        if self._current_patient:
            correct_urgency = self._current_patient.true_urgency
            assigned_urgency = action.urgency_assignment
            
            # Perfect match = 1.0, off by 1 = 0.5, off by 2 = 0.0
            if assigned_urgency == correct_urgency:
                reward = 1.0
            elif abs(assigned_urgency - correct_urgency) == 1:
                reward = 0.5
            else:
                reward = 0.0
            
            return reward, True
        
        return 0.0, True
    
    def _handle_medium_task(self, action: TriageAction) -> tuple[float, bool]:
        """Task 2: Investigation Ordering"""
        if action.ordered_investigations is None:
            return 0.0, True
        
        # Use FIXED patient P001 (chest pain patient) for consistency
        # ALWAYS use P001 for consistency between LLM prompt and grader
        self._current_patient = get_case_by_id("P001")
        
        required = set(self._current_patient.required_investigations)
        ordered = set(action.ordered_investigations)
        
        # Calculate precision and recall
        if len(ordered) == 0:
            return 0.0, True
        
        true_positives = len(required & ordered)
        precision = true_positives / len(ordered) if len(ordered) > 0 else 0.0
        recall = true_positives / len(required) if len(required) > 0 else 0.0
        
        # F1 score as reward
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Penalty for over-testing (ordering unnecessary expensive tests)
        unnecessary_tests = len(ordered - required)
        penalty = unnecessary_tests * 0.1
        
        reward = max(0.0, f1_score - penalty)
        return reward, True
    
    def _handle_hard_task(self, action: TriageAction) -> tuple[float, bool]:
        """Task 3: Full Discharge Decision"""
        if not self._current_patient:
            # Pick a random patient for full workup
            self._current_patient = random.choice(PATIENT_CASES)
        
        total_reward = 0.0
        
        # 1. Diagnosis accuracy (0.3 weight)
        if action.diagnosis:
            if action.diagnosis.lower() == self._current_patient.true_diagnosis.lower():
                total_reward += 0.3
            elif action.diagnosis.lower() in self._current_patient.true_diagnosis.lower():
                total_reward += 0.15  # Partial credit
        
        # 2. Disposition correctness (0.3 weight)
        if action.disposition:
            if action.disposition.lower() == self._current_patient.correct_disposition.lower():
                total_reward += 0.3
        
        # 3. Medication safety (0.2 weight)
        if action.prescribed_medications:
            safe_meds = set(self._current_patient.safe_medications)
            prescribed = set(action.prescribed_medications)
            
            # Check for safe medications
            if prescribed & safe_meds:
                med_score = len(prescribed & safe_meds) / len(safe_meds)
                total_reward += 0.2 * med_score
        
        # 4. Safety violation penalty (-0.5)
        # Never discharge someone with urgency 1 (immediate)
        if (action.disposition == "discharge" and 
            self._current_patient.true_urgency == 1):
            total_reward -= 0.5
        
        # 5. Follow-up appropriateness (0.2 weight)
        if action.follow_up_days:
            if action.disposition == "discharge":
                if 1 <= action.follow_up_days <= 7:
                    total_reward += 0.2
        
        # Clamp to [0.0, 1.0]
        final_reward = max(0.0, min(1.0, total_reward))
        
        return final_reward, True
    
    def _get_task_instruction(self) -> str:
        """Get instruction text for current task"""
        if self._current_task == "easy":
            return (
                "Task 1 - Triage Prioritization:\n"
                "Given a patient case, assign urgency level:\n"
                "  1 = Immediate (life-threatening)\n"
                "  2 = Urgent (needs prompt care)\n"
                "  3 = Non-urgent (stable)\n"
                "Set 'urgency_assignment' to 1, 2, or 3."
            )
        elif self._current_task == "medium":
            return (
                "Task 2 - Investigation Ordering:\n"
                "Select appropriate diagnostic tests from available list.\n"
                "Avoid over-testing (unnecessary tests have cost penalties).\n"
                "Set 'ordered_investigations' to list of test codes."
            )
        elif self._current_task == "hard":
            return (
                "Task 3 - Full Discharge Decision:\n"
                "Provide complete discharge plan:\n"
                "  - diagnosis: your diagnosis string\n"
                "  - disposition: 'admit' or 'discharge'\n"
                "  - prescribed_medications: list of medication names\n"
                "  - follow_up_days: days until follow-up (if discharge)\n"
                "WARNING: Never discharge a patient with critical vitals!"
            )
        return "Invalid task"
    
    @property
    def state(self) -> State:
        """Get current state"""
        return self._state
    
    # ========================================================================
    # GRADING FUNCTIONS (REQUIRED FOR HACKATHON)
    # ========================================================================
    
    def grade_task(self, task_id: str) -> float:
        """
        Grade a completed task. Returns score in [0.0, 1.0]
        
        This is called by the /grader endpoint.
        """
        if task_id == "easy":
            return self._grade_easy()
        elif task_id == "medium":
            return self._grade_medium()
        elif task_id == "hard":
            return self._grade_hard()
        else:
            return 0.0
    
    def _grade_easy(self) -> float:
        """Grade triage prioritization task"""
        return MedicalTriageEnvironment._global_task_scores.get("easy", 0.0)

    def _grade_medium(self) -> float:
        """Grade investigation ordering task"""
        return MedicalTriageEnvironment._global_task_scores.get("medium", 0.0)

    def _grade_hard(self) -> float:
        """Grade discharge decision task"""
        return MedicalTriageEnvironment._global_task_scores.get("hard", 0.0)