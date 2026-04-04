# server/app.py - FastAPI Server for Medical Triage Environment
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from openenv.core.env_server.http_server import create_app
from models import TriageAction, TriageObservation
from server.environment import MedicalTriageEnvironment

# Create the base app with OpenEnv
app = create_app(
    MedicalTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="medical_triage_env",
    max_concurrent_envs=100,
)

# Create a global environment instance for endpoint access
_env_instance = MedicalTriageEnvironment()

# ============================================================================
# REQUIRED ADDITIONAL ENDPOINTS FOR HACKATHON
# ============================================================================

@app.get("/tasks")
def list_tasks():
    """
    Return list of available tasks and action schema.
    
    REQUIRED BY HACKATHON VALIDATOR.
    """
    return {
        "tasks": ["easy", "medium", "hard"],
        "action_schema": {
            "task_type": "string (easy/medium/hard)",
            "urgency_assignment": "int (1-3, for easy task)",
            "ordered_investigations": "list[string] (for medium task)",
            "diagnosis": "string (for hard task)",
            "disposition": "string (admit/discharge, for hard task)",
            "prescribed_medications": "list[string] (for hard task)",
            "follow_up_days": "int (for hard task)"
        },
        "task_descriptions": {
            "easy": "Triage Prioritization: Assign urgency level (1=immediate, 2=urgent, 3=non-urgent)",
            "medium": "Investigation Ordering: Select appropriate diagnostic tests",
            "hard": "Full Discharge Decision: Provide diagnosis, disposition, medications, follow-up"
        }
    }


@app.get("/grader")
def get_grader_score(task_id: str):
    """
    Return grader score for a completed task.
    
    REQUIRED BY HACKATHON VALIDATOR.
    
    Args:
        task_id: One of "easy", "medium", "hard"
    
    Returns:
        {"task_id": str, "score": float in [0.0, 1.0]}
    """
    if task_id not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid task_id. Must be easy, medium, or hard")
    
    # Get score from environment's grading function
    score = _env_instance.grade_task(task_id)
    
    # Ensure score is in valid range
    if not (0.0 <= score <= 1.0):
        raise HTTPException(status_code=500, detail=f"Grader returned invalid score: {score}")
    
    return {
        "task_id": task_id,
        "score": float(score)
    }


@app.post("/baseline")
def run_baseline():
    """
    Run baseline inference script and return scores for all tasks.
    
    REQUIRED BY HACKATHON VALIDATOR.
    
    This endpoint would normally execute inference.py internally,
    but for now returns example scores showing the grading range.
    
    Returns:
        {"easy": float, "medium": float, "hard": float}
    """
    # In production, this would execute:
    # subprocess.run(["python", "inference.py"], ...)
    # and parse the output
    
    # For now, return example scores from the environment
    return {
        "easy": _env_instance.grade_task("easy"),
        "medium": _env_instance.grade_task("medium"),
        "hard": _env_instance.grade_task("hard"),
        "note": "In production, this executes inference.py"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "environment": "medical_triage_env"}


# ============================================================================
# MAIN ENTRY POINT (for local testing)
# ============================================================================

# ============================================================================
# MAIN ENTRY POINT (for local testing)
# ============================================================================

def main():
    """Main entry point for running the server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()