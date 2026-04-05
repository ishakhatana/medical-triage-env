# server/app.py - FastAPI Server for Medical Triage Environment
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv.core.env_server import create_app
from openenv.core.env_server import create_web_interface_app   # built-in web UI
from models import TriageAction, TriageObservation
from server.environment import MedicalTriageEnvironment

# Always enable web interface (required for HF Spaces App tab)
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

env_instance = MedicalTriageEnvironment()

# create_web_interface_app mounts the web UI at /web AND all API endpoints
app = create_web_interface_app(
    env_instance,
    TriageAction,
    TriageObservation,
)


@app.get("/tasks")
def list_tasks():
    """List available tasks and their action schemas."""
    return {
        "tasks": ["easy", "medium", "hard"],
        "task_descriptions": {
            "easy":   "Triage Prioritization — assign urgency 1/2/3.",
            "medium": "Investigation Ordering — select diagnostic tests (multi-step).",
            "hard":   "Full Discharge Decision — diagnosis, disposition, meds, follow-up.",
        },
        "action_schema": {
            "task_type":               "string: easy | medium | hard",
            "urgency_assignment":      "int 1-3 (easy only)",
            "ordered_investigations":  "list[str] (medium only; pass [] to finish)",
            "diagnosis":               "str (hard only)",
            "disposition":             "str: admit | discharge (hard only)",
            "prescribed_medications":  "list[str] (hard only)",
            "follow_up_days":          "int (hard only)",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "environment": "medical_triage_env"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
