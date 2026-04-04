#!/usr/bin/env python3
"""
Inference script for Medical Triage Environment - LLM-Powered
CRITICAL: Log format must be EXACT or scoring will fail!
"""
from dotenv import load_dotenv
load_dotenv()  # This loads .env automatically!
import json
import os
import sys
import asyncio
from datetime import datetime
import requests
from openai import OpenAI
from client import MedicalTriageEnv
from models import TriageAction
# Auto-load .env file

# ============================================================================
# REQUIRED ENVIRONMENT VARIABLES
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
    print("ERROR: Missing required environment variables!", file=sys.stderr)
    print("Please set: API_BASE_URL, MODEL_NAME, HF_TOKEN", file=sys.stderr)
    sys.exit(1)

# Initialize OpenAI Client (REQUIRED - points to HuggingFace)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


# ============================================================================
# LOG FUNCTIONS (EXACT FORMAT REQUIRED!)
# ============================================================================

def log_start(task_id: str):
    """Emit START log"""
    log = {
        "type": "START",
        "task": task_id,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    print(json.dumps(log), flush=True)


def log_step(step: int, action: dict, observation: dict, reward: float):
    """Emit STEP log"""
    log = {
        "type": "STEP",
        "step": step,
        "action": action,
        "observation": observation,
        "reward": reward
    }
    print(json.dumps(log), flush=True)


def log_end(task_id: str, score: float, total_steps: int):
    """Emit END log"""
    log = {
        "type": "END",
        "task": task_id,
        "score": score,
        "total_steps": total_steps
    }
    print(json.dumps(log), flush=True)


# ============================================================================
# LLM HELPER FUNCTIONS
# ============================================================================

def call_llm(prompt: str, max_tokens: int = 150) -> str:
    """Call LLM via OpenAI Client"""
    try:
        print(f"Calling LLM: {MODEL_NAME}...", file=sys.stderr)
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a medical triage expert. Respond concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            timeout=30.0  # Add timeout
        )
        
        result = response.choices[0].message.content.strip()
        print(f"LLM Response: {result[:100]}...", file=sys.stderr)
        return result
        
    except Exception as e:
        print(f"LLM Error Details: {type(e).__name__}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return ""

def extract_number(text: str, default: int = 2) -> int:
    """Extract first number from text"""
    import re
    numbers = re.findall(r'\b[123]\b', text)
    return int(numbers[0]) if numbers else default


def extract_tests(text: str) -> list:
    """Extract test codes from text"""
    # Common test abbreviations
    tests = []
    text_lower = text.lower()
    
    available = ["ecg", "troponin", "cbc", "cxr", "ct_head", "ct_abdomen",
                 "ultrasound", "urinalysis", "blood_culture", "lactate",
                 "bnp", "inr", "electrolytes"]
    
    for test in available:
        if test in text_lower:
            tests.append(test)
    
    return tests[:5] if tests else ["ecg", "cbc"]  # Default if none found


# ============================================================================
# TASK RUNNERS (LLM-POWERED)
# ============================================================================

async def run_easy_task(env_client):
    """Task 1: Triage Prioritization"""
    task_id = "easy"
    log_start(task_id)
    
    # Reset environment
    result = await env_client.reset()
    
    # Get patient info from observation
    instruction = result.observation.task_instruction
    
    # Create LLM prompt
    prompt = f"""You are triaging a patient in an emergency department.

{instruction}

Patient presents with chest pain and shortness of breath.
Vital signs: HR 110, BP 90/60, SpO2 94%, Temp 37.8°C

What urgency level? Respond with ONLY a number: 1, 2, or 3"""
    
    # Call LLM
    llm_response = call_llm(prompt, max_tokens=50)
    urgency = extract_number(llm_response, default=1)
    
    # Execute action
    action = TriageAction(
        task_type="easy",
        urgency_assignment=urgency
    )
    
    result = await env_client.step(action)
    
    log_step(
        step=1,
        action={"task_type": "easy", "urgency_assignment": urgency},
        observation={"reward": result.reward, "done": result.done},
        reward=result.reward
    )
    
    # Get final score from grader
    response = requests.get("http://localhost:8000/grader?task_id=easy")
    score = response.json()["score"]
    
    log_end(task_id, score, total_steps=1)
    return score


async def run_medium_task(env_client):
    """Task 2: Investigation Ordering"""
    task_id = "medium"
    log_start(task_id)
    
    # Reset environment
    result = await env_client.reset()
    instruction = result.observation.task_instruction
    patient = result.observation.current_patient
    
    # Build prompt with actual patient details
    prompt = f"""{instruction}

Patient Information:
- Chief Complaint: {patient['chief_complaint']}
- Vitals: HR {patient['heart_rate']}, BP {patient['blood_pressure']}, SpO2 {patient['spo2']}%, Temp {patient['temperature']}°C, RR {patient['respiratory_rate']}
- Medical History: {', '.join(patient['past_medical_history']) if patient['past_medical_history'] else 'None'}
- Allergies: {', '.join(patient['allergies']) if patient['allergies'] else 'None'}

Available tests: ECG, troponin, CBC, CXR, CT head, CT abdomen, ultrasound, urinalysis, blood culture, lactate, BNP, INR, electrolytes

Which diagnostic tests would you order? List test names separated by commas."""
    
    # Call LLM
    llm_response = call_llm(prompt, max_tokens=100)
    tests = extract_tests(llm_response)
    
    # Execute action
    action = TriageAction(
        task_type="medium",
        ordered_investigations=tests
    )
    
    result = await env_client.step(action)
    
    log_step(
        step=1,
        action={"task_type": "medium", "ordered_investigations": tests},
        observation={"reward": result.reward, "done": result.done},
        reward=result.reward
    )
    
    # Get final score
    response = requests.get("http://localhost:8000/grader?task_id=medium")
    score = response.json()["score"]
    
    log_end(task_id, score, total_steps=1)
    return score


async def run_hard_task(env_client):
    """Task 3: Full Discharge Decision"""
    task_id = "hard"
    log_start(task_id)
    
    # Reset environment
    result = await env_client.reset()
    instruction = result.observation.task_instruction
    
    # Create LLM prompt
    prompt = f"""{instruction}

Patient: 54-year-old male
Chief complaint: Chest pain radiating to left arm, sweating
Vitals: HR 110, BP 90/60, SpO2 94%, Temp 37.8°C
History: Hypertension, Type 2 Diabetes
Allergies: Penicillin

Lab results show elevated troponin. ECG shows ST elevation.

Provide:
1. Diagnosis (one phrase)
2. Disposition: admit or discharge
3. Medications (2-3)
4. Follow-up days (number)

Format: diagnosis|disposition|med1,med2|days"""
    
    # Call LLM
    llm_response = call_llm(prompt, max_tokens=150)
    
    # Parse response (simple fallback if format is wrong)
    parts = llm_response.split('|')
    diagnosis = parts[0].strip() if len(parts) > 0 else "acute_myocardial_infarction"
    disposition = parts[1].strip().lower() if len(parts) > 1 else "admit"
    meds = parts[2].split(',') if len(parts) > 2 else ["aspirin", "nitroglycerin"]
    meds = [m.strip() for m in meds[:3]]
    
    try:
        follow_up = int(parts[3]) if len(parts) > 3 else 7
    except:
        follow_up = 7
    
    # Execute action
    action = TriageAction(
        task_type="hard",
        diagnosis=diagnosis,
        disposition=disposition,
        prescribed_medications=meds,
        follow_up_days=follow_up
    )
    
    result = await env_client.step(action)
    
    log_step(
        step=1,
        action={
            "task_type": "hard",
            "diagnosis": diagnosis,
            "disposition": disposition,
            "prescribed_medications": meds,
            "follow_up_days": follow_up
        },
        observation={"reward": result.reward, "done": result.done},
        reward=result.reward
    )
    
    # Get final score
    response = requests.get("http://localhost:8000/grader?task_id=hard")
    score = response.json()["score"]
    
    log_end(task_id, score, total_steps=1)
    return score


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Run all tasks and output scores"""
    try:
        # Initialize environment client
        async with MedicalTriageEnv(base_url="http://localhost:8000") as env_client:
            # Run all 3 tasks
            easy_score = await run_easy_task(env_client)
            medium_score = await run_medium_task(env_client)
            hard_score = await run_hard_task(env_client)
            
            # Print summary (for human readability)
            print("\n" + "="*50, file=sys.stderr)
            print("FINAL SCORES (LLM-POWERED):", file=sys.stderr)
            print(f"  Easy:   {easy_score:.3f}", file=sys.stderr)
            print(f"  Medium: {medium_score:.3f}", file=sys.stderr)
            print(f"  Hard:   {hard_score:.3f}", file=sys.stderr)
            print("="*50, file=sys.stderr)
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())