"""
inference.py — College Admission Counselling Environment
=========================================================
MANDATORY inference script as required by the hackathon.

Uses OpenAI client format (works with Groq, HF Inference API, or OpenAI).
Set these environment variables before running:

    API_BASE_URL  — LLM endpoint  (Groq: https://api.groq.com/openai/v1)
    MODEL_NAME    — Model to use  (e.g. llama-3.1-8b-instant)
    HF_TOKEN      — Your API key  (Groq key or HF token)

Run:
    python inference.py

Switch to OpenAI: set API_BASE_URL=https://api.openai.com/v1 and MODEL_NAME=gpt-4o-mini
"""

import os
import sys
import json
import time
import textwrap
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI  # Same client works for Groq, HF, and OpenAI

# ── Mandatory variables (from environment config) ──────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")

# Prefer GROQ API key for Groq endpoint; fallback to HF token, OpenAI key or generic API_KEY
API_KEY = (os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or
           os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))

if not API_KEY:
    print("ERROR: Set one of GROQ_API_KEY, HF_TOKEN, OPENAI_API_KEY, or API_KEY environment variables", flush=True)
    print("  Windows: set GROQ_API_KEY=your_key_here", flush=True)
    print("  Mac/Linux: export GROQ_API_KEY=your_key_here", flush=True)
    sys.exit(1)

source = "GROQ_API_KEY" if os.getenv("GROQ_API_KEY") else "HF_TOKEN" if os.getenv("HF_TOKEN") else "OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY") else "API_KEY"
print(f"Using token from {source}, API_BASE_URL={API_BASE_URL}, MODEL_NAME={MODEL_NAME}", flush=True)

# ── OpenAI client (works with Groq/HF/OpenAI — same interface) ────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ── Import environment ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from server.college_env_environment import CollegeEnvironment
    from models import CollegeAction
    from tasks import TASKS, GRADERS
except ImportError as e:
    print(f"Import error: {e}. Run from the college_env directory.", flush=True)
    sys.exit(1)

MAX_STEPS = 15
TEMPERATURE = 0.1  # Low for reproducibility

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Indian college admission counsellor.
    Help the student navigate JOSAA/CSAB counselling to get the best college.

    RULES:
    1. NEVER use "withdraw" — catastrophic and irreversible (-10 reward)
    2. Always start with check_status or check_cutoffs
    3. If upgrade is available, always take it
    4. Pay fee immediately after accepting allotment
    5. report_to_college is the final step (only after fee paid)

    Available actions:
      check_status, check_cutoffs, fill_choices, lock_choices,
      accept_allotment, upgrade_request, pay_seat_fee, report_to_college, withdraw

    Respond ONLY with valid JSON — no explanation, no markdown:
    {"action": "check_status", "target_college": null, "round_number": 1}
""").strip()


def get_llm_action(observation_message: str, step: int, history: List[str]) -> dict:
    """Ask the LLM what action to take. Returns action dict."""
    history_text = "\n".join(history[-4:]) if history else "None"

    user_prompt = textwrap.dedent(f"""
        Step: {step}
        Situation: {observation_message}
        Recent history:
        {history_text}

        What is your next action? Reply with JSON only.
    """).strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=120,
            temperature=TEMPERATURE,
        )
        raw = response.choices[0].message.content.strip()

        # Clean markdown if present
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1].replace("json", "").strip() if len(parts) > 1 else raw
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError:
        # Fallback: find action keyword in response
        raw_lower = raw.lower() if "raw" in dir() else ""
        for act in ["check_cutoffs", "check_status", "fill_choices", "accept_allotment",
                    "upgrade_request", "pay_seat_fee", "report_to_college", "lock_choices"]:
            if act in raw_lower:
                return {"action": act, "target_college": None, "round_number": 1}
        return {"action": "check_status", "target_college": None, "round_number": 1}

    except Exception as e:
        print(f"    LLM error: {e}", flush=True)
        return {"action": "check_status", "target_college": None, "round_number": 1}


def run_episode(task_id: int, run_num: int = 1) -> float:
    """Run one episode. Returns final task score (0.0–1.0)."""
    task = TASKS[task_id]
    env = CollegeEnvironment()
    obs = env._reset_for_task(task_id)
    history: List[str] = []
    total_reward = 0.0
    step_count = 0

    # Output [START] block
    print(f"[START] task={task.name.replace(' ', '_')}", flush=True)

    print(f"\n  Run {run_num} | Task {task_id} ({task.difficulty}): {task.name}", flush=True)
    print(f"  Rank {task.student_rank} ({task.student_category}) | Goal: {task.target_outcome}", flush=True)
    print(f"  {'─' * 55}", flush=True)

    for step in range(1, task.max_steps + 1):
        if obs.done:
            break

        action_data = get_llm_action(obs.message, step, history)
        act = action_data.get("action", "check_status")
        college = action_data.get("target_college")
        rnd = action_data.get("round_number", obs.current_round)

        print(f"  Step {step:2}: {act:<22} college={college or 'N/A'}", flush=True)

        try:
            action = CollegeAction(
                action=act,
                target_college=college,
                round_number=rnd,
            )
            obs = env.step(action)
            total_reward += obs.reward
            step_count += 1
            history.append(f"Step {step}: {act} → reward {obs.reward:+.1f}")
            
            # Output [STEP] block
            print(f"[STEP] step={step_count} reward={obs.reward}", flush=True)
            
            print(f"          reward={obs.reward:+.1f} | score={obs.task_score:.3f} | {obs.message[:65]}", flush=True)
        except Exception as e:
            print(f"          Error: {e}", flush=True)
            break

        time.sleep(0.2)  # Rate limit

    # Output [END] block
    print(f"[END] task={task.name.replace(' ', '_')} score={obs.task_score:.3f} steps={step_count}", flush=True)
    print(f"\n  Final score: {obs.task_score:.3f} / 1.000", flush=True)
    return obs.task_score


def main():
    print("=" * 60, flush=True)
    print("College Admission Counselling — Inference / Baseline", flush=True)
    print(f"API: {API_BASE_URL}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print("=" * 60, flush=True)

    RUNS = 3
    all_results = {}

    for task_id in [1, 2, 3]:
        task = TASKS[task_id]
        print(f"\n{'═' * 60}", flush=True)
        print(f"TASK {task_id} ({task.difficulty}): {task.name}", flush=True)
        print(f"{'═' * 60}", flush=True)

        scores = []
        for run in range(1, RUNS + 1):
            s = run_episode(task_id, run)
            scores.append(s)

        avg = round(sum(scores) / len(scores), 3)
        all_results[task_id] = {
            "name": task.name,
            "difficulty": task.difficulty,
            "scores": scores,
            "average": avg,
        }
        print(f"\n  Task {task_id} avg: {avg:.3f} | runs: {scores}", flush=True)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("RESULTS — paste into README baseline section", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Model: {MODEL_NAME} | Runs: {RUNS}\n", flush=True)
    overall = []
    for tid, d in all_results.items():
        print(f"  Task {tid} ({d['difficulty']:6}): {d['average']:.3f} | {d['name']}", flush=True)
        overall.append(d['average'])
    print(f"\n  Overall: {round(sum(overall)/len(overall), 3):.3f} / 1.000", flush=True)
    print("=" * 60, flush=True)

    with open("baseline_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "api": API_BASE_URL,
                   "runs": RUNS, "results": all_results,
                   "overall": round(sum(overall)/len(overall), 3)}, f, indent=2)
    print("Saved to baseline_results.json", flush=True)


if __name__ == "__main__":
    main()
