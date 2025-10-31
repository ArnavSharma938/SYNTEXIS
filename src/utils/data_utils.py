import json
from typing import List, Dict, Any
import random

from src.config import ALLOWED_TYPES

def _inject_format_hints(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    FINAL_TAG = "[Final-format]:"
    
    def final_hint(t: str) -> str:
        return {
            "number":   "Final must be a single exact number (e.g., 2, 7/3, pi/6).",
            "integer":  "Final must be a single integer.",
            "expr":     "Final must be a single simplified expression.",
            "set":      "Final must be a set like {-1, 1}.",
            "interval": "Final must be a single interval (or union).",
            "bool":     "Final must be either True or False.",
            "string":   "Final must be a short string token, no sentences."
        }.get(t, "Final must be in the specified format.")

    for p in problems:
        prompt = p.get("prompt", "").rstrip()
        if FINAL_TAG not in prompt:
            t = p.get("answer_type")
            prompt = prompt + f"\\n\\n{FINAL_TAG} " + final_hint(t)
        p["prompt"] = prompt
    return problems

def _validate_schema(problems: List[Dict[str, Any]]):
    errors, seen_ids = [], set()
    for p in problems:
        pid = p.get("id")
        if not pid:
            errors.append(f"Problem entry missing 'id'. Problem data: {p}")
        elif pid in seen_ids:
            errors.append(f"Duplicate problem id found: {pid}")
        seen_ids.add(pid)

        t = p.get("answer_type")
        if t not in ALLOWED_TYPES:
            errors.append(f"{pid}: Invalid 'answer_type'='{t}'. Must be one of {ALLOWED_TYPES}.")

        gold_data = p.get("gold")
        if not gold_data or not isinstance(gold_data, dict) or gold_data.get("type") != t or "value" not in gold_data:
            errors.append(f"{pid}: 'gold' field missing, malformed, or 'type' does not match 'answer_type'. Gold data: {gold_data}")

        role = p.get("role")
        if role not in {"original", "counterfactual", "variant"}:
            errors.append(f"{pid}: Invalid 'role'='{role}'. Must be one of \"original\", \"counterfactual\", or \"variant\".")
        if role in ["counterfactual", "variant"] and not p.get("variant_of"):
            errors.append(f"{pid}: Problem with role '{role}' is missing 'variant_of' field.")

        if "prompt" not in p or not isinstance(p["prompt"], str) or not p["prompt"].strip():
            errors.append(f"{pid}: 'prompt' field is missing or empty.")

    if errors:
        raise ValueError("Schema validation failed:\\n- " + "\\n- ".join(errors))
