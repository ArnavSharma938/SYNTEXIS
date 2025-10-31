import logging
import re
import pandas as pd
from transformers import pipeline, AutoTokenizer

logger = logging.getLogger(__name__)

def _is_mathematical_claim(line: str) -> bool:
    # A heuristic to distinguish between conversational text and mathematical claims.
    conversational_starters = [
        "let's", "let", "we have", "solving for", "so,", "the answer is", "thus,",
        "you are a formal mathematics assistant", "according to"
    ]
    if any(line.lower().strip().startswith(starter) for starter in conversational_starters):
        return False

    # Regex to detect common math symbols and LaTeX commands
    math_pattern = re.compile(r"=|\\sqrt|\\frac|<|>|\+|\*|/|\^")
    return math_pattern.search(line) is not None

def run_prm_analysis(df: pd.DataFrame, prm_model_pipeline: pipeline) -> pd.DataFrame:
    scores = []
    justifications = []
    separator = "\n\n"

    for _, row in df.iterrows():
        prompt = str(row['prompt'])
        raw_response = str(row['response'])
        
        # Isolate the assistant's reasoning from the full response
        assistant_markers = ["<|assistant|>", "[/INST]", "Assistant:"]
        last_marker_pos = -1
        for marker in assistant_markers:
            pos = raw_response.rfind(marker)
            if pos > last_marker_pos:
                last_marker_pos = pos
        
        reasoning_text = raw_response[last_marker_pos + len(assistant_markers[0]):] if last_marker_pos != -1 else raw_response
        
        reasoning_steps = [s.strip() for s in reasoning_text.split('\n') if s.strip()]
        
        mathematical_claims = []
        for step in reasoning_steps:
            if _is_mathematical_claim(step):
                parts = step.split('=')
                if len(parts) > 2:
                    for i in range(len(parts) - 1):
                        claim = f"{parts[i].strip()} = {parts[i+1].strip()}"
                        mathematical_claims.append(claim)
                else:
                    mathematical_claims.append(step)

        if not mathematical_claims:
            scores.append(0.0)
            justifications.append({'total_claims_attempted': 0, 'total_claims_valid': 0, 'details': []})
            continue

        valid_claims = 0
        details = []
        
        for i in range(1, len(mathematical_claims) + 1):
            context_steps = mathematical_claims[0:i]
            text_to_evaluate = separator.join((prompt, *context_steps)) + separator
            
            try:
                prediction = prm_model_pipeline(text_to_evaluate)[0]
                is_correct = prediction['label'] == 'LABEL_1'
                
                if is_correct:
                    valid_claims += 1
                
                details.append({
                    'claim': mathematical_claims[i-1],
                    'is_valid': is_correct,
                    'prm_score': prediction['score']
                })
            except Exception as e:
                logger.error(f"Error during PRM evaluation for problem {row.get('problem_id')}: {e}")
                details.append({
                    'claim': mathematical_claims[i-1],
                    'is_valid': False,
                    'error': str(e)
                })

        final_score = valid_claims / len(mathematical_claims) if mathematical_claims else 0.0
        scores.append(final_score)
        justifications.append({
            'total_claims_attempted': len(mathematical_claims), 
            'total_claims_valid': valid_claims,
            'details': details
        })

    df['pss'] = scores
    df['pss_justification'] = justifications
    
    logger.info("PRM-based PSS calculation complete.")
    return df
