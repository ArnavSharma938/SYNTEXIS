import math
import re
import logging
from typing import Any
import pandas as pd
import ast
import string
from tokenize import TokenError
from sympy import sympify, simplify, SympifyError, latex
from sympy.parsing.sympy_parser import parse_expr
from latex2sympy2 import latex2sympy
import sympy.parsing.sympy_parser as T

logger = logging.getLogger(__name__)

def _normalize_text(s: str) -> str:
    # Normalizes a string for comparison by lowercasing and removing punctuation/whitespace.
    if not isinstance(s, str):
        s = str(s)
    return "".join(c for c in s if c not in string.punctuation).lower().strip()

def normalize_answer(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[7:-1]

    # "Zero-trust" parsing
    try:
        return str(latex2sympy(s))
    except Exception:
        logger.debug(f"Could not parse '{s}' with latex2sympy2. Falling back to standard SymPy parsing.")
        try:
            s = s.replace('^', '**')
            transformations = (T.standard_transformations + (T.implicit_multiplication_application,))
            return str(parse_expr(s, transformations=transformations))
        except (SympifyError, TypeError, SyntaxError, AttributeError, TokenError, IndexError) as e:
            logger.warning(f"Could not parse '{s}' as a standard expression: {e}. Proceeding with raw string.")
            return s

def answers_equal_lenient(
    answer_type: str, pred_str: str | None, gold_str: str | None
) -> tuple[bool, Any, Any]:
    try:
        if pred_str is None or gold_str is None:
            return False, None, None

        if answer_type in ('text', 'string'):
            norm_pred = _normalize_text(pred_str)
            norm_gold = _normalize_text(gold_str)
            is_correct = norm_pred == norm_gold
            logger.debug(f"Comparing TEXT: '{norm_pred}' vs '{norm_gold}' -> {is_correct}")
            return is_correct, norm_pred, norm_gold
        
        norm_pred = normalize_answer(pred_str)
        norm_gold = normalize_answer(gold_str)

        if norm_pred is None or norm_gold is None:
            return False, None, None

        try:
            expr_pred = parse_expr(norm_pred, transformations='all')
            expr_gold = parse_expr(norm_gold, transformations='all')
            
            if simplify(expr_pred - expr_gold) == 0:
                logger.debug(f"Symbolic equality TRUE: {expr_pred} == {expr_gold}")
                return True, expr_pred, expr_gold
        except (SympifyError, TypeError, SyntaxError, AttributeError, ValueError, TokenError, IndexError) as e:
            logger.debug(f"Symbolic comparison failed for '{norm_pred}' vs '{norm_gold}': {e}. Attempting numerical.")

        try:
            num_pred = float(parse_expr(norm_pred).evalf())
            num_gold = float(parse_expr(norm_gold).evalf())
            
            if math.isclose(num_pred, num_gold, rel_tol=1e-4):
                logger.debug(f"Numerical equality TRUE: {num_pred} approx {num_gold}")
                return True, num_pred, num_gold
        except (SympifyError, TypeError, SyntaxError, AttributeError, ValueError, TokenError, IndexError) as e:
            logger.debug(f"Numerical comparison also failed for '{norm_pred}' vs '{norm_gold}': {e}")

        logger.debug(f"Comparison FALSE for '{norm_pred}' vs '{norm_gold}'")
        return False, norm_pred, norm_gold
    except Exception as e:
        logger.error(f"An unexpected error occurred in answers_equal_lenient: {e}. Pred: '{pred_str}', Gold: '{gold_str}'. Returning FALSE.")
        return False, pred_str, gold_str

def run_faa_analysis(df: pd.DataFrame) -> pd.DataFrame:
    
    if "format_success" not in df.columns:
        df["format_success"] = df["final_answer"].notna().astype(int)
    else:
        df["format_success"] = pd.to_numeric(df["format_success"], errors='coerce').fillna(0).astype(int)

    def check_correctness(row):
        try:
            gold_value = row.get("gold", {})
            if isinstance(gold_value, str):
                gold_value = ast.literal_eval(gold_value)
            
            answer_type = gold_value.get("type", row.get("answer_type"))
            gold_final_answer = gold_value.get("value", gold_value) # Fallback for simple gold values
            
            model_final_answer = row["final_answer"]
            
            is_correct, _, _ = answers_equal_lenient(answer_type, model_final_answer, str(gold_final_answer))
            return is_correct
        except Exception as e:
            logger.error(f"Error checking correctness for problem {row.get('problem_id')}: {e}")
            return False

    df["is_correct"] = df.apply(check_correctness, axis=1)
    return df
