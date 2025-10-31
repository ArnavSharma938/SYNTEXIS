import pandas as pd
import sys
from pathlib import Path
import logging
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, MODEL_SLUGS
from src.analysis.faa_analyzer import run_faa_analysis
from src.analysis.pss_analyzer import run_prm_analysis
from src.utils.model_utils import load_model_and_tokenizer
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_latest_run_dir() -> Path:
    run_dirs = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("eval_")])
    if not run_dirs:
        raise FileNotFoundError("No evaluation runs found in the output directory.")
    return run_dirs[-1]

def load_responses(run_id: str) -> pd.DataFrame:
    run_dir = OUTPUT_DIR / run_id
    responses_path = run_dir / "all_model_responses.csv"
    
    if not responses_path.exists():
        raise FileNotFoundError(f"Collated responses not found at {responses_path}.")
        
    logging.info(f"Loading responses from {responses_path}")
    return pd.read_csv(responses_path)

def print_summary_metrics(df: pd.DataFrame):
    df['pss_justification'] = df['pss_justification'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df_with_reasoning = df[df['pss_justification'].apply(lambda x: x['total_claims_attempted'] > 0)]

    pss_summary = df_with_reasoning.groupby('model_slug')['pss'].mean().reset_index() if not df_with_reasoning.empty else pd.DataFrame(columns=['model_slug', 'pss'])

    summary = df.groupby('model_slug').agg(
        num_problems=('problem_id', 'nunique'),
        fsr=('format_success', 'mean'),
        faa=('is_correct', lambda x: x[df['format_success'] == 1].mean())
    ).reset_index()

    summary = summary.merge(pss_summary, on='model_slug', how='left').fillna(0)

    summary['fsr'] = summary['fsr'] * 100
    summary['faa'] = summary['faa'] * 100
    summary['pss'] = summary['pss'] * 100

    print("\nBenchmark Summary")
    print(summary.to_string(index=False, float_format="%.2f"))
    print("----------------------------------\n")

def run_analysis(run_id: str):
    prm_model_slug = MODEL_SLUGS.get("prm-judge")
    if not prm_model_slug:
        raise ValueError("PRM model slug 'prm-judge' not found in config.")
    
    prm_model, prm_tokenizer, prm_device = load_model_and_tokenizer(prm_model_slug)
    
    # The PRM is a reward model so it is framed as a text-classification task here
    prm_pipeline = pipeline(
        "text-classification", 
        model=prm_model, 
        tokenizer=prm_tokenizer
    )

    try:
        load_start = time.time()
        responses_df = load_responses(run_id)

        logging.info("Running FAA and FSR analysis...")
        faa_start = time.time()
        analyzed_df = run_faa_analysis(responses_df)
        
        logging.info("Running PSS analysis...")
        pss_start = time.time()
        analyzed_df = run_prm_analysis(analyzed_df, prm_pipeline)

        print_summary_metrics(analyzed_df)

        run_dir = OUTPUT_DIR / run_id
        output_path = run_dir / "final_analysis_results.csv"
        analyzed_df.to_csv(output_path, index=False)
        logging.info(f"Final results saved to {output_path}")

    except FileNotFoundError as e:
        logging.error(f"Analysis failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)

if __name__ == "__main__":
    latest_run = find_latest_run_dir()
    run_analysis(latest_run.name)
