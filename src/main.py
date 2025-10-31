import sys
from pathlib import Path
import argparse
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_SLUGS, SEED, OUTPUT_DIR
from src.pipeline.model_evaluation import run_evaluation, collate_results
from src.run_analysis import run_analysis, find_latest_run_dir

# Entry
def main():
    parser = argparse.ArgumentParser(description="Run the SYNTEXIS benchmark pipeline.")
    parser.add_argument("--evaluate", action="store_true", help="Run the full evaluation, collation, and analysis pipeline.")
    parser.add_argument("--collate-only", action="store_true", help="Run only the collation step on the latest run.")
    parser.add_argument("--analyze", action="store_true", help="Run only the final analysis on the latest collated results.")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_SLUGS.keys()) + ['all'], help="The model to evaluate.")
    parser.add_argument("--dataset_size", type=int, default=None, help="Number of problems to run for a mini-test.")
    parser.add_argument("--run_id", type=str, default=None, help="Specify a run_id for collation or analysis.")
    args = parser.parse_args()

    if args.evaluate:
        if not args.model_name:
            return
        
        run_id = run_evaluation(args.model_name, seeds=[SEED], dataset_size=args.dataset_size)
        
        if run_id:
            collate_results(run_id)
            run_analysis(run_id)
        else:
            return

    elif args.collate_only:
        run_id_to_collate = args.run_id or find_latest_run_dir().name
        collate_results(run_id_to_collate)

    elif args.analyze:
        run_id_to_analyze = args.run_id or find_latest_run_dir().name
        run_analysis(run_id_to_analyze)
        
    else:
        print("No action specified. Use --evaluate, --collate-only, or --analyze.")

if __name__ == "__main__":
    main()
