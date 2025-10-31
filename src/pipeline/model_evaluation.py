import sys
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import json
import time
import uuid
import re

from functools import partial
from torch.utils.data import DataLoader, Dataset

from src.config import (
    MODEL_SLUGS, SEED,
    OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS, DATA_DIR
)
from src.utils.data_utils import _validate_schema, _inject_format_hints
from src.utils.model_utils import (
    load_model_and_tokenizer, generate_responses_batch, prepare_batch_for_generation,
    extract_final_answer
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


def setup_run_directory(run_id: str) -> Path:
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "responses").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def load_evaluation_problems() -> list:
    dataset_file = DATA_DIR / "dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"The expert-validated dataset was not found at {dataset_file}")

    with open(dataset_file, 'r') as f:
        data = json.load(f)
        problems = data.get("problems", [])

    _validate_schema(problems)
    final_problems = _inject_format_hints(problems)
    
    return final_problems


class ProblemDataset(Dataset):
    def __init__(self, problems):
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]


def run_evaluation(model_name: str, seeds: list[int] = [SEED], dataset_size: int | None = None) -> str | None:
    if model_name not in MODEL_SLUGS.keys() and model_name != 'all':
        raise ValueError(f"Model '{model_name}' not in supported models: {list(MODEL_SLUGS.keys())}")

    run_id = f"eval_{time.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    run_dir = setup_run_directory(run_id)

    models_to_run = list(MODEL_SLUGS.keys()) if model_name == 'all' else [model_name]
    
    for model_slug in models_to_run:
        model_hf_path = MODEL_SLUGS[model_slug]
        
        logging.info(f"Loading model: {model_hf_path}")
        model, tokenizer, device = load_model_and_tokenizer(model_hf_path)
        
        run_evaluation_for_model(
            model_hf_path, seeds, run_id, dataset_size, model, tokenizer, device
        )
    
    return run_id

def run_evaluation_for_model(
    model_hf_path: str, seeds: list[int], run_id: str, dataset_size: int | None,
    model, tokenizer, device
):
    run_dir = OUTPUT_DIR / run_id
    all_seed_responses = []
    for seed in seeds:
        logging.info(f"Running evaluation for model: {model_hf_path} with seed: {seed}")
            
        problems = load_evaluation_problems()
        
        if dataset_size:
            if dataset_size > len(problems):
                logging.warning(f"Requested dataset_size of {dataset_size} is larger than the number of available problems ({len(problems)}). Using all problems.")
            else:
                logging.info(f"Running on a subset of {dataset_size} problems for a mini-test.")
                problems = problems[:dataset_size]
        
        collate_fn = partial(
            lambda batch, model_name, tokenizer: (batch, prepare_batch_for_generation([p["prompt"] for p in batch], model_name, tokenizer)),
            model_name=model_hf_path,
            tokenizer=tokenizer
        )

        problem_dataset = ProblemDataset(problems)
        data_loader = DataLoader(
            problem_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True
        )

        responses = []
        with tqdm(total=len(problems), desc=f"Generating responses for {model_hf_path}") as pbar:
            for i, (p_batch, (input_ids, attention_mask)) in enumerate(data_loader):
                batch_start_time = time.time()
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                try:
                    logging.debug(f"Processing batch {i+1}/{len(data_loader)} for model {model_hf_path}, seed {seed}")
                    
                    batch_responses = generate_responses_batch(
                        model, tokenizer, input_ids, attention_mask
                    )
                    
                    post_process_start = time.time()
                    for j, p in enumerate(p_batch):
                        raw_text = batch_responses[j]
                        final_answer = extract_final_answer(raw_text)
                        
                        result_row = {
                            'problem_id': p['id'],
                            'model_slug': model_hf_path,
                            'run_id': run_id,
                            'seed': seed,
                            'prompt': p['prompt'],
                            'response': raw_text,
                            'final_answer': final_answer,
                            'format_success': 1 if final_answer is not None else 0,
                            'answer_type': p.get('answer_type'),
                            'gold': p.get('gold'),
                            'grading_policy': p.get('grading_policy')
                        }
                        responses.append(result_row)
                    
                    pbar.update(len(p_batch))

                except Exception as e:
                    logging.error(f"Batch {i} for model {model_hf_path} with seed {seed} failed: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    pbar.update(len(p_batch))


        seed_results_df = pd.DataFrame(responses)
        seed_output_path = run_dir / "responses" / f"{model_hf_path.replace('/', '_')}_seed_{seed}_responses.csv"
        seed_results_df.to_csv(seed_output_path, index=False)
        logging.info(f"Saved {len(responses)} responses for seed {seed} to {seed_output_path}")

        all_seed_responses.extend(responses)

    if all_seed_responses:
        model_results_df = pd.DataFrame(all_seed_responses)
        model_output_path = run_dir / "responses" / f"{model_hf_path.replace('/', '_')}_all_seeds_responses.csv"
        model_results_df.to_csv(model_output_path, index=False)
        logging.info(f"Saved aggregated results for model {model_hf_path} to {model_output_path}")


def collate_results(run_id: str):
    run_dir = OUTPUT_DIR / run_id
    if not run_dir.is_dir():
        logging.error(f"Run directory {run_dir} not found for collation.")
        return

    logging.info(f"Collating results from run: {run_dir.name}")

    all_dfs = []

    responses_dir = run_dir / "responses"
    if not responses_dir.is_dir():
        logging.warning(f"No 'responses' directory found in {run_dir.name}. Skipping collation.")
        return

    for csv_path in responses_dir.glob("*_all_seeds_responses.csv"):
        if csv_path.exists():
            all_dfs.append(pd.read_csv(csv_path))

    if not all_dfs:
        logging.warning("No model response CSVs found to collate.")
        return
        
    collated_df = pd.concat(all_dfs, ignore_index=True)
    
    output_path = run_dir / "all_model_responses.csv"
    collated_df.to_csv(output_path, index=False)
    logging.info(f"Collated results saved to: {output_path}")


if __name__ == "__main__":
    logging.info("Starting model_evaluation.py")
    raise NotImplementedError("This script is meant to be run via main.py --evaluate or main.py --collate.")


