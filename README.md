# Benchmarking Autoformalization and Subsequent Execution of Mathematical Reasoning in Large Language Models through Chain-of-Thought
[![arXiv](https://img.shields.io/badge/arXiv-2401.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2401.XXXXX)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Existing benchmarks for mathematical reasoning in Large Language Models (LLMs) concentrate mainly on final answer correctness. This answer-based evaluation has a limited view of true LLM reasoning, and it doesn’t differentiate between circumstances where a correct answer comes from valid reasoning versus invalid reasoning. Thus, this paper proposes SYNTEXIS, a benchmark for end-to-end and processbased evaluation of mathematical reasoning. SYNTEXIS assesses model performance over the entire problem solving process: comprehension of a natural language prompt, autoformalization of that prompt into a structured representation, and the execution of solving it step-by-step. We propose two novel evaluation metrics: (1) Final Answer Accuracy (FAA), which uses a symbolic analysis to rigorously check for mathematical equivalence, and (2) Process Soundness Score (PSS), which employs a Process Reward Mode (PRM) to evaluate the logic of the reasoning steps in a step-by-step verification of the chain-of-thought reasoning. By separately evaluating reasoning SYNTEXIS provides more realistic evaluations of LLM mathematical reasoning by providing a review of specific failure modes.

The SYNTEXIS benchmark offers a 146-problem data set
and an evaluation framework with two novel metrics to measure the mathematical reasoning based on multiple aspects. This allows for far more nuanced evaluation of mathematical reasoning in Large Language Models. The methods are discussed below in more nuance:

1.  **Format Success Rate (FSR):** A basic but essential check to see if the model's output adheres to the required format of enclosing the final answer in a `\boxed{...}` environment.
2.  **Final Answer Accuracy (FAA):** This metric goes beyond simple string matching. It uses a layered symbolic parsing system powered by SymPy to evaluate the mathematical equivalence of the model's extracted final answer against the ground truth. It robustly handles various mathematical notations and types.
3.  **Process Soundness Score (PSS):** A novel metric that evaluates the logical integrity of the model's chain-of-thought. Instead of using another LLM as a judge, the PSS analyzer symbolically verifies each mathematical claim made in the reasoning process. It parses each step, identifies mathematical assertions (e.g., `x = y`, `A > B`), and verifies their truthfulness based on the preceding steps. The final score is the ratio of valid claims to total claims, providing a quantitative measure of the reasoning's soundness.

**Note: this benchmark may not perform fairly on Qwen-based models as the Process Reward Model (PRM) that is responsible for calculating the PSS is Qwen-based which will likely result in bias.**

**Note: FSR is a development-based metric used to validate if any model you integrate is returning proper answer formats that can lead to a fair evaluation.**

## Supported Models

The benchmark is configured to support the following models:

*   `mistralai/Mathstral-7B-v0.1`
*   `deepseek-ai/deepseek-math-7b-instruct`

## Benchmark Results

The following results were obtained from a full run on the dataset on a A100 80GB GPU, 42 seed.

| Model Slug | FSR (%) | FAA (%) | PSS (%) |
| :--- | :--- | :--- | :--- |
| `mistralai/Mathstral-7B-v0.1` | 100.00 | 34.93 | 57.72 |
| `deepseek-ai/deepseek-math-7b-instruct`| 100.00 | 43.15 | 60.41 |

## Getting Started

### Prerequisites

*   Python 3.10+
*   UV for dependency management
*   NVIDIA GPU with CUDA support (A100 recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ArnavSharma938/SYNTEXIS.git
    cd SYNTEXIS
    ```

2.  **Install dependencies:**

    We recommend using `uv` for efficient dependency management.

    ```bash
    pip install uv
    uv venv --python 3.12
    source .venv/bin/activate
    uv pip compile pyproject.toml -o uv.lock
    uv pip sync uv.lock
    ```

## Running the Benchmark


### Full Run

You can run the benchmark with `scripts/run_all.sh` to evaluate all supported models defined in `src/config.py`.

```bash
bash scripts/run_all.sh
```

### Singular Run

You can run the benchmark with `src/main.py` to evaluate a select model defined by the `model_name` argument.

```bash
python src/main.py --evaluate --model_name mistral-math
```

### Run a Mini-Test

For quick validation, you can run the benchmark on a small subset of the data using the `--dataset_size` argument.

```bash
python src/main.py --evaluate --model_name mistral-math --dataset_size 4
```

## Citation

```bibtex
@misc{SharmaSyntexis,
  title={Benchmarking Autoformalization and Subsequent Execution of Mathematical Reasoning in Large Language Models through Chain-of-Thought},
  author={Arnav Sharma and Drishti Sharma and Ahmed Wez and Brandon Yee},
  year={2025},
  eprint={2401.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  note={Presented at TBD, CortexPD Labs and Yee-Collins Research Group}
}

```
