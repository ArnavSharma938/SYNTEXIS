from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


MODEL_SLUGS = {
    "mistral-math": "mistralai/Mathstral-7B-v0.1",
    "deepseek-math": "deepseek-ai/deepseek-math-7b-instruct",
    "prm-judge": "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
}

# Standardized chat template
UNIVERSAL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\\n\\n"
        "{% elif message['role'] == 'user' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
            "\\n\\n{{ message['content'] }}"
        "{% endif %}"
    "{% endfor %}"
)

MODEL_CONFIG = {
    "mistralai/Mathstral-7B-v0.1": {
        "chat_template": UNIVERSAL_CHAT_TEMPLATE,
    },
    "deepseek-ai/deepseek-math-7b-instruct": {
        "chat_template": UNIVERSAL_CHAT_TEMPLATE,
    },
    "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2": {
        "is_prm": True,
    },
}


# NUM_WORKERS is decided by your amount of CPU cores, batch_size is changeable based on GPU memory (may affect results), and seed is a standard 42 for reproducibility
BATCH_SIZE = 2
NUM_WORKERS = 4
SEED = 42


SYSTEM_PROMPT = (
    "You are a formal mathematics assistant. Your task is to solve the given "
    "problem and provide a rigorous, step-by-step deductive proof. Each step in your "
    "reasoning must be a direct logical consequence of the previous steps or the "
    "problem statement. Do not include conversational text or explanations. "
    "Conclude your response with the final answer enclosed in a \\boxed{} environment."
)

ALLOWED_TYPES = {"number", "integer", "expr", "set", "interval", "bool", "string"}
