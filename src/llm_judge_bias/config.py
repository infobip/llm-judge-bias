"""Shared configuration for the experiments and analyses."""

from pathlib import Path

SEED = 42
THRESHOLD = 4.0
TEMPERATURES = (0.0, 0.5, 1.0, 2.0)
CONDITIONS = ("C0", "C1", "C2")
MAX_RETRIES = 5
MAX_CONCURRENT = 9

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
TASKS_DIR = ROOT / "resources" / "tasks"

API_MODELS = (
    "infobip-gpt-4-1",
    "infobip-deepseek-r1",
    "bedrock-claude-4.5-sonnet",
)
HUGGINGFACE_MODELS = (
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
)
MODEL_SHORT_NAMES = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "bedrock-claude-4.5-sonnet": "Claude-4.5-Sonnet",
    "google/gemma-2-9b-it": "Gemma-2-9B",
    "infobip-deepseek-r1": "DeepSeek-R1",
    "infobip-gpt-4-1": "GPT-4.1",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B",
}

DEFAULT_RUBRIC = {
    "criteria": [
        {"name": "Faithfulness", "description": "no fabricated facts"},
        {"name": "Coverage", "description": "all key concepts included"},
        {"name": "Coherence", "description": "logical flow, clear structure"},
        {"name": "Concision", "description": "brief without losing essential details"},
    ],
}
CREATIVE_WRITING_RUBRIC = {
    "criteria": [
        {"name": "Creativity", "description": "original ideas and unique perspective"},
        {"name": "Engagement", "description": "compelling narrative"},
        {"name": "Coherence", "description": "logical plot and structure"},
        {"name": "Language", "description": "clear prose, grammar, and style"},
    ],
}
FACTUAL_QA_RUBRIC = {
    "criteria": [
        {"name": "Accuracy", "description": "factually correct information"},
        {"name": "Clarity", "description": "easy to understand explanation"},
        {"name": "Completeness", "description": "covers key aspects"},
        {"name": "Concision", "description": "efficient without unnecessary detail"},
    ],
}
