"""Constants for multiview project.

Matches cache directory structure from old repo for compatibility.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ============================================================================
# Cache Directories
# ============================================================================

# Use same cache root as old repo
CACHE_ROOT = Path("/Users/sarahchen/code/.cache")

# Inference cache
INFERENCE_CACHE_DIR = CACHE_ROOT / "inference_cache"

# HuggingFace models cache
HF_CACHE_DIR = CACHE_ROOT / "huggingface"

# Dataset cache
DATASET_CACHE_DIR = CACHE_ROOT / "dataset_cache"

# ============================================================================
# Cache Path Templates
# ============================================================================

# Cache path template for inference (embeddings, LM completions, etc.)
INFERENCE_CACHE_PATH_TEMPLATE = str(
    INFERENCE_CACHE_DIR / "cached_{cache_type}.{annotator_hash}.json"
)

# ============================================================================
# Global Flags
# ============================================================================

# Global flag to enable/disable caching
# Can be overridden per-call with force_refresh or by setting USE_CACHE env var
USE_CACHE = os.environ.get("USE_CACHE", "true").lower() in ("true", "1", "yes")

# Global flag to enable/disable breakpoints for debugging
# Set STEP_THROUGH=true in environment to enable breakpoints
STEP_THROUGH = os.environ.get("STEP_THROUGH", "false").lower() in ("true", "1", "yes")

# ============================================================================
# Output Directories
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ============================================================================
# API Keys
# ============================================================================

# OpenAI - supports multiple keys for load balancing
OPENAI_API_KEYS = os.environ.get(
    "OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", None)
)
if isinstance(OPENAI_API_KEYS, str):
    OPENAI_API_KEYS = OPENAI_API_KEYS.split(",")

# Anthropic
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)

# Gemini (Google)
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", None)
)

# HuggingFace
HF_API_KEY = os.environ.get(
    "HF_TOKEN",
    os.environ.get("HF_API_KEY", os.environ.get("HUGGINGFACE_API_KEY", None)),
)

# ============================================================================
# Environment Configuration
# ============================================================================

# Set HuggingFace cache directory
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR))

# ============================================================================
# Dataset Identifiers
# ============================================================================

# Infinite Chats dataset configuration
INFINITE_CHATS_DATASET_ID = "liweijiang/infinite-chats-taxonomy"

# AidanBench dataset configuration
AIDANBENCH_REPO_URL = "https://github.com/aidanmclaughlin/AidanBench.git"
AIDANBENCH_CACHE_DIR = CACHE_ROOT / "AidanBench"
AIDANBENCH_RESULTS_PATH = AIDANBENCH_CACHE_DIR / "results" / "results.json"

# ============================================================================
# Create directories
# ============================================================================

# Create cache directories if they don't exist
INFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
