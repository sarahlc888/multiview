"""Configuration and constants.

This file contains:
1. User-configurable settings loaded from .env file
2. Hard-coded constants that don't change
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ============================================================================
# USER-CONFIGURABLE SETTINGS (from .env file)
# ============================================================================
# Add your settings here, e.g.:
# API_KEY = os.getenv("API_KEY", "")
# DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

# ============================================================================
# HARD-CODED CONSTANTS (not user-configurable)
# ============================================================================
# Add your constants here, e.g.:
# DEFAULT_MODEL_NAME = "gpt-4"
# MAX_FILE_SIZE_MB = 10
