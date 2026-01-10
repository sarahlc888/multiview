"""Synthesis configurations for document sets.

This module contains synthesis prompts for LM-based document generation across
different document sets. Each criterion includes prompts for:
- remix_prompt: Single remix prompt template (behavior controlled by runtime variables)
"""

from multiview.utils.prompt_utils import read_or_return

# GSM8K: Math word problems
GSM8K_SYNTHESIS_CONFIGS = {
    "arithmetic": {
        "remix_prompt": read_or_return("prompts/custom/gsm8k_arithmetic_remix.txt"),
    }
}
