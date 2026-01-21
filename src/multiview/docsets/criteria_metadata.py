"""Criterion metadata for document sets.

This module loads metadata for LM-based criteria annotations from available_criteria.yaml.
Each criterion includes:
- description: What the criterion measures (brief)
- default_hint: Default hint used for all hint types if specific hints not provided (optional)
- category_schema_hint: Guidance for creating category-based schemas (optional)
- tag_schema_hint: Guidance for creating tag-based schemas (optional)
- summary_hint: Rich description + format guidance for summaries (optional, auto-generated if not provided)
- triplet_example_hint: Example triplet guidance for LM triplet selection (optional)

Hint resolution order:
1. Specific hint field (e.g., category_schema_hint)
2. default_hint (if specific hint not provided)
3. Auto-generation or None

The YAML values can be either inline text or file references (e.g., "prompts/criteria/file.txt").

Global Criteria:
The YAML file supports a special _global section that defines common criteria available to all datasets.
Datasets automatically inherit global criteria unless they override them with their own definitions.
This eliminates the need to repeatedly define common criteria like word_count across datasets.
"""

from pathlib import Path

import yaml

from multiview.utils.prompt_utils import read_or_return


def load_criteria_metadata():
    """
    Load criteria metadata from YAML config file.

    Supports a _global section for common criteria that all datasets can inherit.
    Dataset-specific criteria override global criteria when both are defined.

    Returns:
        Dictionary mapping dataset names to their criteria metadata
    """
    # Find project root (go up from this file's location to reach the root)
    current_file = Path(__file__)
    # From src/multiview/docsets/criteria_metadata.py -> project root
    project_root = current_file.parent.parent.parent.parent
    config_path = project_root / "configs" / "available_criteria.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Criteria config not found at {config_path}. "
            "Please ensure configs/available_criteria.yaml exists."
        )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Extract global criteria (if present)
    global_criteria_raw = data.pop("_global", {})

    # Process global criteria
    global_criteria = {}
    for criterion_name, metadata in global_criteria_raw.items():
        processed_metadata = {}
        for key, value in metadata.items():
            # Handle nested dicts (like triplet_example_hint)
            if isinstance(value, dict):
                processed_metadata[key] = {
                    k: read_or_return(v, base_dir=project_root) if v else v
                    for k, v in value.items()
                }
            else:
                processed_metadata[key] = (
                    read_or_return(value, base_dir=project_root) if value else value
                )
        global_criteria[criterion_name] = processed_metadata

    # Process each dataset's criteria, merging with global criteria
    processed = {}
    for dataset_name, criteria_dict in data.items():
        # Start with a copy of global criteria
        processed[dataset_name] = global_criteria.copy()

        # Process dataset-specific criteria (these override global)
        for criterion_name, metadata in criteria_dict.items():
            processed_metadata = {}
            for key, value in metadata.items():
                # Handle nested dicts (like triplet_example_hint)
                if isinstance(value, dict):
                    processed_metadata[key] = {
                        k: read_or_return(v, base_dir=project_root) if v else v
                        for k, v in value.items()
                    }
                else:
                    processed_metadata[key] = (
                        read_or_return(value, base_dir=project_root) if value else value
                    )
            # Override global criterion if present
            processed[dataset_name][criterion_name] = processed_metadata

    return processed


# Load all criteria from YAML
_CRITERIA_BY_DATASET = load_criteria_metadata()

# Backward compatibility: expose individual dataset criteria as module-level variables
ANALOGIES_CRITERIA = _CRITERIA_BY_DATASET.get("analogies", {})
GSM8K_CRITERIA = _CRITERIA_BY_DATASET.get("gsm8k", {})
CROSSWORD_CRITERIA = _CRITERIA_BY_DATASET.get("crossword", {})
ROCSTORIES_CRITERIA = _CRITERIA_BY_DATASET.get("rocstories", {})
ABSTRACTSIM_CRITERIA = _CRITERIA_BY_DATASET.get("abstractsim", {})
HACKERNEWS_CRITERIA = _CRITERIA_BY_DATASET.get("hackernews", {})
ONION_NEWS_CRITERIA = _CRITERIA_BY_DATASET.get("onion_headlines", {})
INTENT_EMOTION_CRITERIA = _CRITERIA_BY_DATASET.get("inb_intent_emotion", {})
NYTCLUSTERING_CRITERIA = _CRITERIA_BY_DATASET.get("inb_nytclustering", {})
TRIZ40_CRITERIA = _CRITERIA_BY_DATASET.get("triz40", {})
RATEMYPROF_CRITERIA = _CRITERIA_BY_DATASET.get("inb_ratemyprof", {})
FEEDBACKS_CRITERIA = _CRITERIA_BY_DATASET.get("inb_feedbacks", {})
FEWREL_CRITERIA = _CRITERIA_BY_DATASET.get("inb_fewrel", {})
FEWNERD_CRITERIA = _CRITERIA_BY_DATASET.get("inb_fewnerd", {})
FEWEVENT_CRITERIA = _CRITERIA_BY_DATASET.get("inb_fewevent", {})
INSTRUCTSTSB_CRITERIA = _CRITERIA_BY_DATASET.get("inb_instructstsb", {})
ARXIV_CS_CRITERIA = _CRITERIA_BY_DATASET.get("arxiv_cs", {})
TREX_CRITERIA = _CRITERIA_BY_DATASET.get("trex", {})
HAIKU_CRITERIA = _CRITERIA_BY_DATASET.get("haiku", {})
DICKINSON_CRITERIA = _CRITERIA_BY_DATASET.get("dickinson", {})
INFINITE_PROMPTS_CRITERIA = _CRITERIA_BY_DATASET.get("infinite_prompts", {})
INSPIRED_CRITERIA = _CRITERIA_BY_DATASET.get("inspired", {})
BILLS_CRITERIA = _CRITERIA_BY_DATASET.get("bills", {})
GOODREADS_QUOTES_CRITERIA = _CRITERIA_BY_DATASET.get("goodreads_quotes", {})
ARXIV_ABSTRACT_SENTENCES_CRITERIA = _CRITERIA_BY_DATASET.get(
    "arxiv_abstract_sentences", {}
)
MMLU_CRITERIA = _CRITERIA_BY_DATASET.get("mmlu", {})

# Main criteria lookup - used by the task system
DATASET_CRITERIA = _CRITERIA_BY_DATASET
