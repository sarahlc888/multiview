"""Criterion metadata for document sets.

This module loads metadata for LM-based criteria annotations from available_criteria.yaml.
Each criterion includes:
- description: What the criterion measures
- pairwise_sim_hint: Hint for generating pairwise similarity comparisons (optional)
- category_schema_hint: Guidance for creating category-based schemas
- tag_schema_hint: Guidance for creating tag-based schemas
- summary_hint: Combined guidance + desired format for summaries (optional)
- triplet_example_hint: Example triplet guidance for LM triplet selection (optional)

The YAML values can be either inline text or file references (e.g., "prompts/criteria/file.txt").
"""

from pathlib import Path

import yaml

from multiview.utils.prompt_utils import read_or_return


def load_criteria_metadata():
    """
    Load criteria metadata from YAML config file.

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

    # Process each dataset's criteria
    processed = {}
    for dataset_name, criteria_dict in data.items():
        processed[dataset_name] = {}
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
            processed[dataset_name][criterion_name] = processed_metadata

    return processed


# Load all criteria from YAML
_CRITERIA_BY_DATASET = load_criteria_metadata()

# Backward compatibility: expose individual dataset criteria as module-level variables
GSM8K_CRITERIA = _CRITERIA_BY_DATASET.get("gsm8k", {})
CROSSWORD_CRITERIA = _CRITERIA_BY_DATASET.get("crossword", {})
ROCSTORIES_CRITERIA = _CRITERIA_BY_DATASET.get("rocstories", {})
ABSTRACTSIM_CRITERIA = _CRITERIA_BY_DATASET.get("abstractsim", {})
HACKERNEWS_CRITERIA = _CRITERIA_BY_DATASET.get("hackernews", {})
ONION_NEWS_CRITERIA = _CRITERIA_BY_DATASET.get("onion_news", {})
INTENT_EMOTION_CRITERIA = _CRITERIA_BY_DATASET.get("intent_emotion", {})
NYTCLUSTERING_CRITERIA = _CRITERIA_BY_DATASET.get("nytclustering", {})
TRIZ40_CRITERIA = _CRITERIA_BY_DATASET.get("triz40", {})
RATEMYPROF_CRITERIA = _CRITERIA_BY_DATASET.get("ratemyprof", {})
FEEDBACKS_CRITERIA = _CRITERIA_BY_DATASET.get("feedbacks", {})
FEWREL_CRITERIA = _CRITERIA_BY_DATASET.get("fewrel", {})
FEWNERD_CRITERIA = _CRITERIA_BY_DATASET.get("fewnerd", {})
FEWEVENT_CRITERIA = _CRITERIA_BY_DATASET.get("fewevent", {})
INSTRUCTSTSB_CRITERIA = _CRITERIA_BY_DATASET.get("instructstsb", {})
ARXIV_CS_CRITERIA = _CRITERIA_BY_DATASET.get("arxiv_cs", {})
WN18RR_CRITERIA = _CRITERIA_BY_DATASET.get("wn18rr", {})
FB15K237_CRITERIA = _CRITERIA_BY_DATASET.get("fb15k237", {})
TREX_CRITERIA = _CRITERIA_BY_DATASET.get("trex", {})
HAIKU_CRITERIA = _CRITERIA_BY_DATASET.get("haiku", {})
INSPIRED_CRITERIA = _CRITERIA_BY_DATASET.get("inspired", {})
BILLS_CRITERIA = _CRITERIA_BY_DATASET.get("bills", {})
GOODREADS_QUOTES_CRITERIA = _CRITERIA_BY_DATASET.get("goodreads_quotes", {})

# Main criteria lookup - used by the task system
DATASET_CRITERIA = _CRITERIA_BY_DATASET
