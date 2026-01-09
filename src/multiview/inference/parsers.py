"""Output parsers for inference completions.

Ported from old repo with simplifications. Provides parsers for:
- Embeddings (vector_parser)
- JSON responses (json_parser)
- Raw text (text_parser)
- Dictionary fields (dict_parser)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def vector_parser(completion: dict, **kwargs) -> Any:
    """Parse embedding vectors from completion.

    Args:
        completion: Completion dict from embedding model
            Should have "vector" key with embedding vector

    Returns:
        The vector (numpy array or list)
    """
    if isinstance(completion, dict):
        if "vector" not in completion:
            raise ValueError(f"No vector found in completion keys: {completion.keys()}")
        return completion["vector"]
    else:
        # Assume it's already a vector
        return completion


def json_parser(
    completion: str | dict, annotation_key: str | None = None, **kwargs
) -> Any:
    """Parse JSON from completion text.

    Handles common cases:
    - JSON wrapped in ```json markdown blocks
    - JSON objects and arrays
    - Extracting specific keys via annotation_key

    Args:
        completion: Completion text (may be wrapped in markdown) or dict with "text" key
        annotation_key: If provided, extract this key from the JSON
            If None, return the full parsed JSON

    Returns:
        Parsed JSON value (or specific key if annotation_key provided)

    Examples:
        >>> json_parser('{"score": 5, "reason": "good"}', "score")
        5
        >>> json_parser('[{"score": 5}, {"score": 3}]', "score")
        [5, 3]
        >>> json_parser('```json\\n{"value": true}\\n```', "value")
        True
    """
    if completion == "":
        raise ValueError("The completion is empty.")

    # Handle dict input (from some APIs)
    if isinstance(completion, dict):
        if "text" in completion:
            completion = completion["text"]
        else:
            raise ValueError(f"Dict completion missing 'text' key: {completion.keys()}")

    # Handle list input (from some APIs)
    if isinstance(completion, list):
        if len(completion) != 1:
            raise ValueError(f"Expected list of length 1, got {len(completion)}")
        if isinstance(completion[0], dict) and "text" in completion[0]:
            completion = completion[0]["text"]
        else:
            raise ValueError(f"Unexpected completion format: {type(completion[0])}")

    # Extract JSON from markdown code block if present
    if "```json" in completion:
        # Use greedy match to handle nested backticks within JSON strings
        match = re.search(r"```json(.*)```", completion, re.DOTALL)
        if match:
            completion = match.group(1).strip()

    # Try to parse JSON
    try:
        json_loaded = json.loads(completion)
    except json.JSONDecodeError:
        # Try wrapping in braces (common LM mistake)
        try:
            json_loaded = json.loads(f"{{\n{completion}\n}}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from completion: {completion[:200]}")
            raise ValueError(f"Invalid JSON in completion: {e}") from e

    # Extract annotation key if requested
    if annotation_key is None:
        # Return full JSON (wrapped in list for consistency)
        return json_loaded if isinstance(json_loaded, list) else [json_loaded]

    # Extract key from dict or list of dicts
    if isinstance(json_loaded, dict):
        return json_loaded[annotation_key]
    elif isinstance(json_loaded, list):
        return [item[annotation_key] for item in json_loaded]
    else:
        raise ValueError(f"Unexpected JSON type: {type(json_loaded)}")


def text_parser(completion: str | dict, **kwargs) -> str:
    """Parse raw text from completion.

    Args:
        completion: Completion (string or dict with "text" key)

    Returns:
        Raw text string
    """
    if isinstance(completion, dict):
        if "text" in completion:
            return completion["text"]
        elif "content" in completion:
            return completion["content"]
        else:
            raise ValueError(
                f"No text/content found in completion: {completion.keys()}"
            )
    elif isinstance(completion, str):
        return completion
    else:
        raise ValueError(f"Unexpected completion type: {type(completion)}")


def dict_parser(completion: dict, use_key: str, **kwargs) -> Any:
    """Parse a specific field from completion dict.

    Args:
        completion: Completion dict
        use_key: Key to extract from dict

    Returns:
        Value at the specified key
    """
    if not isinstance(completion, dict):
        raise ValueError(f"Expected dict, got {type(completion)}")
    if use_key not in completion:
        raise ValueError(
            f"Key '{use_key}' not found in completion: {completion.keys()}"
        )
    return completion[use_key]


def noop_parser(completion: Any, **kwargs) -> Any:
    """Return completion as-is (no-op parser).

    Args:
        completion: Any completion

    Returns:
        The completion unchanged
    """
    return completion


def regex_parser(
    completion: str | dict, outputs_to_match: dict[str, Any], **kwargs
) -> Any:
    r"""Parse completion using regex patterns.

    Searches for regex patterns in the completion text and returns the
    first match's associated value.

    Args:
        completion: Completion text (string or dict with "text" key)
        outputs_to_match: Dict mapping regex patterns to return values
            e.g., {"pattern1": 1, "pattern2": 2}

    Returns:
        The value associated with the first matching pattern

    Example:
        >>> completion = "Final judgement: 5"
        >>> patterns = {r"[Jj]udgement:?\s*5": 5, r"[Jj]udgement:?\s*1": 1}
        >>> regex_parser(completion, patterns)
        5
    """
    # Extract text from dict if needed
    if isinstance(completion, dict):
        if "text" in completion:
            text = completion["text"]
        elif "content" in completion:
            text = completion["content"]
        else:
            raise ValueError(f"No text/content in completion: {completion.keys()}")
    else:
        text = completion

    # Compile patterns if they're strings
    compiled_patterns = {}
    for pattern, value in outputs_to_match.items():
        if isinstance(pattern, str):
            compiled_patterns[re.compile(pattern)] = value
        else:
            compiled_patterns[pattern] = value

    # Find first match
    first_match = None
    first_value = None
    for pattern, value in compiled_patterns.items():
        match = pattern.search(text)
        if match:
            if first_match is None or match.start() < first_match.start():
                first_match = match
                first_value = value

    if first_match is None:
        logger.warning(f"No regex match found in completion: {text[:200]}")
        return None

    return first_value


def delimiter_parser(completion: str | dict, delimiter: str = "###", **kwargs) -> str:
    """Parse completion by extracting text after a delimiter.

    Useful for extracting final answers that come after a delimiter marker.

    Args:
        completion: Completion text (string or dict with "text" key)
        delimiter: Delimiter string to split on (default: "###")

    Returns:
        Text after the last occurrence of the delimiter, stripped

    Example:
        >>> completion = "reasoning...\\n###\\nFinal answer here"
        >>> delimiter_parser(completion, delimiter="###")
        "Final answer here"
    """
    # Extract text from dict if needed
    if isinstance(completion, dict):
        if "text" in completion:
            text = completion["text"]
        elif "content" in completion:
            text = completion["content"]
        else:
            raise ValueError(f"No text/content in completion: {completion.keys()}")
    else:
        text = completion

    # Split by delimiter and take everything after the last occurrence
    if delimiter in text:
        parts = text.split(delimiter)
        result = parts[-1].strip()
        return result
    else:
        logger.warning(f"Delimiter '{delimiter}' not found in completion")
        # Return full text if delimiter not found
        return text.strip()


# Registry for parser lookup
PARSER_REGISTRY = {
    "vector": vector_parser,
    "json": json_parser,
    "text": text_parser,
    "dict": dict_parser,
    "noop": noop_parser,
    "regex": regex_parser,
    "delimiter": delimiter_parser,
}


def get_parser(name: str):
    """Get a parser function by name.

    Args:
        name: Parser name (e.g., "vector", "json", "text")

    Returns:
        Parser function

    Raises:
        ValueError: If parser name is not found
    """
    if name not in PARSER_REGISTRY:
        raise ValueError(
            f"Unknown parser: {name}. "
            f"Available parsers: {sorted(PARSER_REGISTRY.keys())}"
        )
    return PARSER_REGISTRY[name]
