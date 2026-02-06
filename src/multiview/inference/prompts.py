"""Prompt formatting utilities for inference.

Handles complex prompt formatting including:
- Embedding query/doc instructions
- Force prefills
- Image signatures (for future VLM support)
- Prompt deduplication

Ported from old repo with simplifications.
"""

from __future__ import annotations

import hashlib
import logging
import re

from multiview.inference.presets import InferenceConfig
from multiview.utils.prompt_utils import read_or_return

logger = logging.getLogger(__name__)


_ESCAPED_LBRACE = "\x00\x00"
_ESCAPED_RBRACE = "\x01\x01"


def _safe_format_template(template: str, kwargs: dict) -> str:
    """Format a prompt template by substituting only {identifier} placeholders.

    Many prompt templates contain literal braces (e.g., JSON examples). Using
    Python's `str.format` would treat those as placeholders and raise KeyError.
    """
    # Match Python's behavior for escaped braces.
    working = template.replace("{{", _ESCAPED_LBRACE).replace("}}", _ESCAPED_RBRACE)

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in kwargs:
            available = ", ".join(sorted(kwargs.keys()))
            raise KeyError(
                f"Missing template key '{key}'. Available keys: [{available}]"
            )
        return str(kwargs[key])

    working = re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", _replace, working)
    return working.replace(_ESCAPED_LBRACE, "{").replace(_ESCAPED_RBRACE, "}")


def _hash_bytes(payload: bytes) -> str:
    """Hash bytes payload for image signature."""
    digest = hashlib.sha256(payload).hexdigest()
    return f"bytes:{len(payload)}:{digest}"


def _build_image_signature(image_payload) -> str:
    """Build a signature string for an image payload.

    Used to create unique cache keys for prompts with images.
    """
    if image_payload is None:
        return "none"
    if isinstance(image_payload, str):
        return image_payload
    if isinstance(image_payload, bytes):
        return _hash_bytes(image_payload)
    if isinstance(image_payload, dict):
        if image_payload.get("bytes") is not None:
            return _hash_bytes(image_payload["bytes"])
        if image_payload.get("path") is not None:
            return f"path:{image_payload['path']}"
        if image_payload.get("url") is not None:
            return image_payload["url"]
        return f"dict:{repr(image_payload)}"
    # PIL Image-like object
    if (
        hasattr(image_payload, "tobytes")
        and hasattr(image_payload, "size")
        and hasattr(image_payload, "mode")
    ):
        try:
            image_bytes = image_payload.tobytes()
            digest = hashlib.sha256(image_bytes).hexdigest()
            return f"pil:{image_payload.mode}:{image_payload.size}:{digest}"
        except Exception:
            pass
    return f"obj:{type(image_payload).__name__}:{repr(image_payload)}"


def _build_packed_prompt(
    base_prompt: str,
    instruction: str | None = None,
    image=None,
    force_prefill: str | None = None,
    query: str | None = None,
) -> str:
    """Build a packed prompt for cache key generation.

    Packed prompts include all components that affect the completion:
    - Query (for rerankers)
    - Embedding instruction (prepended)
    - Base prompt
    - Image signature (if present)
    - Force prefill (if present)

    Args:
        base_prompt: The main prompt text
        instruction: Embedding instruction
        image: Image payload (will be converted to signature)
        force_prefill: Forced prefill string
        query: Query string (for providers like Voyage that need separate query/doc)

    Returns:
        Packed prompt string suitable for cache key generation
    """
    packed = base_prompt

    # Prepend query (if present) - critical for reranker cache keys
    if query:
        packed = f"Query: {query}\n{packed}"

    # Prepend instruction
    if instruction:
        packed = instruction + packed

    # Append image signature(s)
    # Support both single image and list of images
    if image is not None:
        if isinstance(image, list):
            # Multi-image: build signatures for each
            image_sigs = [
                _build_image_signature(img) for img in image if img is not None
            ]
            if image_sigs:
                # Join all signatures to create unique cache key
                combined_sig = "|".join(image_sigs)
                packed = (
                    f"{packed}\n\n<image_signatures>{combined_sig}</image_signatures>"
                )
        else:
            # Single image (backward compatible)
            image_sig = _build_image_signature(image)
            packed = f"{packed}\n\n<image_signature>{image_sig}</image_signature>"

    # Append prefill
    if force_prefill:
        packed = packed + force_prefill

    return packed


class PromptCollection:
    """Collection of prompts with associated metadata.

    `packed_prompts` are used as cache keys. They include all the components
    (base prompt + instructions + prefills + image signature) to differentiate
    between queries that have the same `prompt` field but different instructions.
    """

    def __init__(
        self,
        packed_prompts: list[str],
        prompts: list[str],
        instructions: list[str] | None = None,
        force_prefill: list[str] | None = None,
        images: list | None = None,
        queries: list[str] | None = None,
    ):
        """Initialize prompt collection.

        Args:
            packed_prompts: Prompts packed with instructions (used as cache keys)
            prompts: Base prompts without instructions
            instructions: Embedding instructions (if any)
            force_prefill: Forced prefill strings (if any)
            images: Image payloads (if any)
            queries: Query strings (for providers like Voyage that need separate query/doc)
        """
        self.packed_prompts = packed_prompts
        self.prompts = prompts
        self.instructions = instructions
        self.force_prefill = force_prefill
        self.images = images
        self.queries = queries

        # Validate lengths match
        assert prompts is not None
        assert packed_prompts is not None
        assert len(packed_prompts) == len(prompts)
        if instructions is not None:
            assert len(instructions) == len(prompts)
        if force_prefill is not None:
            assert len(force_prefill) == len(prompts)
        if images is not None:
            assert len(images) == len(prompts)
        if queries is not None:
            assert len(queries) == len(prompts)

    def to_dict(self) -> dict:
        """Convert to dictionary for passing to completion functions."""
        kwargs = {
            "non_packed_prompts": self.prompts,
            "packed_prompts": self.packed_prompts,
        }

        if self.instructions is not None:
            kwargs["instructions"] = self.instructions
        if self.force_prefill is not None:
            kwargs["force_prefills"] = self.force_prefill
        if self.images is not None:
            kwargs["images"] = self.images
        if self.queries is not None:
            kwargs["queries"] = self.queries

        return kwargs

    def dedup(self) -> tuple[list[int], PromptCollection]:
        """Deduplicate prompts based on packed_prompts.

        Returns:
            Tuple of (remap_idxs, deduped_collection)
            remap_idxs maps from original index to deduped index
        """
        # Find unique prompts
        idcs_to_keep = []
        seen_prompts = set()
        for i, packed_prompt in enumerate(self.packed_prompts):
            if packed_prompt not in seen_prompts:
                idcs_to_keep.append(i)
                seen_prompts.add(packed_prompt)

        # Create deduped collection
        deduped_packed_prompts = [self.packed_prompts[i] for i in idcs_to_keep]
        deduped_prompts = [self.prompts[i] for i in idcs_to_keep]

        # Create remap indices
        remap_idxs = [deduped_packed_prompts.index(p) for p in self.packed_prompts]

        # Dedup other fields
        deduped_instructions = (
            [self.instructions[i] for i in idcs_to_keep]
            if self.instructions is not None
            else None
        )
        deduped_force_prefill = (
            [self.force_prefill[i] for i in idcs_to_keep]
            if self.force_prefill is not None
            else None
        )
        deduped_images = (
            [self.images[i] for i in idcs_to_keep] if self.images is not None else None
        )
        deduped_queries = (
            [self.queries[i] for i in idcs_to_keep]
            if self.queries is not None
            else None
        )

        return remap_idxs, PromptCollection(
            packed_prompts=deduped_packed_prompts,
            prompts=deduped_prompts,
            instructions=deduped_instructions,
            force_prefill=deduped_force_prefill,
            images=deduped_images,
            queries=deduped_queries,
        )


def format_prompts(
    inputs: dict[str, list],
    config: InferenceConfig,
    verbose: bool = False,
) -> PromptCollection:
    """Format prompts from inputs and config.

    Args:
        inputs: Dictionary of input lists (e.g., {"documents": [...], "criterion": "word_count"})
            Input keys are automatically aliased to singular forms for template use:
            - "documents" → "document"
            - "criteria" → "criterion"
        config: InferenceConfig with prompt templates
        verbose: Whether to log verbose output

    Returns:
        PromptCollection with formatted prompts
    """
    # Determine the length - find first non-singleton list
    n_items = None
    for _key, values in inputs.items():
        if isinstance(values, list) and len(values) > 1:
            n_items = len(values)
            break
    if n_items is None:
        # All are singletons or there's only one input
        n_items = 1 if not inputs else len(next(iter(inputs.values())))

    # Broadcast singleton inputs to match n_items
    broadcasted_inputs = {}
    for key, values in inputs.items():
        if not isinstance(values, list):
            values = [values]
        if len(values) == 1 and n_items > 1:
            broadcasted_inputs[key] = values * n_items
        else:
            broadcasted_inputs[key] = values
            assert (
                len(values) == n_items
            ), f"Input {key} has length {len(values)}, expected {n_items}"

    # Input key aliasing convention:
    # Users pass plural keys (documents, criteria, etc.) which is more natural,
    # but templates use singular forms (document, criterion, etc.) which is more readable.
    # We add both to format_kwargs so both conventions work in templates.
    INPUT_KEY_ALIASES = {
        "documents": "document",
        "criteria": "criterion",
    }

    format_kwargs = {}
    for key, values in broadcasted_inputs.items():
        format_kwargs[key] = values
        # Add singular alias if this key has one
        if key in INPUT_KEY_ALIASES:
            format_kwargs[INPUT_KEY_ALIASES[key]] = values

    # Load prompt template (from file if path, inline if string)
    prompt_template = read_or_return(config.prompt_template)

    # Format base prompts
    prompts = []
    for i in range(n_items):
        kwargs_i = {k: v[i] for k, v in format_kwargs.items()}
        prompt = _safe_format_template(prompt_template, kwargs_i)
        prompts.append(prompt)

    # Format embedding instructions (if specified)
    instructions = None
    if config.instruction is not None:
        instruction_template = read_or_return(config.instruction)
        instructions = []
        for i in range(n_items):
            kwargs_i = {k: v[i] for k, v in format_kwargs.items()}
            instr = _safe_format_template(instruction_template, kwargs_i)
            instructions.append(instr)

    # Format force prefills (if specified)
    force_prefill = None
    if config.force_prefill_template is not None:
        force_prefill_template = read_or_return(config.force_prefill_template)
        force_prefill = []
        for i in range(n_items):
            kwargs_i = {k: v[i] for k, v in format_kwargs.items()}
            prefill = _safe_format_template(force_prefill_template, kwargs_i)
            force_prefill.append(prefill)

    # Handle images (if present)
    images = broadcasted_inputs.get("images")

    # Handle queries (if present) - for providers like Voyage that need separate query/doc
    queries = broadcasted_inputs.get("query")

    # Create packed prompts for caching
    # Packed prompts include all components to create unique cache keys
    # CRITICAL: Must include query for rerankers to avoid cache collisions
    packed_prompts = [
        _build_packed_prompt(
            base_prompt=prompts[i],
            instruction=instructions[i] if instructions else None,
            image=images[i] if images else None,
            force_prefill=force_prefill[i] if force_prefill else None,
            query=queries[i] if queries else None,
        )
        for i in range(n_items)
    ]

    # Debug: Check if images are being included in packed prompts
    if images:
        sample_image_sigs = []
        for i in range(min(3, len(packed_prompts))):
            if "<image_signatures>" in packed_prompts[i]:
                start = packed_prompts[i].index("<image_signatures>") + len(
                    "<image_signatures>"
                )
                end = packed_prompts[i].index("</image_signatures>")
                sample_image_sigs.append(
                    packed_prompts[i][start:end][:100]
                )  # First 100 chars
        if sample_image_sigs:
            logger.debug(
                f"Image signatures in packed prompts (first 3): {sample_image_sigs}"
            )
        else:
            logger.warning(
                "Images provided but NO image signatures found in packed prompts!"
            )

    if verbose and len(prompts) > 0:
        logger.debug(f"Example formatted prompt:\n{prompts[0]}")
        logger.debug(f"Example packed prompt:\n{packed_prompts[0]}")

    return PromptCollection(
        packed_prompts=packed_prompts,
        prompts=prompts,
        instructions=instructions,
        force_prefill=force_prefill,
        images=images,
        queries=queries,
    )
