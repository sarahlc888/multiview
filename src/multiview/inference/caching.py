"""Caching utilities for inference.

Handles caching of LM/embedding model completions to disk.

Improvements over old version:
- File locking to prevent race conditions
- Hashed keys for packed prompts (full prompts stored in values for debugging)
- Thread-safe and process-safe concurrent writes
"""

import hashlib
import json
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl
    HAS_FCNTL = False
    try:
        import msvcrt

        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False

logger = logging.getLogger(__name__)


@contextmanager
def file_lock(file_path: str):
    """Context manager for file locking (cross-platform).

    Args:
        file_path: Path to lock file

    Yields:
        Lock file handle
    """
    lock_path = f"{file_path}.lock"
    lock_file = open(lock_path, "w")

    try:
        if HAS_FCNTL:
            # Unix/Linux/Mac - use fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif HAS_MSVCRT:
            # Windows - use msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        # If neither available, proceed without locking (best effort)

        yield lock_file
    finally:
        if HAS_FCNTL:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        elif HAS_MSVCRT:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

        lock_file.close()
        try:
            os.remove(lock_path)
        except OSError:
            pass


def hash_prompt(prompt: str) -> str:
    """Hash a prompt to use as cache key.

    Args:
        prompt: Prompt string to hash

    Returns:
        Hash string (SHA256, truncated to 32 chars)
    """
    return hashlib.sha256(prompt.encode()).hexdigest()[:32]


def load_cached_completions(cache_path: str | None) -> dict:
    """Load cached completions from disk.

    Cache format:
    {
        "completions": {
            "<prompt_hash>": {
                "result": <completion_result>,
                "prompt": "<original_packed_prompt>"  # for debugging
            },
            ...
        }
    }

    Args:
        cache_path: Path to cache file (JSON format)

    Returns:
        Dictionary of cached completions (empty dict if file doesn't exist)
    """
    if cache_path is None or not Path(cache_path).exists():
        return {}

    try:
        with file_lock(cache_path):
            with open(cache_path) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return {}


def save_cached_completions(
    cache: dict, cache_path: str | None, verbose: bool = False
) -> None:
    """Save cached completions to disk with file locking.

    Uses file locking to prevent race conditions from concurrent processes.
    Refreshes cache from disk before saving to preserve concurrent writes.

    Args:
        cache: Dictionary of completions to save
        cache_path: Path to cache file (JSON format)
        verbose: Whether to log verbose output
    """
    if cache_path is None:
        return

    # Create parent directory if needed
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with file_lock(cache_path):
        # Refresh cache from disk and merge (within lock)
        try:
            with open(cache_path) as f:
                existing_cache = json.load(f)
                # Merge: existing cache gets updated with new cache
                # This preserves entries from concurrent processes
                for field_name in cache:
                    if field_name not in existing_cache:
                        existing_cache[field_name] = {}
                    existing_cache[field_name].update(cache[field_name])
                cache = existing_cache
        except FileNotFoundError:
            pass  # New cache file
        except Exception as e:
            logger.warning(f"Failed to refresh cache from {cache_path}: {e}")

        # Write atomically: temp file + rename
        temp_path = f"{cache_path}.tmp.{os.getpid()}"
        try:
            with open(temp_path, "w") as f:
                json.dump(cache, f, indent=2)
            os.replace(temp_path, cache_path)  # os.replace is atomic

            if verbose:
                total_entries = sum(
                    len(v) for v in cache.values() if isinstance(v, dict)
                )
                logger.info(f"Saved {total_entries} cached entries to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


def get_cache_hash(config_dict: dict, cache_alias: str) -> str:
    """Generate a cache hash for the given config.

    Args:
        config_dict: Configuration dictionary to hash
        cache_alias: Human-readable alias to include in hash

    Returns:
        Cache hash string in format: {alias}__{hash}
    """
    # Create a simplified dict for hashing
    hash_dict = {
        "cache_alias": cache_alias,
        "config": config_dict,
    }

    # Sort keys for consistent hashing
    hash_str = json.dumps(hash_dict, sort_keys=True)
    hash_hex = hashlib.sha256(hash_str.encode()).hexdigest()

    return f"{cache_alias}__{hash_hex[:16]}"


def cached_fn_completions(
    packed_prompts: list[str],
    non_packed_prompts: list[str],
    fn_completions: Callable,
    completion_cache: dict,
    completion_cache_path: str | None = None,
    completion_field_name: str = "completions",
    force_refresh: bool = False,
    verbose: bool = False,
    return_type: str = "list",
    mute_if_cache_hit: bool = False,
    **fn_completions_kwargs,
) -> list:
    """Wrap fn_completions to cache the completion for each packed_prompt.

    This is the main caching function ported from the old repo.
    Uses hashed prompts as keys to reduce cache file size.

    Args:
        packed_prompts: Prompts with all components (used as cache keys)
        non_packed_prompts: Base prompts without instructions (passed to provider)
        fn_completions: Completion function to call for uncached prompts
        completion_cache: Cache dictionary (will be modified in-place)
        completion_cache_path: Path to cache file on disk
        completion_field_name: Key to use in cache for this completion type
        force_refresh: If True, ignore cache and recompute all
        verbose: Whether to log verbose output
        return_type: "list" or "dict" - format of return value
        mute_if_cache_hit: If True, don't log when all prompts are cached
        **fn_completions_kwargs: Additional kwargs to pass to fn_completions
            (e.g., force_prefills, embed_query_instrs, embed_doc_instrs, images)

    Returns:
        List of completions (or dict with completion_field_name key)
    """
    # Initialize cache field if needed
    if completion_field_name not in completion_cache:
        completion_cache[completion_field_name] = {}

    if verbose and len(packed_prompts) > 0:
        logger.info(f"Example packed prompt:\n{packed_prompts[0]}")

    # Hash prompts for cache keys
    prompt_hashes = [hash_prompt(p) for p in packed_prompts]

    # Determine which prompts need to be computed
    if force_refresh:
        logger.info(f"Force refresh: recomputing all {len(packed_prompts)} prompts")
        uncached_prompts = packed_prompts
        uncached_hashes = prompt_hashes
        uncached_indices = list(range(len(packed_prompts)))
        uncached_non_packed = non_packed_prompts
    else:
        # Find prompts not in cache (or with empty values)
        def is_non_empty(val):
            if isinstance(val, str):
                return len(val.strip()) > 0
            return True

        uncached_indices = [
            i
            for i, prompt_hash in enumerate(prompt_hashes)
            if prompt_hash not in completion_cache[completion_field_name]
            or not is_non_empty(
                completion_cache[completion_field_name][prompt_hash].get("result")
            )
        ]
        uncached_prompts = [packed_prompts[i] for i in uncached_indices]
        uncached_hashes = [prompt_hashes[i] for i in uncached_indices]
        uncached_non_packed = [non_packed_prompts[i] for i in uncached_indices]

        # Filter kwargs arrays to match uncached prompts
        for key in [
            "force_prefills",
            "embed_query_instrs",
            "embed_doc_instrs",
            "images",
        ]:
            if key in fn_completions_kwargs:
                fn_completions_kwargs[key] = [
                    fn_completions_kwargs[key][i] for i in uncached_indices
                ]

    # Log cache status
    if verbose or len(uncached_prompts) > 0:
        if not (mute_if_cache_hit and len(uncached_prompts) == 0):
            logger.info(
                f"Running {len(uncached_prompts)} uncached completions "
                f"(out of {len(packed_prompts)} total) @ {completion_cache_path}"
            )

    # Run completions for uncached prompts
    if len(uncached_prompts) > 0:
        # Use non-packed prompts (base prompts without instructions/prefills)
        # Instructions and prefills are passed separately via fn_completions_kwargs
        # Packed prompts are only used as cache keys
        uncached_completions = fn_completions(
            prompts=uncached_non_packed,
            **fn_completions_kwargs,
        )

        # Validate output
        assert len(uncached_non_packed) == len(
            uncached_completions[completion_field_name]
        ), (
            f"Length mismatch: {len(uncached_non_packed)} prompts but "
            f"{len(uncached_completions[completion_field_name])} completions"
        )

        # Update cache with hashed keys and full prompts for debugging
        for prompt, prompt_hash, result in zip(
            uncached_prompts,
            uncached_hashes,
            uncached_completions[completion_field_name],
            strict=False,
        ):
            completion_cache[completion_field_name][prompt_hash] = {
                "result": result,
                "prompt": prompt,  # Store full prompt for debugging
            }

    # Validate cache has all prompts
    if return_type == "dict":
        for prompt_hash in prompt_hashes:
            assert (
                prompt_hash in completion_cache[completion_field_name]
            ), f"Prompt hash not in cache: {prompt_hash}"

    # Save cache to disk if we added new entries
    if len(uncached_prompts) > 0 and completion_cache_path is not None:
        save_cached_completions(
            completion_cache, completion_cache_path, verbose=verbose
        )

    # Return completions in requested format
    if return_type == "list":
        return [
            completion_cache[completion_field_name][h]["result"] for h in prompt_hashes
        ]
    elif return_type == "dict":
        return {
            completion_field_name: [
                completion_cache[completion_field_name][h]["result"]
                for h in prompt_hashes
            ]
        }
    else:
        raise ValueError(f"Unsupported return_type: {return_type}")
