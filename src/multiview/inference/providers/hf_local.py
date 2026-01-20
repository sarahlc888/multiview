"""HuggingFace local transformers provider for reranker models and hidden states.

This provider loads models locally using transformers library (not via API).
Supports:
- Reranker models with custom tokenization
- Hidden state extraction for embedding-like representations
"""

import logging

from multiview.constants import HF_CACHE_DIR

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}


def hf_local_reranker_completions(
    prompts: list[str],
    model_name: str,
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 8192,
    **kwargs,
) -> dict:
    """Get reranker scores from local HuggingFace model.

    Args:
        prompts: List of formatted prompts (query-document pairs already formatted)
        model_name: Model name on HuggingFace Hub (e.g., "Qwen/Qwen3-Reranker-8B")
        device: Device to run model on ("cuda" or "cpu")
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        **kwargs: Additional parameters including:
            - use_fp16: Whether to use FP16 precision (default: True if cuda)
            - instruction: Custom instruction (already included in prompts)

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "score" key with the relevance score (0-1)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers and torch packages required. "
            "Install with: pip install transformers torch"
        ) from e

    use_fp16 = kwargs.get("use_fp16", device == "cuda")

    logger.info("=" * 70)
    logger.info(f"Loading reranker model: {model_name}")
    logger.info(f"Cache directory: {HF_CACHE_DIR}")
    logger.info("=" * 70)

    # Load tokenizer with left padding (required for batch inference)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(HF_CACHE_DIR),
        padding_side="left",
    )
    logger.info("✓ Tokenizer loaded")

    # Configure model loading based on settings
    model_kwargs = {"cache_dir": str(HF_CACHE_DIR)}
    if use_fp16 and device == "cuda":
        model_kwargs["dtype"] = torch.float16

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info("✓ Model loaded")

    if device == "cuda":
        model = model.cuda()

    model = model.eval()

    # Define prefix and suffix tokens (as per official Qwen3-Reranker usage)
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    # Encode prefix and suffix tokens
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    # Get token IDs for "yes" (true) and "no" (false)
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    def process_inputs(batch_prompts: list[str]) -> dict:
        """Tokenize inputs with prefix/suffix tokens following official usage."""
        # First tokenize without padding
        inputs = tokenizer(
            batch_prompts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )

        # Add prefix and suffix tokens to each sequence
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i] = (
                prefix_tokens + inputs["input_ids"][i] + suffix_tokens
            )

        # Now pad the batch
        inputs = tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=max_length
        )

        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)

        return inputs

    # Process in batches
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize with proper prefix/suffix
            inputs = process_inputs(batch_prompts)

            # Get model outputs
            outputs = model(**inputs)

            # Extract logits for the last token
            batch_scores = outputs.logits[:, -1, :]

            # Stack true/false token logits
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores_stacked = torch.stack([false_vector, true_vector], dim=1)

            # Apply log_softmax and get probabilities for "yes" token
            batch_scores_log = torch.nn.functional.log_softmax(
                batch_scores_stacked, dim=1
            )
            scores = batch_scores_log[:, 1].exp().cpu().tolist()

            all_scores.extend(scores)

    logger.info(f"Computed {len(all_scores)} reranker scores")

    # Format as completions (similar to embedding provider)
    completions = [{"score": float(score)} for score in all_scores]

    return {"completions": completions}


def hf_local_contextual_reranker_completions(
    prompts: list[str],
    model_name: str,
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 8192,
    **kwargs,
) -> dict:
    """Get reranker scores from local HuggingFace model (ContextualAI-style).

    This function is designed for rerankers that output scores encoded as BF16
    values in the first token logits (e.g., ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b).

    Args:
        prompts: List of formatted prompts (query-document pairs already formatted)
        model_name: Model name on HuggingFace Hub
        device: Device to run model on ("cuda" or "cpu")
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        **kwargs: Additional parameters including:
            - use_fp16: Whether to use FP16 precision (ignored, uses BF16)

    Returns:
        Dict with "completions" key containing list of completion dicts.
        Each completion dict has "score" key with the relevance score.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers and torch packages required. "
            "Install with: pip install transformers torch"
        ) from e

    # ContextualAI models use BF16 for score encoding
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    logger.info("=" * 70)
    logger.info(f"Loading ContextualAI reranker model: {model_name}")
    logger.info(f"Cache directory: {HF_CACHE_DIR}")
    logger.info("=" * 70)

    # Load tokenizer with left padding (required for batch inference)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(HF_CACHE_DIR),
        use_fast=True,
    )
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # so -1 is the real last token for all prompts
    logger.info("✓ Tokenizer loaded")

    # Load model with BF16
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(HF_CACHE_DIR),
        torch_dtype=dtype,
    ).to(device)
    logger.info("✓ Model loaded")

    model = model.eval()

    # Process in batches
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize with left padding
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Get model outputs
            out = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract logits for the last token
            next_logits = out.logits[:, -1, :]  # [batch, vocab]

            # Extract scores from first token logits (ContextualAI method)
            scores_bf16 = next_logits[:, 0].to(torch.bfloat16)
            scores = scores_bf16.float().cpu().tolist()

            all_scores.extend(scores)

    logger.info(f"Computed {len(all_scores)} reranker scores")

    # Format as completions
    completions = [{"score": float(score)} for score in all_scores]

    return {"completions": completions}


def hf_local_hidden_state_completions(
    prompts: list[str],
    model_name: str,
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 2048,
    hidden_layer_idx: int = -1,
    is_chatml_prompt: bool = False,
    force_prefills: str | None = None,
    **kwargs,
) -> dict:
    """Get hidden state embeddings from local HuggingFace causal LM.

    This function extracts hidden states from a causal language model,
    suitable for use as embeddings in similarity tasks.

    Args:
        prompts: List of text prompts to encode
        model_name: Model name on HuggingFace Hub (e.g., "Qwen/Qwen3-8B")
        device: Device to run model on ("cuda" or "cpu")
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        hidden_layer_idx: Which layer to extract (-1 = last layer)
        is_chatml_prompt: Whether to apply chat template to prompts
        force_prefills: Optional text to append after chat template
        **kwargs: Additional parameters including:
            - use_fp16: Whether to use FP16 precision (default: True if cuda)

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "vector" key with the hidden state vector
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers, torch, and numpy packages required. "
            "Install with: pip install transformers torch numpy"
        ) from e

    use_fp16 = kwargs.get("use_fp16", device == "cuda")

    # Check cache for model and tokenizer
    cache_key = f"{model_name}_{device}_{use_fp16}"

    if cache_key not in _MODEL_CACHE:
        logger.info("=" * 70)
        logger.info(f"Loading hidden state model: {model_name}")
        logger.info(f"Cache directory: {HF_CACHE_DIR}")
        logger.info("=" * 70)

        # Load tokenizer with left padding (for batch inference)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(HF_CACHE_DIR),
            padding_side="left",
        )

        # Set pad token if not set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Tokenizer loaded")

        # Configure model loading
        model_kwargs = {"cache_dir": str(HF_CACHE_DIR)}
        if use_fp16 and device == "cuda":
            model_kwargs["dtype"] = torch.float16

        logger.info(f"Loading model with config: {model_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info("✓ Model loaded")

        if device == "cuda":
            model = model.cuda()

        model = model.eval()

        # Set pad token in model config
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id

        # Cache for future use
        _MODEL_CACHE[cache_key] = model
        _TOKENIZER_CACHE[cache_key] = tokenizer
    else:
        logger.info(f"Using cached model: {model_name}")
        model = _MODEL_CACHE[cache_key]
        tokenizer = _TOKENIZER_CACHE[cache_key]

    # Process prompts - apply chat template if needed
    processed_prompts = list(prompts)  # Make a copy
    if is_chatml_prompt:
        processed_prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in processed_prompts
        ]

    # Add force_prefills if specified
    if force_prefills:
        assert len(force_prefills) == len(processed_prompts)
        processed_prompts = [
            p + force_prefill
            for p, force_prefill in zip(processed_prompts, force_prefills, strict=True)
        ]

    # Process in batches
    all_vectors = []

    with torch.no_grad():
        for i in range(0, len(processed_prompts), batch_size):
            batch_prompts = processed_prompts[i : i + batch_size]

            # Tokenize with left padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,  # Already added by chat template if needed
            )

            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Get model outputs with hidden states
            outputs = model(**inputs, output_hidden_states=True)

            # Extract hidden states from specified layer at last token position
            # Shape: (batch_size, seq_len, hidden_size)
            hidden_states = outputs.hidden_states[hidden_layer_idx]

            # Extract last token's hidden state for each sequence
            # With left padding, last token is always at index -1
            batch_vectors = hidden_states[:, -1, :]  # Shape: (batch_size, hidden_size)

            # Convert to numpy and store
            batch_vectors_np = batch_vectors.cpu().float().numpy()
            all_vectors.extend(batch_vectors_np)

    logger.info(
        f"Extracted {len(all_vectors)} hidden state vectors "
        f"(dim={all_vectors[0].shape[0] if all_vectors else 'N/A'})"
    )

    # Format as completions (list of dicts with "vector" key)
    completions = [{"vector": vec.tolist()} for vec in all_vectors]

    return {"completions": completions}
