"""HuggingFace API provider for embeddings and chat completions.

Uses the HuggingFace Inference API for embedding models and the OpenAI-compatible
router for chat completions (e.g., GLM-4.6V-Flash vision model).
"""

import logging
import time

from tqdm import tqdm

from multiview.constants import HF_API_KEY

logger = logging.getLogger(__name__)


def hf_embedding_completions(
    prompts: list[str],
    model_name: str,
    **kwargs,
) -> dict:
    """Get embeddings from HuggingFace Inference API.

    Args:
        prompts: List of texts to embed (without instructions)
        model_name: Model name on HuggingFace Hub (e.g., "Qwen/Qwen2.5-Embedding-7B")
        **kwargs: Additional parameters including:
            - instructions: Optional list of instructions to prepend

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "vector" key with the embedding
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise ImportError(
            "huggingface_hub package required. Install with: pip install huggingface_hub"
        ) from None

    # Initialize client with provider="auto" to avoid StopIteration bug
    api_key = HF_API_KEY
    if not api_key:
        logger.warning(
            "HF_TOKEN environment variable not set. Using public API (rate limited)."
        )

    client = (
        InferenceClient(provider="auto", api_key=api_key)
        if api_key
        else InferenceClient(provider="auto")
    )

    # Handle embedding instructions by prepending them to prompts
    instructions = kwargs.pop("instructions", None)

    logger.debug(f"instructions: {instructions}")
    logger.debug(f"prompts (before prepending): {prompts[:4]}")

    final_prompts = []
    for i, prompt in enumerate(prompts):
        final_prompt = prompt

        # Format instructions using the proper instruction-tuned format
        # See: https://huggingface.co/Qwen/Qwen3-Embedding-8B
        # Format: "Instruct: {task_description}\nQuery: {text}"
        if instructions and i < len(instructions) and instructions[i]:
            final_prompt = f"Instruct: {instructions[i]}\nQuery: {final_prompt}"

        final_prompts.append(final_prompt)

    logger.debug(f"final_prompts (after prepending): {final_prompts[:4]}")

    # Validate instructions were applied correctly
    if instructions:
        for i, (original, final) in enumerate(
            zip(prompts, final_prompts, strict=False)
        ):
            if instructions and i < len(instructions) and instructions[i]:
                assert final != original, f"Instruction not applied to prompt {i}"
                assert final.startswith(
                    "Instruct: "
                ), f"Instruction format incorrect for prompt {i}"
                assert instructions[i] in final, f"Instruction not in final prompt {i}"

    # Batch embeddings for efficiency (process all prompts at once)
    try:
        # Call feature extraction endpoint with batch
        result = client.feature_extraction(
            final_prompts,  # Pass list directly, not text= parameter
            model=model_name,
        )

        # Handle different response formats
        import numpy as np

        if isinstance(result, np.ndarray):
            embeddings = result
        elif isinstance(result, list):
            embeddings = np.array(result)
        else:
            raise ValueError(f"Unexpected response type: {type(result)}")

        # Handle nested arrays (some models return [batch, 1, dim])
        if len(embeddings.shape) == 3:
            embeddings = embeddings.squeeze(1)

        # Convert to list of dicts with Python floats (not numpy types)
        completions = [{"vector": [float(x) for x in emb]} for emb in embeddings]

    except StopIteration:
        logger.error(
            f"StopIteration error from HuggingFace API. This is likely caused by an unsupported model: {model_name}. "
            f"Check that the model is available via HF Inference API at https://huggingface.co/{model_name}"
        )
        # Return empty vectors
        completions = [{"vector": []} for _ in final_prompts]
    except Exception as e:
        logger.warning(f"Error getting embeddings: {e}. Retrying after 5s...")
        time.sleep(5)
        try:
            result = client.feature_extraction(
                final_prompts,
                model=model_name,
            )
            import numpy as np

            if isinstance(result, np.ndarray):
                embeddings = result
            elif isinstance(result, list):
                embeddings = np.array(result)
            else:
                embeddings = np.array([[]] * len(final_prompts))

            if len(embeddings.shape) == 3:
                embeddings = embeddings.squeeze(1)

            completions = [{"vector": [float(x) for x in emb]} for emb in embeddings]
        except Exception as e2:
            logger.error(f"Error getting embeddings after retry: {e2}")
            # Return empty vectors on error
            completions = [{"vector": []} for _ in final_prompts]

    return {"completions": completions}


def hf_chat_completions(
    prompts: list[str],
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    images: list[str | list[str] | None] | None = None,
    max_workers: int = 5,
    **kwargs,
) -> dict:
    """Get chat completions from HuggingFace via OpenAI-compatible router.

    Uses the HF router (https://router.huggingface.co/v1) with OpenAI client.
    Supports multi-image prompts with <image> placeholder markers.

    Args:
        prompts: List of prompts
        model_name: Model name on HuggingFace (e.g., "zai-org/GLM-4.6V-Flash")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        images: Optional list of image sources for vision tasks. Each element can be:
                - str: Single image (URL or file path)
                - list[str]: Multiple images for multi-image prompts
                - None: No images for text-only prompts
        max_workers: Maximum concurrent API requests (default 5)
        **kwargs: Additional API parameters

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "text" key with the generated text
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required. Install with: pip install openai"
        ) from None

    api_key = HF_API_KEY
    if not api_key:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Get your token at https://huggingface.co/settings/tokens"
        )

    # Initialize OpenAI client with HF router
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )

    # Strip internal/provider-level arguments that must not be forwarded to the API
    kwargs.pop("max_retries", None)
    kwargs.pop("initial_retry_delay", None)
    kwargs.pop("retry_backoff_factor", None)
    # max_workers is used by this function, so pop it separately
    max_workers_param = kwargs.pop("max_workers", max_workers)

    # Process prompts in parallel
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    process_fn = partial(
        _hf_single_chat_completion,
        client=client,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    # Prepare inputs for processing
    inputs = []
    for i, prompt in enumerate(prompts):
        image = None
        if images and i < len(images):
            image = images[i]
        inputs.append((prompt, image))

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers_param) as executor:
        completions = list(
            tqdm(
                executor.map(lambda x: process_fn(prompt=x[0], image=x[1]), inputs),
                total=len(inputs),
                desc=f"HF chat ({model_name})",
            )
        )

    return {"completions": completions}


def _hf_single_chat_completion(
    client,
    prompt: str,
    image: str | list[str] | None,
    model_name: str,
    temperature: float,
    max_tokens: int,
    **kwargs,
) -> dict:
    """Process a single HF chat completion with vision support.

    Args:
        client: OpenAI client instance (configured for HF router)
        prompt: Prompt text (may contain <image> placeholders)
        image: Optional image source(s) for vision tasks. Can be:
               - Single image: str (URL or file path)
               - Multiple images: list[str] (for multi-image prompts)
               - None for text-only prompts
        model_name: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional API parameters

    Returns:
        Dict with "text" key containing the completion
    """
    # Build content array
    # Support both single image (str) and multiple images (list)
    images_to_process = [image] if isinstance(image, str) else (image if image else [])

    # Convert images to base64 data URIs for HuggingFace models
    # Import VLM utilities for image handling
    from multiview.inference.vlm_utils import prepare_image_for_openai

    try:
        # Convert all valid images to base64 data URIs
        image_data_uris = []
        failed_images = []

        for img in images_to_process:
            if not img:  # Skip None values
                continue

            try:
                data_uri = prepare_image_for_openai(img)
                image_data_uris.append(data_uri)
                logger.debug(f"Prepared image for HF: {img}")
            except Exception as img_error:
                logger.warning(f"Failed to load image {img}: {img_error}")
                failed_images.append(img)
                # Continue with other images

        if not image_data_uris and images_to_process and any(images_to_process):
            # All images failed, fall back to text-only
            logger.warning(
                f"No valid images to process. Failed: {len(failed_images)}/{len(images_to_process)}"
            )
            images_to_process = []

        # Build content array with text and images
        # If prompt contains <image> markers, interleave images at those locations
        # Otherwise, put all images at the start
        content = []

        if image_data_uris and "<image>" in prompt:
            # Split text by <image> markers and interleave with images
            text_parts = prompt.split("<image>")

            # Interleave text and images
            for i, text_part in enumerate(text_parts):
                if text_part:  # Add non-empty text
                    content.append({"type": "text", "text": text_part})
                if i < len(image_data_uris):  # Add image if available
                    content.append(
                        {"type": "image_url", "image_url": {"url": image_data_uris[i]}}
                    )

            # If there are more images than markers, add remaining images at end
            if len(image_data_uris) > len(text_parts) - 1:
                for remaining_img in image_data_uris[len(text_parts) - 1 :]:
                    content.append(
                        {"type": "image_url", "image_url": {"url": remaining_img}}
                    )
        elif image_data_uris:
            # Default: all images at start, then text
            for img_uri in image_data_uris:
                content.append({"type": "image_url", "image_url": {"url": img_uri}})
            content.append({"type": "text", "text": prompt})
        else:
            # Text-only
            content.append({"type": "text", "text": prompt})

    except Exception as e:
        logger.warning(f"Failed to process images, falling back to text-only: {e}")
        # Fall back to text-only
        content = [{"type": "text", "text": prompt}]

    # Call chat completions API
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        message = response.choices[0].message
        completion_text = message.content

        # Some models (like GLM-4.6V-Flash) use reasoning_content for actual output
        # when they're reasoning models (similar to OpenAI's o1)
        if not completion_text and hasattr(message, "reasoning_content"):
            completion_text = message.reasoning_content
            logger.debug("Using reasoning_content instead of content")

        return {"text": completion_text}

    except Exception as e:
        logger.error(f"Error getting HF chat completion: {e}")
        return {"text": ""}
