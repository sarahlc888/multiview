#!/usr/bin/env python3
"""

Generate a Web app that has a large set of documents, and if you type in the
instruction, it will re-embedded and re-represent everything. So you should
be able to represent any corpus with any criteria.

---
Visualize document corpus embeddings in 2D or generate GSM8K computational graphs.

NOTE: This script is for exploring NEW UNLABELED CORPORA (no triplets).
      For visualizing BENCHMARK EVALUATION RESULTS, enable auto_visualize
      in your benchmark config (automatic with run_eval.py).

This script has three modes:
1. Raw corpus exploration: Visualize unlabeled datasets for investigation
2. Benchmark mode: Visualize evaluation results (auto_visualize is preferred)
3. GSM8K graphs: Generate standalone computational graph visualizations

Usage examples:
    # ============================================================
    # RAW CORPUS EXPLORATION (investigating new unlabeled data)
    # ============================================================

    # Basic t-SNE visualization of a raw corpus
    python scripts/analyze_corpus.py \\
        --dataset gsm8k \\
        --embedding-preset hf_qwen3_embedding_8b \\
        --reducer tsne \\
        --output outputs/viz/gsm8k_tsne \\
        --max-docs 500

    # Visualize from custom JSONL file
    python scripts/analyze_corpus.py \\
        --input-jsonl outputs/my_documents.jsonl \\
        --embedding-preset hf_qwen3_embedding_8b \\
        --reducer tsne \\
        --output outputs/viz/my_viz

    # In-one-word embeddings
    python scripts/analyze_corpus.py \\
        --dataset gsm8k \\
        --embedding-preset inoneword_hf_qwen3_4b \\
        --criterion arithmetic \\
        --in-one-word-context "Categories: addition, subtraction, multiplication" \\
        --reducer tsne \\
        --output outputs/viz/gsm8k_inoneword \\
        --max-docs 100

    # Thumbnails as markers
    python scripts/analyze_corpus.py \\
        --dataset gsm8k \\
        --reducer tsne \\
        --marker-type thumbnail \\
        --image-zoom 0.08 \\
        --output outputs/viz/gsm8k_graphs \\
        --max-docs 100

    # Dendrogram with image thumbnails (hierarchical clustering)
    python scripts/analyze_corpus.py \\
        --dataset gsm8k \\
        --embedding-preset hf_qwen3_embedding_8b \\
        --reducer dendrogram \\
        --dendrogram-method average \\
        --marker-type thumbnail \\
        --output outputs/viz/gsm8k_dendrogram \\
        --max-docs 50 \\
        --figsize 24,12

    # Dendrogram with grid layout and cluster coloring
    python scripts/analyze_corpus.py \\
        --dataset gsm8k \\
        --embedding-preset hf_qwen3_embedding_8b \\
        --reducer dendrogram \\
        --dendrogram-method average \\
        --marker-type thumbnail \\
        --dendrogram-images-per-row 15 \\
        --dendrogram-num-clusters 8 \\
        --output outputs/viz/gsm8k_dendrogram_grid \\
        --max-docs 100 \\
        --figsize 24,16

    # ============================================================
    # BENCHMARK MODE (auto_visualize in config is preferred!)
    # ============================================================

    # Visualize benchmark results (after run_eval.py)
    # Better: Enable auto_visualize in your benchmark config
    # This mode is mainly for one-off custom visualizations
    python scripts/analyze_corpus.py \\
        --from-benchmark benchmark_debug \\
        --task gsm8k__arithmetic \\
        --method qwen3_8b_with_instructions \\
        --reducer tsne \\
        --output outputs/viz/gsm8k_arithmetic \\
        --export-format web

    # ============================================================
    # GSM8K GRAPH MODE (standalone graph generation)
    # ============================================================

    # Generate computational graph visualizations
    python scripts/analyze_corpus.py \\
        --mode gsm8k_graphs \\
        --input documents.jsonl \\
        --output outputs/viz/gsm8k \\
        --num 5
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: I001

# NOTE: This script is runnable directly; we modify sys.path above so these
# imports work without installing the package. Ruff flags E402 for this pattern.
from multiview.docsets import DOCSETS  # noqa: E402
from multiview.inference import run_inference  # noqa: E402
from multiview.utils.logging_utils import setup_logging  # noqa: E402
from multiview.utils.prompt_utils import read_or_return  # noqa: E402
from multiview.visualization import (  # noqa: E402
    CorpusVisualizer,
    DendrogramReducer,
    PCAReducer,
    SOMReducer,
    TSNEReducer,
    UMAPReducer,
    create_gsm8k_marker_images,
)
from multiview.visualization.gsm8k_graph import (  # noqa: E402
    GSM8KComputationalGraph,
    parse_gsm8k_document,
)

# Set up logger
logger = logging.getLogger(__name__)


def load_documents_from_jsonl(
    jsonl_path: str,
    max_docs: int | None = None,
    text_field: str = "document",
) -> tuple[list[Any], list[str]]:
    """Load documents from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file
        max_docs: Maximum number of documents to load
        text_field: Field name to extract text from (tries multiple common fields)

    Returns:
        Tuple of (documents, doc_texts)
    """
    documents = []
    doc_texts = []

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break

            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            documents.append(doc)

            # Try to extract text from various common field names
            text = None
            for field in [text_field, "document", "text", "content", "body"]:
                if field in doc:
                    text = doc[field]
                    break

            if text is None:
                # If no standard field found, use the whole dict as string
                text = str(doc)

            doc_texts.append(text)

    logger.info(f"Loaded {len(documents)} documents from {jsonl_path}")
    return documents, doc_texts


def load_documents_from_dataset(
    dataset_name: str,
    dataset_config: dict[str, Any] | None = None,
    max_docs: int | None = None,
) -> tuple[list[Any], list[str]]:
    """Load documents from a dataset."""
    if dataset_name not in DOCSETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {', '.join(DOCSETS.keys())}"
        )

    # Get docset class and instantiate
    docset_cls = DOCSETS[dataset_name]
    config = dataset_config or {}

    if max_docs:
        config["max_docs"] = max_docs

    docset = docset_cls(config=config)

    # Load documents
    documents = docset.load_documents()

    # Extract text for embedding
    doc_texts = [docset.get_document_text(d) for d in documents]

    return documents, doc_texts


def find_benchmark_cache_file(task_name: str, method_name: str) -> Path | None:
    """Find cached embeddings for a benchmark method.

    Looks for pattern: {task_name}_eval_{method_name}__{hash}.json
    in INFERENCE_CACHE_DIR.

    Args:
        task_name: Task name (e.g., 'gsm8k__arithmetic')
        method_name: Method name (e.g., 'qwen3_8b_with_instructions')

    Returns:
        Path to cache file if found, None otherwise
    """
    from multiview.constants import INFERENCE_CACHE_DIR

    cache_pattern = f"{task_name}_eval_{method_name}__"
    matching_files = list(INFERENCE_CACHE_DIR.glob(f"{cache_pattern}*.json"))

    if not matching_files:
        logger.warning(f"No cache file found matching {cache_pattern}*.json")
        return None

    if len(matching_files) > 1:
        # Use most recent
        matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        logger.warning(
            f"Multiple cache files found, using most recent: {matching_files[0].name}"
        )

    return matching_files[0]


def load_embeddings_from_cache(cache_file: Path) -> np.ndarray:
    """Load embeddings from benchmark cache file.

    Args:
        cache_file: Path to cache JSON file

    Returns:
        Array of embeddings (n_docs, embedding_dim)
    """
    logger.info(f"Loading embeddings from cache: {cache_file.name}")

    with open(cache_file) as f:
        cache_data = json.load(f)

    if "completions" not in cache_data:
        raise ValueError("Invalid cache format: missing 'completions' key")

    completions = cache_data["completions"]
    logger.info(f"Found {len(completions)} cached embeddings")

    # Extract embeddings - handle both formats
    embeddings = []
    for completion_hash in sorted(completions.keys()):  # Sort for consistency
        result = completions[completion_hash]["result"]

        if isinstance(result, dict) and "vector" in result:
            embeddings.append(result["vector"])
        elif isinstance(result, list):
            embeddings.append(result)
        else:
            raise ValueError(f"Unknown result format: {type(result)}")

    embeddings_array = np.array(embeddings, dtype=np.float32)
    logger.info(
        f"Loaded {len(embeddings_array)} embeddings of dim {embeddings_array.shape[1]}"
    )

    return embeddings_array


def load_from_benchmark(
    run_name: str,
    task_name: str,
    method_name: str,
) -> tuple[list[Any], list[str], str, np.ndarray | None, str]:
    """Load documents and embeddings from benchmark run.

    Args:
        run_name: Benchmark run name (e.g., 'benchmark_debug')
        task_name: Task name (e.g., 'gsm8k__arithmetic__tag__5')
        method_name: Method name (e.g., 'qwen3_8b_with_instructions')

    Returns:
        Tuple of (documents, doc_texts, dataset_name, embeddings, criterion)
        If embeddings are found in outputs dir, returns them directly.
        Otherwise returns None and caller should regenerate from cache.
    """
    from multiview.benchmark.artifacts import (
        load_documents_from_jsonl as load_docs_artifact,
    )

    output_base = Path("outputs") / run_name

    # Load documents
    documents_dir = output_base / "documents"
    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents not found: {documents_dir}\n"
            f"Run 'python scripts/create_eval.py --config-name {run_name}' first"
        )

    documents = load_docs_artifact(output_dir=str(documents_dir), task_name=task_name)

    def _to_text(doc: Any) -> str:
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            text = doc.get("text", "")
            if text:
                return str(text)
            if doc.get("image_path"):
                return "<image>"
            return json.dumps(doc, ensure_ascii=False)
        return str(doc)

    doc_texts = [_to_text(doc) for doc in documents]

    # Try to load embeddings from outputs directory first
    embeddings_dir = output_base / "embeddings" / task_name
    embeddings_file = embeddings_dir / f"{method_name}.npy"

    embeddings = None
    if embeddings_file.exists():
        embeddings = np.load(embeddings_file)

        # Check if this is an NxN similarity matrix (e.g., from BM25) rather than NxD embeddings
        # Similarity matrices can only be used with heatmap visualization, not dimensionality reduction
        is_similarity_matrix = embeddings.ndim == 2 and embeddings.shape[
            0
        ] == embeddings.shape[1] == len(documents)

        if is_similarity_matrix:
            logger.info(
                f"Loaded NxN similarity matrix ({embeddings.shape[0]}x{embeddings.shape[1]}) for {method_name}. "
                f"This method supports heatmap visualization only."
            )
            # Don't raise error - heatmap visualization will handle this correctly
        else:
            # Check for NaN values (common in document rewrite methods that only embed docs in triplets)
            nan_count = np.isnan(embeddings).sum()
            if nan_count > 0:
                # Filter out documents with NaN embeddings
                valid_mask = ~np.isnan(embeddings).any(axis=1)
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) == 0:
                    raise ValueError(
                        f"All embeddings contain NaN values for {task_name} / {method_name}\n"
                        f"This method may not be suitable for visualization."
                    )

                nan_vector_count = (~valid_mask).sum()
                logger.warning(f"Found {nan_vector_count} NaN vectors in embeddings")
                # Filter embeddings and documents
                embeddings = embeddings[valid_mask]
                documents = [documents[i] for i in valid_indices]
                doc_texts = [doc_texts[i] for i in valid_indices]
    else:
        # Embeddings not found - raise helpful error
        raise FileNotFoundError(
            f"No embeddings found for {task_name} / {method_name}\n"
            f"Expected: {embeddings_file}\n"
            f"Run evaluation first: python scripts/run_eval.py --config-name {run_name}"
        )

    # Infer dataset name and criterion from task
    # Task format: dataset__criterion__style__num
    task_parts = task_name.split("__")
    dataset_name = task_parts[0]
    # Criterion is everything between dataset and style (could have multiple parts)
    # e.g., "gsm8k__final_expression__tag__5" -> criterion = "final_expression"
    if len(task_parts) >= 2:
        # Find where the style/num parts start (tag, lm, random, etc.)
        style_indicators = ["tag", "lm", "random"]
        criterion_parts = []
        for i, part in enumerate(task_parts[1:], 1):
            if part in style_indicators and i < len(task_parts) - 1:
                # This is the style part
                break
            criterion_parts.append(part)
        criterion = "__".join(criterion_parts) if criterion_parts else "default"
    else:
        criterion = "default"

    # Store criterion in a way we can access it later
    return documents, doc_texts, dataset_name, embeddings, criterion


def list_benchmark_tasks_and_methods(run_name: str) -> None:
    """List available tasks and methods in a benchmark run.

    Args:
        run_name: Benchmark run name
    """
    output_base = Path("outputs") / run_name

    if not output_base.exists():
        logger.error(f"Benchmark run not found: {run_name}")
        logger.error("\nAvailable runs:")
        for d in Path("outputs").iterdir():
            if d.is_dir() and not d.name.startswith("."):
                logger.error(f"  - {d.name}")
        return

    method_logs_dir = output_base / "method_logs"
    if not method_logs_dir.exists():
        logger.error(f"No method logs in {run_name}")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Available tasks in {run_name}:")
    logger.info("=" * 60)

    for task_dir in sorted(method_logs_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        logger.info(f"\nTask: {task_dir.name}")

        method_files = list(task_dir.glob("*.jsonl"))
        if method_files:
            logger.info("  Methods:")
            for mf in sorted(method_files):
                logger.info(f"    - {mf.stem}")

    logger.info("=" * 60 + "\n")


def generate_embeddings(
    doc_texts: list[str],
    embedding_preset: str,
    cache_alias: str | None = None,
    force_refresh: bool = False,
    criterion: str | None = None,
    in_one_word_context: str | None = None,
    pseudologit_classes: str | None = None,
) -> np.ndarray:
    """Generate embeddings for documents.

    Args:
        doc_texts: List of document texts
        embedding_preset: Preset name
        cache_alias: Cache identifier
        force_refresh: Force refresh cache
        criterion: Criterion for in-one-word (e.g., 'arithmetic')
        in_one_word_context: Context string for in-one-word embeddings
        pseudologit_classes: Path to classes JSON for pseudologit embeddings

    Returns:
        Embeddings array
    """
    logger.info(f"Generating embeddings with {embedding_preset}")

    # Build inputs dict dynamically based on provided arguments
    inputs = {"document": doc_texts}

    # Add criterion if provided (for instruction-tuned embeddings)
    if criterion:
        inputs["criterion"] = criterion
        inputs["criterion_description"] = ""  # Use empty description

    # Add context if provided (for any preset that uses it, like inoneword)
    if in_one_word_context:
        context_text = read_or_return(in_one_word_context)
        context_preview = context_text[:100] + (
            "..." if len(context_text) > 100 else ""
        )
        logger.info(f"Using context: {context_preview}")
        inputs["context"] = context_text

    # Prepare config overrides
    config_overrides = {}

    # Add pseudologit classes override if provided
    if pseudologit_classes:
        from multiview.inference.presets import get_preset

        logger.info(f"Using pseudologit classes from: {pseudologit_classes}")

        # Get the preset to merge extra_kwargs properly
        preset_config = get_preset(embedding_preset)
        merged_extra_kwargs = preset_config.extra_kwargs.copy()
        merged_extra_kwargs["classes_file"] = pseudologit_classes
        config_overrides["extra_kwargs"] = merged_extra_kwargs

    # Run inference with dynamically built inputs
    results = run_inference(
        inputs=inputs,
        config=embedding_preset,
        cache_alias=cache_alias,
        force_refresh=force_refresh,
        **config_overrides,
    )

    # Convert to numpy array
    embeddings = np.array(results, dtype=np.float32)

    logger.info(
        f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}"
    )

    return embeddings


def create_reducer(args: argparse.Namespace, n_samples: int | None = None):
    """Create dimensionality reducer based on args.

    Args:
        args: Command-line arguments
        n_samples: Number of samples (used for auto-sizing SOM grid)
    """
    if args.reducer == "tsne":
        return TSNEReducer(
            perplexity=args.perplexity,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
    elif args.reducer == "pca":
        return PCAReducer(
            n_components=2,
            whiten=True,
            random_state=args.random_state,
        )
    elif args.reducer == "umap":
        return UMAPReducer(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
        )
    elif args.reducer == "som":
        grid_size_str = args.som_grid_size

        # Auto-calculate grid size if using default and n_samples is provided
        if grid_size_str == "auto" and n_samples is not None:
            # Calculate a roughly square grid with slight padding
            # Use 1.1x samples to give SOM some breathing room
            import math

            target_cells = int(n_samples * 1.1)
            side = int(math.sqrt(target_cells))
            # Find factors close to square
            grid_rows = side
            grid_cols = math.ceil(target_cells / side)
            logger.info(
                f"Auto-selected SOM grid size: {grid_rows}x{grid_cols} = {grid_rows * grid_cols} cells for {n_samples} samples"
            )
        elif grid_size_str == "auto":
            # Fallback to default if n_samples not provided
            grid_rows, grid_cols = 20, 20
            logger.warning(
                "Auto grid size requested but n_samples not provided, using default 20x20"
            )
        else:
            grid_rows, grid_cols = map(int, grid_size_str.split(","))

        return SOMReducer(
            grid_size=(grid_rows, grid_cols),
            learning_rate=args.som_learning_rate,
            iterations=args.som_iterations,
            random_state=args.random_state,
            unique_assignment=args.som_unique_assignment,
        )
    elif args.reducer == "dendrogram":
        return DendrogramReducer(
            method=args.dendrogram_method,
            metric=args.dendrogram_metric,
            optimal_ordering=True,
        )
    elif args.reducer == "heatmap":
        # Heatmap doesn't need a reducer - works directly from embeddings/similarity matrix
        return None
    else:
        raise ValueError(f"Unknown reducer: {args.reducer}")


# ============================================================================
# GSM8K Graph Mode Functions
# ============================================================================


def load_gsm8k_documents(input_path: str, max_docs: int = -1) -> list[dict]:
    """Load GSM8K documents from a JSONL file.

    Args:
        input_path: Path to the JSONL file
        max_docs: Maximum number of documents to load (-1 for all)

    Returns:
        List of document dictionaries
    """
    documents = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            if max_docs > 0 and i >= max_docs:
                break
            doc = json.loads(line)
            documents.append(doc)
    return documents


def visualize_gsm8k_batch(
    documents: list[dict],
    output_dir: str,
    format: str = "png",
    show_question: bool = True,
    prefix: str = "problem",
):
    """Visualize a batch of GSM8K problems.

    Args:
        documents: List of GSM8K document dictionaries
        output_dir: Directory to save visualizations
        format: Output format (png, svg, pdf, etc.)
        show_question: Whether to include question text in visualization
        prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Visualizing {len(documents)} GSM8K problems")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Format: {format}")

    for i, doc in enumerate(documents):
        doc_id = doc.get("doc_id", i)
        output_path = os.path.join(output_dir, f"{prefix}_{doc_id}")

        try:
            question, answer = parse_gsm8k_document(doc)
            graph = GSM8KComputationalGraph(question, answer)
            graph.parse()

            # Print summary
            num_inputs = sum(1 for n in graph.nodes.values() if n.node_type == "input")
            num_calcs = sum(
                1 for n in graph.nodes.values() if n.node_type == "operation"
            )

            logger.info(
                f"[{i+1}/{len(documents)}] Doc {doc_id}: Inputs={num_inputs}, Calculations={num_calcs}, Final={graph.final_answer}"
            )

            # Render
            output_file = graph.render(
                output_path, show_question=show_question, format=format, view=False
            )
            logger.debug(f"  Saved: {output_file}")

            # Save graph structure as JSON
            json_path = output_path + ".json"
            with open(json_path, "w") as f:
                json.dump(graph.to_dict(), f, indent=2)
            logger.debug(f"  Graph data: {json_path}")

        except Exception as e:
            logger.error(f"[{i+1}/{len(documents)}] Doc {doc_id}: ERROR - {e}")

    logger.info(f"Visualizations saved to: {output_dir}")


def run_gsm8k_graph_mode(args: argparse.Namespace):
    """Run GSM8K graph visualization mode."""
    logger.info("=" * 60)
    logger.info("GSM8K COMPUTATIONAL GRAPH VISUALIZATION")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Format: {args.format}")
    logger.info("")

    # Load documents
    all_documents = load_gsm8k_documents(args.input, max_docs=-1)

    # Apply start and num filters
    if args.start > 0:
        all_documents = all_documents[args.start :]

    if args.num > 0:
        all_documents = all_documents[: args.num]

    logger.info(f"Loaded {len(all_documents)} documents from {args.input}")
    logger.info("")

    # Visualize
    visualize_gsm8k_batch(
        all_documents,
        args.output,
        format=args.format,
        show_question=not args.no_question,
        prefix=args.prefix,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info("")


def create_thumbnail_images(
    documents: list[Any],
    dataset_name: str,
    criterion: str | None,
    output_dir: str,
) -> list[str]:
    """Create thumbnail images for documents based on dataset type.

    Args:
        documents: List of document objects
        dataset_name: Name of the dataset (e.g., 'gsm8k', 'arxiv_cs')
        criterion: Optional criterion being analyzed (e.g., 'arithmetic')
        output_dir: Directory to save thumbnails

    Returns:
        List of paths to generated thumbnail images
    """
    # Check if thumbnails already exist
    output_path = Path(output_dir)
    if output_path.exists():
        # Check for both marker_* (GSM8K) and thumb_* (image datasets) patterns
        existing_images = list(output_path.glob("marker_*.*")) + list(
            output_path.glob("thumb_*.*")
        )
        if len(existing_images) == len(documents):
            return sorted([str(p) for p in existing_images])

    # GSM8K: Generate computational graph thumbnails (for arithmetic/final_expression)
    if dataset_name == "gsm8k" and criterion in ["arithmetic", "final_expression"]:
        return create_gsm8k_marker_images(
            documents,
            output_dir,
            show_question=False,
            figsize=(16, 12),  # Much larger for better quality
            dpi=300,  # High resolution
            minimal=True,  # Always use minimal style for markers
        )

    # Image datasets: Use the actual images as thumbnails
    elif dataset_name in ["ut_zappos50k", "met_museum", "example_images"]:
        logger.info(f"Using actual images as thumbnails for '{dataset_name}'...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                image_path = doc.get("image_path")
                if image_path:
                    # Handle data URIs (base64 encoded images)
                    if image_path.startswith("data:image"):
                        import base64
                        import re

                        # Extract format and base64 data
                        match = re.match(r"data:image/(\w+);base64,(.+)", image_path)
                        if match:
                            img_format = match.group(1)
                            b64_data = match.group(2)

                            # Decode base64 data
                            img_bytes = base64.b64decode(b64_data)

                            # Save to thumbnails directory
                            dest_name = f"thumb_{i}.{img_format}"
                            dest_path = output_path / dest_name
                            with open(dest_path, "wb") as f:
                                f.write(img_bytes)

                            image_paths.append(str(dest_path))
                        else:
                            logger.warning(f"Invalid data URI format for document {i}")
                            image_paths.append(None)
                    # Handle URLs - keep as-is (will be handled by visualization code)
                    elif image_path.startswith("http://") or image_path.startswith(
                        "https://"
                    ):
                        image_paths.append(image_path)
                    # Handle local file paths
                    elif Path(image_path).exists():
                        image_paths.append(image_path)
                    else:
                        logger.warning(
                            f"Image path not found for document {i}: {image_path}"
                        )
                        image_paths.append(None)
                else:
                    logger.warning(f"Document missing image_path: {doc}")
                    image_paths.append(None)
            else:
                logger.warning(f"Document is not a dict: {doc}")
                image_paths.append(None)

        logger.info(
            f"Found {len([p for p in image_paths if p])} images out of {len(documents)} documents"
        )
        return image_paths

    # Default: No thumbnails for unknown datasets (use scatter markers instead)
    else:
        logger.info(
            f"No specialized thumbnails for '{dataset_name}', use --marker-type scatter instead"
        )
        return []


def load_documents(
    args: argparse.Namespace,
) -> tuple[list[Any], list[str], str | None, np.ndarray | None, str | None]:
    """Load documents based on args, returning (documents, texts, dataset_name, embeddings, criterion).

    For benchmark mode, returns embeddings directly if available in outputs dir.
    For other modes, returns None for embeddings (will generate fresh).
    """
    # Benchmark mode
    if hasattr(args, "from_benchmark") and args.from_benchmark:
        logger.debug(f"Task: {args.task}")
        logger.debug(f"Method: {args.method}")

        documents, doc_texts, dataset_name, embeddings, criterion = load_from_benchmark(
            args.from_benchmark,
            args.task,
            args.method,
        )
        return documents, doc_texts, dataset_name, embeddings, criterion

    # Existing JSONL mode
    elif args.input_jsonl:
        logger.info(f"Input: {args.input_jsonl}")
        documents, doc_texts = load_documents_from_jsonl(
            args.input_jsonl, args.max_docs, args.text_field
        )
        # Infer dataset name from input file for thumbnails
        dataset_name = "gsm8k" if "gsm8k" in args.input_jsonl.lower() else None
        criterion = (
            args.criterion if hasattr(args, "criterion") and args.criterion else None
        )
        return documents, doc_texts, dataset_name, None, criterion

    # Existing dataset mode
    else:
        logger.info(f"Dataset: {args.dataset}")
        documents, doc_texts = load_documents_from_dataset(
            args.dataset, args.dataset_config, args.max_docs
        )
        dataset_name = args.dataset

        # Save if requested
        if args.save_docs_jsonl:
            save_documents_jsonl(documents, args.save_docs_jsonl)

        logger.info(f"Reducer: {args.reducer}")
        logger.info(f"Max docs: {args.max_docs or 'all'}")
        logger.info("")
        logger.info(f"Loaded {len(documents)} documents")
        criterion = (
            args.criterion if hasattr(args, "criterion") and args.criterion else None
        )
        return documents, doc_texts, dataset_name, None, criterion


def prepare_markers(
    documents: list[Any], dataset_name: str | None, args: argparse.Namespace
) -> tuple[list[str] | None, list[str] | None]:
    """Prepare image_paths and labels based on marker type.

    Returns:
        (image_paths, labels) tuple
    """
    image_paths = None
    labels = None

    if args.marker_type == "thumbnail":
        if dataset_name:
            # Thumbnails are task-level (shared by all methods), not method-level
            # args.output is outputs/viz/benchmark/task/method/, so parent is task-level
            output_dir = Path(args.output).parent / "_markers"
            image_paths = create_thumbnail_images(
                documents, dataset_name, args.criterion, str(output_dir)
            )
            if image_paths:
                # Note: create_thumbnail_images logs whether it generated or reused existing
                pass
            else:
                logger.info("No thumbnails generated, falling back to scatter markers")
                image_paths = None
        else:
            logger.info(
                "No dataset specified, cannot generate thumbnails. Use scatter markers instead."
            )
            image_paths = None

    elif args.marker_type == "text" or args.show_text_labels:
        labels = [str(i) for i in range(len(documents))]

    return image_paths, labels


def save_documents_jsonl(documents: list[Any], output_path: str):
    """Save documents to JSONL file."""
    logger.info(f"Saving documents to {output_path}...")
    with open(output_path, "w") as f:
        for doc in documents:
            # If document is a string, wrap it in a dict
            doc_dict = {"document": doc} if isinstance(doc, str) else doc
            f.write(json.dumps(doc_dict) + "\n")
    logger.info(f"Saved {len(documents)} documents to {output_path}")


def save_som_manifest(
    image_paths: list[str],
    assignments: np.ndarray,
    reducer_rows: int,
    reducer_cols: int,
    output_base: str,
):
    """Save SOM grid assignment manifest."""
    manifest = []
    for idx, (img_path, node_idx) in enumerate(
        zip(image_paths, assignments, strict=False)
    ):
        row = node_idx // reducer_cols
        col = node_idx % reducer_cols
        manifest.append(
            {
                "index": idx,
                "grid_row": row,
                "grid_col": col,
                "grid_node": int(node_idx),
                "image_path": str(img_path),
            }
        )

    manifest_file = f"{output_base}_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest: {manifest_file}")


def save_coordinates(coords_2d: np.ndarray, reducer, args: argparse.Namespace):
    """Save 2D coordinates and metadata to JSON."""
    coords_data = {
        "coords_2d": coords_2d.tolist(),
        "reducer": args.reducer,
        "n_documents": len(coords_2d),
    }

    # Add reducer-specific params
    if args.reducer == "tsne":
        coords_data["perplexity"] = args.perplexity
        coords_data["max_iter"] = args.max_iter
    elif args.reducer == "pca":
        if hasattr(reducer, "explained_variance_ratio_"):
            coords_data["explained_variance_ratio"] = (
                reducer.explained_variance_ratio_.tolist()
            )
    elif args.reducer == "umap":
        coords_data["n_neighbors"] = args.n_neighbors
        coords_data["min_dist"] = args.min_dist
    elif args.reducer == "som":
        coords_data["grid_size"] = args.som_grid_size

    coords_path = f"{args.output}_coords.json"
    with open(coords_path, "w") as f:
        json.dump(coords_data, f, indent=2)
    logger.info(f"Saved coordinates: {coords_path}")


def export_for_web_viewer(
    documents: list[Any],
    embeddings: np.ndarray,
    coords_2d: np.ndarray,
    reducer_name: str,
    output_dir: Path,
    dataset_name: str,
    criterion: str = "default",
    method: str | None = None,
    linkage_matrix: np.ndarray | None = None,
    image_paths: list[str] | None = None,
    dendrogram_image: str | None = None,
    som_grid_image: str | None = None,
    quiet: bool = False,
):
    """Export visualization data for web viewer.

    Saves:
    - manifest.json: metadata + file paths
    - documents.txt: one doc per line
    - embeddings.npy: raw embeddings
    - layout_{reducer}.npy: 2D coordinates
    - linkage_matrix.npy: (optional) for dendrogram
    - dendrogram.png: (optional) matplotlib dendrogram image
    - som_grid.png: (optional) SOM grid composite image
    - thumbnails/*.png: (optional) thumbnail images

    Args:
        documents: List of documents (strings or dicts)
        embeddings: Document embeddings array
        coords_2d: 2D coordinates from reducer
        reducer_name: Name of the reducer used
        output_dir: Output directory path
        dataset_name: Name of the dataset
        criterion: Criterion name (e.g., 'arithmetic')
        linkage_matrix: Optional linkage matrix for dendrogram
        image_paths: Optional list of thumbnail image paths
        dendrogram_image: Optional path to matplotlib dendrogram image
        som_grid_image: Optional path to SOM grid composite image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing manifest if it exists
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if not quiet:
            logger.debug(f"Loading existing manifest from {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        # Create new manifest
        manifest = {
            "version": 1,
            "dataset": dataset_name,
            "criterion": criterion,
            "n_docs": len(documents),
            "embedding_dim": embeddings.shape[1],
            "documents_path": "documents.txt",
            "embeddings_path": "embeddings.npy",
            "layouts": {},
        }
        if method is not None:
            manifest["method"] = method

    def _doc_to_text(doc: Any) -> str:
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            text = doc.get("text", "")
            if text:
                return str(text)
            if doc.get("image_path"):
                return "<image>"
            return json.dumps(doc, ensure_ascii=False)
        return str(doc)

    # Save documents as text file (only if not already exists or needs update)
    documents_path = output_dir / "documents.txt"
    if not documents_path.exists():
        with open(documents_path, "w", encoding="utf-8") as f:
            for doc in documents:
                # Escape newlines within documents
                doc_escaped = _doc_to_text(doc).replace("\n", "\\n")
                f.write(doc_escaped + "\n")

    # Save embeddings (only if not already exists)
    embeddings_path = output_dir / "embeddings.npy"
    if not embeddings_path.exists():
        np.save(embeddings_path, embeddings)

    # Save layout coordinates (if available - heatmap doesn't need them)
    if coords_2d is not None:
        layout_path = output_dir / f"layout_{reducer_name}.npy"
        np.save(layout_path, coords_2d)

        # Update layouts in manifest
        manifest["layouts"][reducer_name] = f"layout_{reducer_name}.npy"
    elif reducer_name == "heatmap":
        # Heatmap works directly from embeddings, just add to manifest
        manifest["layouts"]["heatmap"] = None  # No layout file needed

    # Save linkage matrix if provided
    if linkage_matrix is not None:
        linkage_path = output_dir / "linkage_matrix.npy"
        np.save(linkage_path, linkage_matrix)
        manifest["linkage_matrix"] = "linkage_matrix.npy"

    # Copy dendrogram image if provided
    if dendrogram_image and Path(dendrogram_image).exists():
        dendrogram_dest = output_dir / "dendrogram.png"
        shutil.copy2(dendrogram_image, dendrogram_dest)
        manifest["dendrogram_image"] = "dendrogram.png"

    # Copy SOM grid image if provided
    if som_grid_image and Path(som_grid_image).exists():
        som_dest = output_dir / "som_grid.png"
        shutil.copy2(som_grid_image, som_dest)
        manifest["som_grid_image"] = "som_grid.png"

    # Copy thumbnails if provided (only if not already present)
    if image_paths:
        thumbnails_dir = output_dir / "thumbnails"
        thumbnails_dir.mkdir(exist_ok=True)

        # Check if thumbnails already exist
        existing_thumbs = list(thumbnails_dir.glob("thumb_*"))
        if len(existing_thumbs) == len(image_paths):
            # Thumbnails already exist, skip copy
            # Build refs from existing files
            thumbnail_refs = []
            for i in range(len(image_paths)):
                # Find matching thumbnail
                matching = [f for f in existing_thumbs if f.stem == f"thumb_{i}"]
                if matching:
                    thumbnail_refs.append(f"thumbnails/{matching[0].name}")
                else:
                    thumbnail_refs.append(None)
        else:
            # Generate/copy thumbnails
            thumbnail_refs = []
            for i, img_path in enumerate(image_paths):
                if not img_path:
                    thumbnail_refs.append(None)
                    continue

                # Handle data URIs (base64 encoded images)
                if img_path.startswith("data:image"):
                    # Decode and save to disk (like gsm8k does)
                    import base64
                    import re

                    # Extract format and base64 data
                    match = re.match(r"data:image/(\w+);base64,(.+)", img_path)
                    if match:
                        img_format = match.group(1)
                        b64_data = match.group(2)

                        # Decode base64 data
                        img_bytes = base64.b64decode(b64_data)

                        # Save to thumbnails directory
                        dest_name = f"thumb_{i}.{img_format}"
                        dest_path = thumbnails_dir / dest_name
                        with open(dest_path, "wb") as f:
                            f.write(img_bytes)

                        thumbnail_refs.append(f"thumbnails/{dest_name}")
                    else:
                        logger.warning(f"Invalid data URI format for thumbnail {i}")
                        thumbnail_refs.append(None)
                # Handle URLs
                elif img_path.startswith("http://") or img_path.startswith("https://"):
                    # Keep URLs as-is for web viewer to fetch directly
                    thumbnail_refs.append(img_path)
                # Handle local file paths
                elif Path(img_path).exists():
                    # Copy to thumbnails directory with consistent naming
                    ext = Path(img_path).suffix
                    dest_name = f"thumb_{i}{ext}"
                    dest_path = thumbnails_dir / dest_name
                    shutil.copy2(img_path, dest_path)
                    thumbnail_refs.append(f"thumbnails/{dest_name}")
                else:
                    logger.warning(f"Thumbnail {i} not found or invalid: {img_path}")
                    thumbnail_refs.append(None)

        manifest["thumbnails"] = thumbnail_refs

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def print_summary(
    args: argparse.Namespace, image_paths: list[str] | None, output_dir: Path | None
):
    """Print final summary of outputs."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output}.{args.format}")
    if args.save_coords:
        logger.info(f"Coordinates: {args.output}_coords.json")
    if image_paths and output_dir:
        logger.info(f"Markers: {output_dir}/")
    if args.save_docs_jsonl and not args.input_jsonl:
        logger.info(f"Documents: {args.save_docs_jsonl}")
    logger.info("")


def create_som_grid_composite(
    visualizer: CorpusVisualizer,
    reducer: SOMReducer,
    embeddings: np.ndarray,
    image_paths: list[str],
    args: argparse.Namespace,
):
    """Create SOM grid composite visualization."""
    quiet = getattr(args, "quiet", False)
    if not quiet:
        logger.debug(f"Reducing {len(embeddings)} embeddings to 2D...")

    # Train SOM and get assignments
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    if reducer.weights is None:
        reducer._coords = reducer._build_coords()
        reducer._initialize_weights(embeddings_arr)
    reducer._train(embeddings_arr)
    assignments = reducer.assign_unique_nodes(embeddings_arr)

    if not quiet:
        logger.debug("Creating visualization...")

    # Parse background color
    bg_hex = args.som_background_color.lstrip("#")
    if len(bg_hex) == 3:
        bg_hex = "".join(ch * 2 for ch in bg_hex)
    background_color = tuple(int(bg_hex[i : i + 2], 16) for i in (0, 2, 4))

    # Create and save grid
    grid_image = visualizer.create_grid_composite(
        image_paths=image_paths,
        assignments=assignments,
        grid_rows=reducer.rows,
        grid_cols=reducer.cols,
        tile_size=args.som_tile_size,
        padding=args.som_padding,
        background_color=background_color,
    )

    output_file = f"{args.output}.{args.format}"
    if not quiet:
        logger.debug(f"Saving to {output_file}...")
    grid_image.save(output_file, format=args.format.upper())
    if not quiet:
        logger.debug(f"Saved visualization: {output_file}")

    # Save manifest
    save_som_manifest(image_paths, assignments, reducer.rows, reducer.cols, args.output)


def create_dendrogram_plot(
    visualizer: CorpusVisualizer,
    embeddings: np.ndarray,
    image_paths: list[str],
    figsize: tuple,
    args: argparse.Namespace,
):
    """Create dendrogram visualization with images.

    Args:
        visualizer: CorpusVisualizer instance
        embeddings: Document embeddings
        image_paths: List of image paths for thumbnails
        figsize: Figure size tuple
        args: Command-line arguments
    """
    quiet = getattr(args, "quiet", False)
    if not quiet:
        logger.debug(f"Reducing {len(embeddings)} embeddings to dendrogram ordering...")

    # Fit the dendrogram reducer to get the hierarchical clustering
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    visualizer.reducer.fit_transform(embeddings_arr)

    if not quiet:
        logger.debug("Creating dendrogram visualization with images...")

    # Create the dendrogram plot
    fig, ax = visualizer.plot_dendrogram_with_images(
        image_paths=image_paths,
        figsize=figsize,
        title=args.title,
        image_size=args.dendrogram_image_size,
        orientation=args.dendrogram_orientation,
        images_per_row=args.dendrogram_images_per_row,
        num_clusters=args.dendrogram_num_clusters,
    )

    # Save the figure
    output_file = f"{args.output}.{args.format}"
    if not quiet:
        logger.debug(f"Saving to {output_file}...")
    fig.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    if not quiet:
        logger.debug(f"Saved visualization: {output_file}")


def create_scatter_plot(
    visualizer: CorpusVisualizer,
    embeddings: np.ndarray,
    documents: list[Any],
    labels: list[str] | None,
    classes: list[str] | None,
    image_paths: list[str] | None,
    figsize: tuple,
    args: argparse.Namespace,
) -> np.ndarray:
    """Create standard scatter plot visualization.

    Returns:
        coords_2d: The 2D coordinates
    """
    coords_2d, _, _ = visualizer.visualize_corpus(
        embeddings=embeddings,
        documents=documents,
        labels=labels,
        classes=classes,
        image_paths=image_paths,
        output_path=args.output,
        format=args.format,
        dpi=args.dpi,
        figsize=figsize,
        title=args.title,
        marker_size=args.marker_size,
        alpha=args.alpha,
        image_zoom=args.image_zoom,
        show_text_labels=args.show_text_labels,
    )

    # Save coordinates if requested
    if args.save_coords:
        save_coordinates(coords_2d, visualizer.reducer, args)

    return coords_2d


def visualize(
    embeddings: np.ndarray,
    documents: list[Any],
    classes: list[str] | None,
    image_paths: list[str] | None,
    labels: list[str] | None,
    reducer,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, str | None]:
    """Create and save visualization.

    Returns:
        Tuple of (coords_2d, linkage_matrix, dendrogram_image_path, som_grid_image_path).
        linkage_matrix is only set for dendrogram.
        dendrogram_image_path is the path to the saved matplotlib dendrogram image.
        som_grid_image_path is the path to the saved SOM grid composite image.
    """
    figsize = tuple(map(float, args.figsize.split(",")))
    visualizer = CorpusVisualizer(reducer=reducer)
    coords_2d = None
    linkage_matrix = None
    dendrogram_image_path = None
    som_grid_image_path = None

    # Check for dendrogram with images
    if args.reducer == "dendrogram":
        if not image_paths:
            raise ValueError(
                "Dendrogram visualization requires image thumbnails. "
                "Use --marker-type thumbnail with a compatible dataset."
            )
        create_dendrogram_plot(visualizer, embeddings, image_paths, figsize, args)
        # Get linkage matrix from reducer after fitting
        if hasattr(visualizer.reducer, "linkage_matrix"):
            linkage_matrix = visualizer.reducer.linkage_matrix
        # Get 2D coords for dendrogram (leaf positions)
        if hasattr(visualizer.reducer, "coords_2d_"):
            coords_2d = visualizer.reducer.coords_2d_
        # The dendrogram was saved to args.output + format
        dendrogram_image_path = f"{args.output}.{args.format}"
    # Check if we should use grid composite (SOM with images)
    elif args.reducer == "som" and image_paths and isinstance(reducer, SOMReducer):
        create_som_grid_composite(visualizer, reducer, embeddings, image_paths, args)
        # For SOM, we can get the grid coordinates
        if hasattr(reducer, "_coords"):
            # Get the assigned coordinates for each document
            embeddings_arr = np.array(embeddings, dtype=np.float32)
            assignments = reducer.assign_unique_nodes(embeddings_arr)
            coords_2d = reducer._coords[assignments]
        # The SOM grid was saved to args.output + format
        som_grid_image_path = f"{args.output}.{args.format}"
    else:
        coords_2d = create_scatter_plot(
            visualizer,
            embeddings,
            documents,
            labels,
            classes,
            image_paths,
            figsize,
            args,
        )

    return coords_2d, linkage_matrix, dendrogram_image_path, som_grid_image_path


def visualize_benchmark_task(
    benchmark_run: str,
    task_name: str,
    method_name: str,
    reducer: str = "tsne",
    output_dir: str | Path | None = None,
    use_thumbnails: bool = False,
    quiet: bool = False,
) -> bool:
    """Compatibility wrapper for benchmark visualization from script context."""
    from multiview.utils.visualization_utils import (
        visualize_benchmark_task as _visualize_benchmark_task,
    )

    return _visualize_benchmark_task(
        benchmark_run=benchmark_run,
        task_name=task_name,
        method_name=method_name,
        reducer=reducer,
        output_dir=output_dir,
        use_thumbnails=use_thumbnails,
        quiet=quiet,
    )


def run_embedding_mode(args: argparse.Namespace):
    """Compatibility wrapper. Implementation lives in multiview.utils.visualization_utils."""
    from multiview.utils.visualization_utils import (
        run_embedding_mode as _run_embedding_mode,
    )

    _run_embedding_mode(args)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with organized groups."""
    parser = argparse.ArgumentParser(
        description="Visualize corpus embeddings in 2D or generate GSM8K graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        default="embedding",
        choices=["embedding", "gsm8k_graphs"],
        help="Visualization mode: 'embedding' for corpus embeddings (default), 'gsm8k_graphs' for computational graphs",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    # Data source options (embedding mode)
    data_group = parser.add_argument_group("Data source options")
    data_group.add_argument(
        "--input-jsonl",
        help="Path to JSONL file with documents (overrides --dataset if provided)",
    )
    data_group.add_argument(
        "--text-field",
        default="document",
        help="Field name in JSONL to extract text from (default: document)",
    )
    data_group.add_argument(
        "--dataset",
        help="Dataset name (e.g., gsm8k)",
    )
    data_group.add_argument(
        "--dataset-config",
        type=json.loads,
        help="Dataset config as JSON string",
    )
    data_group.add_argument(
        "--max-docs",
        type=int,
        help="Maximum number of documents to process",
    )

    # GSM8K graph mode options
    gsm8k_group = parser.add_argument_group("GSM8K graph mode options")
    gsm8k_group.add_argument(
        "--input",
        "-i",
        help="Path to GSM8K documents JSONL file",
    )
    gsm8k_group.add_argument(
        "--num",
        "-n",
        type=int,
        default=5,
        help="Number of problems to visualize (-1 for all, default: 5)",
    )
    gsm8k_group.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index (default: 0)",
    )
    gsm8k_group.add_argument(
        "--no-question",
        action="store_true",
        help="Hide question text",
    )
    gsm8k_group.add_argument(
        "--prefix",
        "-p",
        default="problem",
        help="Prefix for output filenames (default: problem)",
    )

    # Benchmark mode options
    benchmark_group = parser.add_argument_group(
        "Benchmark mode (alternative to dataset/input-jsonl)"
    )
    benchmark_group.add_argument(
        "--from-benchmark",
        help="Load from benchmark run (e.g., 'benchmark_debug')",
    )
    benchmark_group.add_argument(
        "--task",
        help="Task name for benchmark mode (e.g., 'gsm8k__arithmetic')",
    )
    benchmark_group.add_argument(
        "--method",
        help="Method name for benchmark mode (e.g., 'qwen3_8b_with_instructions')",
    )
    benchmark_group.add_argument(
        "--list-benchmark",
        action="store_true",
        help="List available tasks and methods in benchmark run",
    )

    # Embedding options
    embed_group = parser.add_argument_group("Embedding generation")
    embed_group.add_argument(
        "--embedding-preset",
        default="hf_qwen3_embedding_8b",
        help="Embedding preset (default: hf_qwen3_embedding_8b)",
    )
    embed_group.add_argument(
        "--cache-alias",
        help="Cache identifier for embeddings",
    )
    embed_group.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh embeddings cache",
    )
    embed_group.add_argument(
        "--criterion",
        help="Criterion for in-one-word embeddings (e.g., 'arithmetic')",
    )
    embed_group.add_argument(
        "--in-one-word-context",
        help="Context for in-one-word embeddings: either an inline string or path to a text file "
        "(e.g., 'Categories: addition, subtraction' or 'prompts/custom/my_context.txt'). "
        "Required for in-one-word presets.",
    )
    embed_group.add_argument(
        "--pseudologit-classes",
        help="Path to JSON file defining taxonomy classes for pseudologit embeddings "
        "(e.g., 'prompts/custom/gsm8k_classes.json'). Overrides the classes_file in the preset.",
    )

    # Dimensionality reduction options
    reducer_group = parser.add_argument_group("Dimensionality reduction")
    reducer_group.add_argument(
        "--reducer",
        default="tsne",
        choices=["tsne", "pca", "umap", "som", "dendrogram"],
        help="Dimensionality reduction method (default: tsne). Note: dendrogram requires image thumbnails.",
    )
    reducer_group.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    reducer_group.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30)",
    )
    reducer_group.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="t-SNE max iterations (default: 1000)",
    )
    reducer_group.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15)",
    )
    reducer_group.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1)",
    )
    reducer_group.add_argument(
        "--som-grid-size",
        default="auto",
        help="SOM grid size as rows,cols or 'auto' (default: auto)",
    )
    reducer_group.add_argument(
        "--som-learning-rate",
        type=float,
        default=0.5,
        help="SOM learning rate (default: 0.5)",
    )
    reducer_group.add_argument(
        "--som-iterations",
        type=int,
        default=1000,
        help="SOM iterations (default: 1000)",
    )
    reducer_group.add_argument(
        "--som-unique-assignment",
        action="store_true",
        default=True,
        help="Assign each sample to unique node (default: True)",
    )
    reducer_group.add_argument(
        "--no-som-unique-assignment",
        dest="som_unique_assignment",
        action="store_false",
        help="Allow multiple samples per node",
    )
    reducer_group.add_argument(
        "--dendrogram-method",
        default="average",
        choices=[
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        ],
        help="Dendrogram linkage method (default: average)",
    )
    reducer_group.add_argument(
        "--dendrogram-metric",
        default="euclidean",
        help="Dendrogram distance metric (default: euclidean)",
    )
    reducer_group.add_argument(
        "--dendrogram-orientation",
        default="top",
        choices=["top", "bottom", "left", "right"],
        help="Dendrogram orientation (default: top)",
    )
    reducer_group.add_argument(
        "--dendrogram-image-size",
        type=float,
        default=0.8,
        help="Dendrogram image size relative to spacing (default: 0.8)",
    )
    reducer_group.add_argument(
        "--dendrogram-images-per-row",
        type=int,
        default=None,
        help="Number of images per row in grid layout below dendrogram. "
        "Arranges leaf images in multiple rows for better visibility. "
        "Example: --dendrogram-images-per-row 15 (default: None, auto-calculated)",
    )
    reducer_group.add_argument(
        "--dendrogram-num-clusters",
        type=int,
        default=None,
        help="Number of clusters to color-code in dendrogram. "
        "Colors dendrogram branches and image borders by cluster membership. "
        "Example: --dendrogram-num-clusters 8 (default: None, auto ~10%% of samples)",
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization options")
    viz_group.add_argument(
        "--marker-type",
        default="scatter",
        choices=["scatter", "thumbnail", "text"],
        help="Marker type (default: scatter)",
    )
    viz_group.add_argument(
        "--annotations-file",
        help="Path to annotations JSONL for coloring",
    )
    viz_group.add_argument(
        "--color-by",
        default="category",
        help="Annotation field to color by (default: category)",
    )
    viz_group.add_argument(
        "--marker-size",
        type=int,
        default=50,
        help="Scatter marker size (default: 50)",
    )
    viz_group.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Marker transparency (default: 0.7)",
    )
    viz_group.add_argument(
        "--image-zoom",
        type=float,
        default=0.5,
        help="Image marker zoom (default: 0.5)",
    )
    viz_group.add_argument(
        "--show-text-labels",
        action="store_true",
        help="Show text labels on points",
    )
    viz_group.add_argument(
        "--color-by-benchmark-annotations",
        action="store_true",
        help="Color points by annotations from benchmark (only with --from-benchmark)",
    )
    viz_group.add_argument(
        "--som-tile-size",
        type=int,
        default=200,
        help="SOM tile size in pixels (default: 200)",
    )
    viz_group.add_argument(
        "--som-padding",
        type=int,
        default=10,
        help="SOM tile padding (default: 10)",
    )
    viz_group.add_argument(
        "--som-background-color",
        default="000000",
        help="SOM background color hex (default: 000000)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output",
        "-o",
        help="Output path (without extension)",
    )
    output_group.add_argument(
        "--format",
        default="png",
        choices=["png", "svg", "pdf", "jpg"],
        help="Output format (default: png)",
    )
    output_group.add_argument(
        "--export-format",
        choices=["png", "svg", "web", "all"],
        help="Export format: 'png', 'svg' for plots only, 'web' for web viewer data, 'all' for both. "
        "Overrides --format when specified.",
    )
    output_group.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster formats (default: 150)",
    )
    output_group.add_argument(
        "--figsize",
        default="12,8",
        help="Figure size as width,height (default: 12,8)",
    )
    output_group.add_argument(
        "--title",
        help="Plot title",
    )
    output_group.add_argument(
        "--save-coords",
        action="store_true",
        help="Save 2D coordinates to JSON",
    )
    output_group.add_argument(
        "--save-docs-jsonl",
        help="Save loaded documents to JSONL",
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(level=args.log_level)

    # List benchmark mode
    if hasattr(args, "list_benchmark") and args.list_benchmark:
        if not args.from_benchmark:
            parser.error("--list-benchmark requires --from-benchmark")
        list_benchmark_tasks_and_methods(args.from_benchmark)
        return

    # Mode-specific validation and execution
    if args.mode == "gsm8k_graphs":
        if not args.input:
            parser.error("--input is required for gsm8k_graphs mode")
        if not args.output:
            parser.error("--output is required for gsm8k_graphs mode")
        run_gsm8k_graph_mode(args)
    else:
        # Require --output for non-list modes
        if not args.output:
            parser.error("--output is required for visualization")

        # Benchmark mode validation
        if hasattr(args, "from_benchmark") and args.from_benchmark:
            if not args.task:
                parser.error(
                    "--task is required with --from-benchmark (use --list-benchmark to see options)"
                )
            if not args.method:
                parser.error(
                    "--method is required with --from-benchmark (use --list-benchmark to see options)"
                )
        # Standard mode validation
        elif not args.input_jsonl and not args.dataset:
            parser.error(
                "Either --input-jsonl, --dataset, or --from-benchmark is required for embedding mode"
            )

        run_embedding_mode(args)


if __name__ == "__main__":
    main()
