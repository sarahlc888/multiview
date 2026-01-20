#!/usr/bin/env python3
"""Visualize document corpus embeddings in 2D or generate GSM8K computational graphs.

This script has two modes:
1. Embedding visualization: Generate 2D visualizations of document embeddings using
   t-SNE, PCA, UMAP, or self-organizing maps (SOM).
2. GSM8K graphs: Generate standalone computational graph visualizations.

Usage examples:
    # Basic t-SNE visualization
    python scripts/visualize_corpus.py \\
        --dataset gsm8k \\
        --embedding-preset hf_qwen3_embedding_8b \\
        --reducer tsne \\
        --output outputs/viz/gsm8k_tsne \\
        --max-docs 500

    # Visualize from JSONL file
    python scripts/visualize_corpus.py \\
        --input-jsonl outputs/my_documents.jsonl \\
        --embedding-preset hf_qwen3_embedding_8b \\
        --reducer tsne \\
        --output outputs/viz/my_viz

    # In-one-word embeddings
    python scripts/visualize_corpus.py \\
        --dataset gsm8k \\
        --embedding-preset inoneword_hf_qwen3_4b \\
        --criterion arithmetic \\
        --in-one-word-context "Categories: addition, subtraction, multiplication" \\
        --reducer tsne \\
        --output outputs/viz/gsm8k_inoneword \\
        --max-docs 100

    # Thumbnails as markers
    python scripts/visualize_corpus.py \\
        --dataset gsm8k \\
        --reducer tsne \\
        --marker-type thumbnail \\
        --image-zoom 0.08 \\
        --output outputs/viz/gsm8k_graphs \\
        --max-docs 100

    # Dendrogram with image thumbnails (hierarchical clustering)
    python scripts/visualize_corpus.py \\
        --dataset gsm8k \\
        --embedding-preset hf_qwen3_embedding_8b \\
        --reducer dendrogram \\
        --dendrogram-method average \\
        --marker-type thumbnail \\
        --output outputs/viz/gsm8k_dendrogram \\
        --max-docs 50 \\
        --figsize 24,12

    # Dendrogram with grid layout and cluster coloring
    python scripts/visualize_corpus.py \\
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

    # GSM8K computational graph mode
    python scripts/visualize_corpus.py \\
        --mode gsm8k_graphs \\
        --input documents.jsonl \\
        --output viz/gsm8k \\
        --num 5
"""

import argparse
import json
import logging
import os
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
    load_annotation_classes,
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
    logger.info(f"Generating thumbnails for dataset '{dataset_name}'...")

    # GSM8K: Generate computational graph thumbnails (only for arithmetic criterion)
    if dataset_name == "gsm8k" and criterion == "arithmetic":
        logger.info("Using GSM8K computational graph thumbnails (minimal style)...")
        return create_gsm8k_marker_images(
            documents,
            output_dir,
            show_question=False,
            figsize=(16, 12),  # Much larger for better quality
            dpi=300,  # High resolution
            minimal=True,  # Always use minimal style for markers
        )

    # Future: Add more dataset-specific thumbnail generators
    # elif dataset_name == "arxiv_cs":
    #     return create_text_thumbnails(documents, output_dir, field="title")
    # elif dataset_name == "example_images":
    #     return [doc.get("image_path") for doc in documents]

    # Default: No thumbnails for unknown datasets (use scatter markers instead)
    else:
        logger.info(
            f"No specialized thumbnails for '{dataset_name}', use --marker-type scatter instead"
        )
        return []


def load_documents(args: argparse.Namespace) -> tuple[list[Any], list[str], str | None]:
    """Load documents based on args, returning (documents, texts, dataset_name)."""
    logger.info("Loading documents...")

    if args.input_jsonl:
        logger.info(f"Input: {args.input_jsonl}")
        documents, doc_texts = load_documents_from_jsonl(
            args.input_jsonl, args.max_docs, args.text_field
        )
        # Infer dataset name from input file for thumbnails
        dataset_name = "gsm8k" if "gsm8k" in args.input_jsonl.lower() else None
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
    return documents, doc_texts, dataset_name


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
            output_dir = Path(args.output).parent / f"{Path(args.output).stem}_markers"
            image_paths = create_thumbnail_images(
                documents, dataset_name, args.criterion, str(output_dir)
            )
            if image_paths:
                logger.info(f"Generated {len(image_paths)} thumbnail markers")
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
    logger.info(f"Reducing {len(embeddings)} embeddings to 2D...")

    # Train SOM and get assignments
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    if reducer.weights is None:
        reducer._coords = reducer._build_coords()
        reducer._initialize_weights(embeddings_arr)
    reducer._train(embeddings_arr)
    assignments = reducer.assign_unique_nodes(embeddings_arr)

    logger.info("Creating visualization...")

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
    logger.info(f"Saving to {output_file}...")
    grid_image.save(output_file, format=args.format.upper())
    logger.info(f"Saved visualization: {output_file}")

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
    logger.info(f"Reducing {len(embeddings)} embeddings to dendrogram ordering...")

    # Fit the dendrogram reducer to get the hierarchical clustering
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    visualizer.reducer.fit_transform(embeddings_arr)

    logger.info("Creating dendrogram visualization with images...")

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
    logger.info(f"Saving to {output_file}...")
    fig.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    logger.info(f"Saved visualization: {output_file}")


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
):
    """Create and save visualization."""
    figsize = tuple(map(float, args.figsize.split(",")))
    visualizer = CorpusVisualizer(reducer=reducer)

    # Check for dendrogram with images
    if args.reducer == "dendrogram":
        if not image_paths:
            raise ValueError(
                "Dendrogram visualization requires image thumbnails. "
                "Use --marker-type thumbnail with a compatible dataset."
            )
        create_dendrogram_plot(visualizer, embeddings, image_paths, figsize, args)
    # Check if we should use grid composite (SOM with images)
    elif args.reducer == "som" and image_paths and isinstance(reducer, SOMReducer):
        create_som_grid_composite(visualizer, reducer, embeddings, image_paths, args)
    else:
        create_scatter_plot(
            visualizer,
            embeddings,
            documents,
            labels,
            classes,
            image_paths,
            figsize,
            args,
        )


def run_embedding_mode(args: argparse.Namespace):
    """Run embedding visualization mode.

    Loads documents, generates embeddings, applies dimensionality
    reduction, and creates visualization.
    """
    logger.info("=" * 60)
    logger.info(f"CORPUS VISUALIZATION: {args.reducer.upper()}")
    logger.info("=" * 60)

    # Step 1: Load documents
    documents, doc_texts, dataset_name = load_documents(args)

    # Step 2: Generate embeddings
    embeddings = generate_embeddings(
        doc_texts,
        args.embedding_preset,
        args.cache_alias,
        args.force_refresh,
        args.criterion,
        args.in_one_word_context,
        args.pseudologit_classes,
    )

    # Step 3: Load annotations if needed
    classes = None
    if args.annotations_file:
        logger.info(f"Loading annotations from {args.annotations_file}...")
        classes = load_annotation_classes(args.annotations_file, args.color_by)
        classes = classes[: len(documents)]
        logger.info(f"Loaded {len(classes)} class labels")

    # Step 4: Prepare markers
    image_paths, labels = prepare_markers(documents, dataset_name, args)

    # Step 5: Create reducer
    logger.info(f"Initializing {args.reducer} reducer...")
    reducer = create_reducer(args, n_samples=len(embeddings))

    # Step 6: Visualize
    visualize(embeddings, documents, classes, image_paths, labels, reducer, args)

    # Step 7: Print summary
    output_dir = (
        Path(args.output).parent / f"{Path(args.output).stem}_markers"
        if image_paths
        else None
    )
    print_summary(args, image_paths, output_dir)


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
        required=True,
        help="Output path (without extension)",
    )
    output_group.add_argument(
        "--format",
        default="png",
        choices=["png", "svg", "pdf", "jpg"],
        help="Output format (default: png)",
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

    # Mode-specific validation and execution
    if args.mode == "gsm8k_graphs":
        if not args.input:
            parser.error("--input is required for gsm8k_graphs mode")
        run_gsm8k_graph_mode(args)
    else:
        if not args.input_jsonl and not args.dataset:
            parser.error(
                "Either --input-jsonl or --dataset is required for embedding mode"
            )
        run_embedding_mode(args)


if __name__ == "__main__":
    main()
