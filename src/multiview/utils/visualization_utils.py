"""Shared utilities for visualization scripts."""

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from multiview.utils.benchmark_loading import load_from_benchmark
from multiview.visualization import (
    CorpusVisualizer,
    DendrogramReducer,
    PCAReducer,
    SOMReducer,
    TSNEReducer,
    UMAPReducer,
    create_gsm8k_marker_images,
    load_annotation_classes,
)

logger = logging.getLogger(__name__)


def _decode_data_uri_image(data_uri: str) -> tuple[str, bytes] | None:
    """Parse a base64 image data URI into (extension, bytes)."""
    match = re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", data_uri)
    if not match:
        return None
    mime_subtype = match.group(1).lower()
    ext = mime_subtype.split("+", 1)[0]
    if ext == "jpeg":
        ext = "jpg"
    try:
        image_bytes = base64.b64decode(match.group(2))
    except Exception:
        return None
    return ext, image_bytes


def _safe_path_exists(path_str: str) -> bool:
    """Return False for invalid/oversized pseudo-paths instead of raising OSError."""
    try:
        return Path(path_str).exists()
    except OSError:
        return False


def visualize_benchmark_task(
    benchmark_run: str,
    task_name: str,
    method_name: str,
    reducer: str = "tsne",
    output_dir: str | Path | None = None,
    output_benchmark_run: str | None = None,
    use_thumbnails: bool = False,
    quiet: bool = False,
    preloaded_data: tuple[list[Any], list[str], str, np.ndarray, str] | None = None,
) -> bool:
    """Visualize a single task/method combination from a benchmark run."""

    # Create a minimal args object with required fields
    class Args:
        pass

    args = Args()
    args.from_benchmark = benchmark_run
    args.task = task_name
    args.method = method_name
    args.reducer = reducer
    args.output = str(output_dir) if output_dir else None
    args.output_benchmark_run = output_benchmark_run
    args.export_format = "web"
    args.force_refresh = False
    args.color_by_benchmark_annotations = False
    args.annotations_file = None
    args.color_by = None
    args.marker_type = "thumbnail" if use_thumbnails else "dot"
    args.marker_size = 40
    args.image_size = 100
    args.format = "png"  # Default to PNG for image generation
    args.show = False
    args.dpi = 300
    args.figsize = "20,10"  # Default figure size for dendrograms/SOMs
    args.embedding_preset = None
    args.cache_alias = None
    args.criterion = None
    args.in_one_word_context = None
    args.pseudologit_classes = None
    args.input_jsonl = None
    args.dataset = None
    args.preloaded_data = preloaded_data

    # Reducer parameters
    args.random_state = 42
    args.perplexity = 30.0
    args.max_iter = 1000
    args.n_neighbors = 15
    args.min_dist = 0.1
    args.som_grid_size = "auto"
    args.som_learning_rate = 0.5
    args.som_iterations = 1000
    args.som_unique_assignment = True
    args.dendrogram_method = "average"
    args.dendrogram_metric = "euclidean"
    args.dendrogram_orientation = "top"
    args.dendrogram_image_size = 0.8
    args.dendrogram_images_per_row = None
    args.dendrogram_num_clusters = None

    # Visualization parameters
    args.alpha = 0.7
    args.image_zoom = 0.5
    args.show_text_labels = False
    args.title = None  # No title for web viewer images
    args.som_tile_size = 200
    args.som_padding = 10
    args.som_background_color = "000000"
    args.save_coords = False
    args.save_docs_jsonl = False
    args.quiet = quiet

    try:
        _run_benchmark_embedding_mode(args)
        return True
    except Exception as e:
        logger.error(f"Failed to visualize {task_name}/{method_name}: {e}")
        return False


def _infer_preferred_view(output_benchmark_run: str | None) -> str | None:
    """Infer preferred source view from output benchmark key/path."""
    if not output_benchmark_run:
        return None
    parts = Path(output_benchmark_run).parts
    if "triples" in parts:
        return "triples"
    if "corpus" in parts:
        return "corpus"
    return None


def _embedding_candidate_files(
    output_base: Path,
    task_name: str,
    method_name: str,
    preferred_view: str | None = None,
) -> list[Path]:
    """Build ordered embedding candidate paths with view-aware preference."""
    triples_path = (
        output_base / "triples" / "embeddings" / task_name / f"{method_name}.npy"
    )
    corpus_path = (
        output_base / "corpus" / "embeddings" / task_name / f"{method_name}.npy"
    )

    if preferred_view == "triples":
        return [triples_path, corpus_path]
    if preferred_view == "corpus":
        return [corpus_path, triples_path]
    # Default to eval artifacts first.
    return [triples_path, corpus_path]


def _doc_to_viewer_text(doc: Any) -> str:
    """Serialize a document into concise viewer text, preserving image-doc metadata."""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        text = doc.get("text", "")
        normalized_text = str(text).strip() if text is not None else ""
        if normalized_text and normalized_text.lower() != "<image>":
            return normalized_text

        # For image-only docs, surface compact scalar metadata if available.
        metadata_parts = []
        ignored_keys = {"text", "image_path", "embedding_viz", "_metadata"}
        for key, value in doc.items():
            if key in ignored_keys or value is None:
                continue
            if isinstance(value, str | int | float | bool):
                value_str = str(value).strip()
                if value_str:
                    metadata_parts.append(f"{key}: {value_str}")

        if metadata_parts:
            return " | ".join(metadata_parts)
        if doc.get("image_path"):
            return "<image>"
        return json.dumps(doc, ensure_ascii=False)
    return str(doc)


def _load_from_benchmark(
    run_name: str,
    task_name: str,
    method_name: str,
    preferred_view: str | None = None,
) -> tuple[list[Any], list[str], str, np.ndarray, str]:
    """Load documents and embeddings for benchmark visualization."""
    (
        documents,
        doc_texts,
        dataset_name,
        embeddings,
        criterion,
        _criterion_description,
    ) = load_from_benchmark(
        run_name=run_name,
        task_name=task_name,
        method_name=method_name,
        preferred_view=preferred_view,
    )
    return (
        documents,
        doc_texts,
        dataset_name,
        np.array(embeddings, dtype=np.float32),
        criterion,
    )


def _finite_embedding_row_mask(embeddings: np.ndarray) -> np.ndarray:
    """Return mask of rows with all-finite values."""
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] == 0:
        return np.array([], dtype=bool)
    return np.isfinite(emb).all(axis=1)


def _create_thumbnail_images(
    documents: list[Any],
    dataset_name: str,
    criterion: str | None,
    output_dir: Path,
) -> list[str]:
    """Create/reuse thumbnail images for marker visualization."""
    if output_dir.exists():
        existing_images = list(output_dir.glob("marker_*.*")) + list(
            output_dir.glob("thumb_*.*")
        )
        if len(existing_images) == len(documents):
            return sorted([str(p) for p in existing_images])

    if dataset_name == "gsm8k" and criterion in ["arithmetic", "final_expression"]:
        return create_gsm8k_marker_images(
            documents,
            str(output_dir),
            show_question=False,
            figsize=(16, 12),
            dpi=300,
            minimal=True,
        )

    if dataset_name in ["ut_zappos50k", "met_museum", "example_images"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for idx, doc in enumerate(documents):
            if not isinstance(doc, dict):
                image_paths.append(None)
                continue
            image_path = doc.get("image_path")
            if not image_path:
                image_paths.append(None)
                continue
            if isinstance(image_path, str) and image_path.startswith("data:image"):
                decoded = _decode_data_uri_image(image_path)
                if decoded is None:
                    image_paths.append(None)
                    continue
                ext, image_bytes = decoded
                dest_path = output_dir / f"thumb_{idx}.{ext}"
                if not dest_path.exists():
                    with open(dest_path, "wb") as f:
                        f.write(image_bytes)
                image_paths.append(str(dest_path))
                continue
            image_paths.append(image_path)
        return image_paths

    return []


def _prepare_markers(
    documents: list[Any],
    dataset_name: str | None,
    criterion: str | None,
    args: Any,
) -> tuple[list[str] | None, list[str] | None]:
    """Prepare image markers/labels."""
    image_paths = None
    labels = None
    if args.marker_type == "thumbnail" and dataset_name:
        output_dir = Path(args.output).parent / "_markers"
        image_paths = _create_thumbnail_images(
            documents=documents,
            dataset_name=dataset_name,
            criterion=criterion,
            output_dir=output_dir,
        )
        if not image_paths:
            image_paths = None
    elif args.marker_type == "text" or args.show_text_labels:
        labels = [str(i) for i in range(len(documents))]
    return image_paths, labels


def _create_reducer(args: Any, n_samples: int | None = None):
    """Create dimensionality reducer."""
    if args.reducer == "tsne":
        return TSNEReducer(
            perplexity=args.perplexity,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
    if args.reducer == "pca":
        return PCAReducer(
            n_components=2,
            whiten=True,
            random_state=args.random_state,
        )
    if args.reducer == "umap":
        return UMAPReducer(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
        )
    if args.reducer == "som":
        if args.som_grid_size == "auto" and n_samples is not None:
            import math

            target_cells = int(n_samples * 1.1)
            side = int(math.sqrt(target_cells))
            rows = side
            cols = math.ceil(target_cells / side)
        elif args.som_grid_size == "auto":
            rows, cols = 20, 20
        else:
            rows, cols = map(int, args.som_grid_size.split(","))
        return SOMReducer(
            grid_size=(rows, cols),
            learning_rate=args.som_learning_rate,
            iterations=args.som_iterations,
            random_state=args.random_state,
            unique_assignment=args.som_unique_assignment,
        )
    if args.reducer == "dendrogram":
        return DendrogramReducer(
            method=args.dendrogram_method,
            metric=args.dendrogram_metric,
            optimal_ordering=True,
        )
    if args.reducer == "heatmap":
        return None
    raise ValueError(f"Unknown reducer: {args.reducer}")


def _create_som_grid_composite(
    visualizer: CorpusVisualizer,
    reducer: SOMReducer,
    embeddings: np.ndarray,
    image_paths: list[str],
    args: Any,
) -> str:
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    if reducer.weights is None:
        reducer._coords = reducer._build_coords()
        reducer._initialize_weights(embeddings_arr)
    reducer._train(embeddings_arr)
    assignments = reducer.assign_unique_nodes(embeddings_arr)

    bg_hex = args.som_background_color.lstrip("#")
    if len(bg_hex) == 3:
        bg_hex = "".join(ch * 2 for ch in bg_hex)
    background_color = tuple(int(bg_hex[i : i + 2], 16) for i in (0, 2, 4))

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
    grid_image.save(output_file, format=args.format.upper())
    return output_file


def _create_dendrogram_plot(
    visualizer: CorpusVisualizer,
    embeddings: np.ndarray,
    image_paths: list[str],
    figsize: tuple[float, float],
    args: Any,
) -> str:
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    visualizer.reducer.fit_transform(embeddings_arr)
    fig, _ax = visualizer.plot_dendrogram_with_images(
        image_paths=image_paths,
        figsize=figsize,
        title=args.title,
        image_size=args.dendrogram_image_size,
        orientation=args.dendrogram_orientation,
        images_per_row=args.dendrogram_images_per_row,
        num_clusters=args.dendrogram_num_clusters,
    )
    output_file = f"{args.output}.{args.format}"
    fig.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    return output_file


def _visualize(
    embeddings: np.ndarray,
    documents: list[Any],
    classes: list[str] | None,
    image_paths: list[str] | None,
    labels: list[str] | None,
    reducer,
    args: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, str | None]:
    """Create visualization and return layout/artifact paths."""
    figsize = tuple(map(float, args.figsize.split(",")))
    visualizer = CorpusVisualizer(reducer=reducer)
    coords_2d = None
    linkage_matrix = None
    dendrogram_image_path = None
    som_grid_image_path = None

    if args.reducer == "dendrogram":
        if not image_paths:
            raise ValueError("Dendrogram visualization requires thumbnail markers.")
        dendrogram_image_path = _create_dendrogram_plot(
            visualizer, embeddings, image_paths, figsize, args
        )
        if hasattr(visualizer.reducer, "linkage_matrix"):
            linkage_matrix = visualizer.reducer.linkage_matrix
        if hasattr(visualizer.reducer, "coords_2d_"):
            coords_2d = visualizer.reducer.coords_2d_
    elif args.reducer == "som" and image_paths and isinstance(reducer, SOMReducer):
        som_grid_image_path = _create_som_grid_composite(
            visualizer, reducer, embeddings, image_paths, args
        )
        if hasattr(reducer, "_coords"):
            embeddings_arr = np.array(embeddings, dtype=np.float32)
            assignments = reducer.assign_unique_nodes(embeddings_arr)
            coords_2d = reducer._coords[assignments]
    else:
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

    return coords_2d, linkage_matrix, dendrogram_image_path, som_grid_image_path


def _export_for_web_viewer(
    documents: list[Any],
    embeddings: np.ndarray,
    coords_2d: np.ndarray | None,
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
) -> None:
    """Export visualization artifacts for web viewer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
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

    # Always refresh core manifest metadata so reruns can't leave stale values.
    manifest["version"] = 1
    manifest["dataset"] = dataset_name
    manifest["criterion"] = criterion
    manifest["n_docs"] = len(documents)
    manifest["embedding_dim"] = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    manifest["documents_path"] = "documents.txt"
    manifest["embeddings_path"] = "embeddings.npy"
    manifest.setdefault("layouts", {})
    if method is not None:
        manifest["method"] = method

    documents_path = output_dir / "documents.txt"
    with open(documents_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(_doc_to_viewer_text(doc).replace("\n", "\\n") + "\n")

    embeddings_path = output_dir / "embeddings.npy"
    # Always overwrite embeddings so filtered/rerun outputs are reflected immediately.
    np.save(embeddings_path, embeddings)

    if coords_2d is not None:
        layout_path = output_dir / f"layout_{reducer_name}.npy"
        np.save(layout_path, coords_2d)
        manifest["layouts"][reducer_name] = f"layout_{reducer_name}.npy"
    elif reducer_name == "heatmap":
        manifest["layouts"]["heatmap"] = None

    if linkage_matrix is not None:
        linkage_path = output_dir / "linkage_matrix.npy"
        np.save(linkage_path, linkage_matrix)
        manifest["linkage_matrix"] = "linkage_matrix.npy"

    if dendrogram_image and Path(dendrogram_image).exists():
        dendrogram_dest = output_dir / "dendrogram.png"
        shutil.copy2(dendrogram_image, dendrogram_dest)
        manifest["dendrogram_image"] = "dendrogram.png"

    if som_grid_image and Path(som_grid_image).exists():
        som_dest = output_dir / "som_grid.png"
        shutil.copy2(som_grid_image, som_dest)
        manifest["som_grid_image"] = "som_grid.png"

    if image_paths:
        thumbnail_refs = []
        thumbnails_dir = output_dir / "thumbnails"
        thumbnails_dir.mkdir(exist_ok=True)
        for i, img_path in enumerate(image_paths):
            if not img_path:
                thumbnail_refs.append(None)
                continue
            if img_path.startswith("http://") or img_path.startswith("https://"):
                thumbnail_refs.append(img_path)
                continue
            if img_path.startswith("data:image"):
                decoded = _decode_data_uri_image(img_path)
                if decoded is None:
                    thumbnail_refs.append(None)
                    continue
                ext, img_bytes = decoded
                dest_name = f"thumb_{i}.{ext}"
                dest_path = thumbnails_dir / dest_name
                with open(dest_path, "wb") as f:
                    f.write(img_bytes)
                thumbnail_refs.append(f"thumbnails/{dest_name}")
                continue
            if _safe_path_exists(img_path):
                ext = Path(img_path).suffix
                dest_name = f"thumb_{i}{ext}"
                dest_path = thumbnails_dir / dest_name
                shutil.copy2(img_path, dest_path)
                thumbnail_refs.append(f"thumbnails/{dest_name}")
                continue
            thumbnail_refs.append(None)
        manifest["thumbnails"] = thumbnail_refs

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    if not quiet:
        logger.debug(f"Manifest: wrote {manifest_path}")


def _run_benchmark_embedding_mode(args: Any) -> None:
    """Run benchmark embedding visualization pipeline."""
    if not getattr(args, "from_benchmark", None):
        raise ValueError(
            "run_embedding_mode in visualization_utils requires benchmark mode"
        )
    quiet = getattr(args, "quiet", False)
    preloaded_data = getattr(args, "preloaded_data", None)
    if preloaded_data is not None:
        documents, doc_texts, dataset_name, embeddings, loaded_criterion = (
            preloaded_data
        )
    else:
        preferred_view = _infer_preferred_view(
            getattr(args, "output_benchmark_run", None)
        )
        documents, doc_texts, dataset_name, embeddings, loaded_criterion = (
            _load_from_benchmark(
                args.from_benchmark,
                args.task,
                args.method,
                preferred_view=preferred_view,
            )
        )
    embeddings = np.array(embeddings, dtype=np.float32)
    finite_mask = _finite_embedding_row_mask(embeddings)
    if finite_mask.size == 0:
        raise ValueError("Embeddings are empty or malformed; cannot visualize.")
    if not finite_mask.any():
        raise ValueError("All embedding rows contain NaN/Inf; cannot visualize.")
    if not finite_mask.all():
        dropped = int((~finite_mask).sum())
        logger.info(
            "Dropping %d/%d documents with NaN/Inf embeddings from visualization.",
            dropped,
            embeddings.shape[0],
        )
        embeddings = embeddings[finite_mask]
        documents = [doc for i, doc in enumerate(documents) if finite_mask[i]]
        doc_texts = [text for i, text in enumerate(doc_texts) if finite_mask[i]]

    classes = None
    if args.annotations_file:
        classes = load_annotation_classes(args.annotations_file, args.color_by)
        if not finite_mask.all():
            classes = [
                c
                for i, c in enumerate(classes)
                if i < len(finite_mask) and finite_mask[i]
            ]
        classes = classes[: len(documents)]
    elif getattr(args, "color_by_benchmark_annotations", False):
        annotations_file = (
            Path("outputs") / args.from_benchmark / "annotations" / f"{args.task}.jsonl"
        )
        if annotations_file.exists():
            classes = load_annotation_classes(str(annotations_file), args.color_by)
            if not finite_mask.all():
                classes = [
                    c
                    for i, c in enumerate(classes)
                    if i < len(finite_mask) and finite_mask[i]
                ]
            classes = classes[: len(documents)]

    image_paths, labels = _prepare_markers(
        documents, dataset_name, loaded_criterion, args
    )
    reducer = _create_reducer(args, n_samples=len(embeddings))

    export_format = args.export_format or args.format
    should_export_plot = (
        export_format in ["png", "svg", "all"] or not args.export_format
    )
    should_export_web = export_format in ["web", "all"]

    coords_2d = None
    linkage_matrix = None
    dendrogram_image_path = None
    som_grid_image_path = None
    if should_export_plot:
        coords_2d, linkage_matrix, dendrogram_image_path, som_grid_image_path = (
            _visualize(
                embeddings, documents, classes, image_paths, labels, reducer, args
            )
        )
    else:
        if args.reducer == "dendrogram" and should_export_web:
            coords_2d, linkage_matrix, dendrogram_image_path, som_grid_image_path = (
                _visualize(
                    embeddings, documents, classes, image_paths, labels, reducer, args
                )
            )
        elif args.reducer == "som" and image_paths and should_export_web:
            coords_2d, linkage_matrix, dendrogram_image_path, som_grid_image_path = (
                _visualize(
                    embeddings, documents, classes, image_paths, labels, reducer, args
                )
            )
        elif args.reducer != "heatmap":
            embeddings_arr = np.array(embeddings, dtype=np.float32)
            coords_2d = reducer.fit_transform(embeddings_arr)
            if args.reducer == "dendrogram" and hasattr(reducer, "linkage_matrix"):
                linkage_matrix = reducer.linkage_matrix

    if should_export_web:
        if coords_2d is None and args.reducer != "heatmap":
            logger.error("No 2D coordinates available for web export")
        else:
            criterion = loaded_criterion or args.criterion or "default"
            if not dataset_name:
                dataset_name = Path(args.output).stem
            _export_for_web_viewer(
                documents=doc_texts,
                embeddings=embeddings,
                coords_2d=coords_2d,
                reducer_name=args.reducer,
                output_dir=Path(args.output),
                dataset_name=dataset_name,
                criterion=criterion,
                method=getattr(args, "method", None),
                linkage_matrix=linkage_matrix,
                image_paths=image_paths,
                dendrogram_image=dendrogram_image_path,
                som_grid_image=som_grid_image_path,
                quiet=quiet,
            )


def find_tasks_and_methods(benchmark_run: str) -> dict[str, list[str]]:
    """Find all task/method combinations in a benchmark run.

    Args:
        benchmark_run: Benchmark run name

    Returns:
        Dict mapping task names to list of method names
    """
    output_base = Path("outputs") / benchmark_run
    method_logs_dir = output_base / "method_logs"

    if not method_logs_dir.exists():
        raise FileNotFoundError(f"No method logs found in {benchmark_run}")

    task_methods = {}

    for task_dir in sorted(method_logs_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        methods = []

        # Find all method log files
        for method_file in sorted(task_dir.glob("*.jsonl")):
            method_name = method_file.stem
            methods.append(method_name)

        if methods:
            task_methods[task_name] = methods

    return task_methods


def should_use_thumbnails(task_name: str, use_thumbnails: bool) -> bool:
    """Determine if task should use thumbnail markers.

    Args:
        task_name: Task name (e.g., 'gsm8k__arithmetic')
        use_thumbnails: User-specified thumbnail preference

    Returns:
        True if thumbnails should be used
    """
    if not use_thumbnails:
        return False

    # Extract dataset name from task
    dataset_name = task_name.split("__")[0]
    criterion = task_name.split("__")[1] if "__" in task_name else None

    # GSM8K with computational criteria uses computational graphs
    if dataset_name == "gsm8k" and criterion in ["arithmetic", "final_expression"]:
        return True

    # Image datasets use actual images
    if dataset_name in ["ut_zappos50k", "met_museum", "example_images"]:
        return True

    return False


def is_embedding_method(method_name: str) -> bool:
    """Check if a method is expected to produce embeddings.

    Args:
        method_name: Method name

    Returns:
        True if method produces embeddings, False for BM25/rerank methods
    """
    # Methods that don't produce embeddings (sparse retrieval, rerankers, etc.)
    non_embedding_patterns = [
        "bm25",
        "rerank",
        "lexical",
        "triplet",  # LM judge methods
    ]
    method_lower = method_name.lower()
    return not any(pattern in method_lower for pattern in non_embedding_patterns)


def copy_triplet_logs_for_method(
    benchmark_run: str,
    output_benchmark_run: str,
    task_name: str,
    method_name: str,
    output_base: Path,
) -> bool:
    """Copy triplet files (triplets.json and triplet_logs.jsonl) to visualization directory.

    Args:
        benchmark_run: Benchmark run name to read source artifacts from
        output_benchmark_run: Benchmark run name to write visualization artifacts under
        task_name: Task name
        method_name: Method name
        output_base: Base output directory

    Returns:
        True if at least one file was copied successfully
    """
    import shutil

    output_dir = output_base / output_benchmark_run / task_name / method_name
    output_dir.mkdir(parents=True, exist_ok=True)

    success = False

    preferred_view = _infer_preferred_view(output_benchmark_run)
    output_base_src = Path("outputs") / benchmark_run
    candidate_files = _embedding_candidate_files(
        output_base=output_base_src,
        task_name=task_name,
        method_name=method_name,
        preferred_view=preferred_view,
    )
    embedding_file = next((p for p in candidate_files if p.exists()), None)
    id_map: dict[int, int] | None = None
    if embedding_file is not None:
        emb = np.load(embedding_file)
        mask = _finite_embedding_row_mask(emb)
        if mask.size > 0 and not mask.all() and mask.any():
            kept_ids = np.where(mask)[0].tolist()
            id_map = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}

    def _remap_triplet_ids(entry: dict[str, Any]) -> dict[str, Any] | None:
        if id_map is None:
            return entry
        a = entry.get("anchor_id")
        p = entry.get("positive_id")
        n = entry.get("negative_id")
        if a not in id_map or p not in id_map or n not in id_map:
            return None
        out = dict(entry)
        out["anchor_id"] = id_map[a]
        out["positive_id"] = id_map[p]
        out["negative_id"] = id_map[n]
        return out

    # Copy or filter triplet_logs.jsonl (per-method evaluation results with margin/correctness)
    logs_source = (
        Path("outputs")
        / benchmark_run
        / "method_logs"
        / task_name
        / f"{method_name}.jsonl"
    )

    if logs_source.exists():
        logs_dest = output_dir / "triplet_logs.jsonl"
        if id_map is None:
            shutil.copy2(logs_source, logs_dest)
        else:
            kept_lines: list[str] = []
            with open(logs_source) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    remapped = _remap_triplet_ids(entry)
                    if remapped is not None:
                        kept_lines.append(json.dumps(remapped))
            with open(logs_dest, "w") as f:
                for line in kept_lines:
                    f.write(line + "\n")
        success = True

    # Copy or filter triplets.json (shared across all methods for a task)
    triplets_source = (
        Path("outputs") / benchmark_run / "triplets" / task_name / "triplets.json"
    )

    if triplets_source.exists():
        triplets_dest = output_dir / "triplets.json"
        if id_map is None:
            shutil.copy2(triplets_source, triplets_dest)
        else:
            with open(triplets_source) as f:
                triplets = json.load(f)
            filtered_triplets = []
            for triplet in triplets:
                remapped = _remap_triplet_ids(triplet)
                if remapped is not None:
                    filtered_triplets.append(remapped)
            with open(triplets_dest, "w") as f:
                json.dump(filtered_triplets, f, indent=2)
        success = True

    return success


def is_similarity_matrix_method(method_name: str) -> bool:
    """Check if a method produces similarity matrices rather than embeddings.

    Args:
        method_name: Method name

    Returns:
        True if method produces NxN similarity matrices (e.g., BM25)
    """
    # Methods that produce similarity matrices instead of embeddings
    similarity_matrix_patterns = [
        "bm25",
        "lexical",
    ]
    method_lower = method_name.lower()
    return any(pattern in method_lower for pattern in similarity_matrix_patterns)


def run_visualization(
    benchmark_run: str,
    output_benchmark_run: str,
    task_name: str,
    method_name: str,
    reducer: str,
    output_base: Path,
    use_thumbnails: bool = False,
    quiet: bool = False,
    preloaded_data: tuple[list[Any], list[str], str, np.ndarray, str] | None = None,
) -> bool | None:
    """Run visualization for a single task/method/reducer combination.

    Args:
        benchmark_run: Benchmark run name to read source artifacts from
        output_benchmark_run: Benchmark run name to write visualization artifacts under
        task_name: Task name
        method_name: Method name
        reducer: Reducer name
        output_base: Base output directory
        use_thumbnails: Whether to use thumbnail markers
        quiet: Suppress output (capture_output=True)

    Returns:
        True if successful, False if failed, None if skipped (no embeddings expected)
    """
    # Check if this is a similarity matrix method (like BM25)
    # These can only be visualized as heatmaps, not with dimensionality reduction
    if is_similarity_matrix_method(method_name):
        if reducer != "heatmap":
            # Silently skip non-heatmap reducers for similarity matrix methods
            return None

    # Skip methods that are known not to produce embeddings and are not handled
    # by the similarity-matrix heatmap path.
    if not is_embedding_method(method_name) and not is_similarity_matrix_method(
        method_name
    ):
        return None

    # Construct output path: outputs/viz/{benchmark}/{task}/{method}/
    output_dir = output_base / output_benchmark_run / task_name / method_name

    # Only log per-item details if not quiet
    # (Summary logs at end are still shown)

    try:
        # Determine if we should use thumbnails
        use_thumbnails_for_task = should_use_thumbnails(task_name, use_thumbnails)

        # Call the function directly
        success = visualize_benchmark_task(
            benchmark_run=benchmark_run,
            task_name=task_name,
            method_name=method_name,
            reducer=reducer,
            output_dir=output_dir,
            output_benchmark_run=output_benchmark_run,
            use_thumbnails=use_thumbnails_for_task,
            quiet=quiet,
            preloaded_data=preloaded_data,
        )

        if success:
            # Don't log per-item success (summary at end is sufficient)
            return True
        else:
            logger.error(f"✗ Failed to visualize {task_name}/{method_name}/{reducer}")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to visualize {task_name}/{method_name}/{reducer}: {e}")
        return False


def create_visualization_index(
    output_base: Path,
    benchmark_run: str,
    task_methods: dict,
    output_benchmark_run: str | None = None,
) -> Path:
    """Create/update index.json for web viewer.

    Supports multiple benchmark runs by merging into existing index.

    Args:
        output_base: Base output directory (e.g., outputs/viz/)
        benchmark_run: Benchmark run name to read source artifacts from
        task_methods: Dict of task -> methods
        output_benchmark_run: Benchmark run name to write in index/output paths.
            Defaults to benchmark_run.

    Returns:
        Path to created/updated index.json
    """
    output_run = output_benchmark_run or benchmark_run
    index_path = output_base / "index.json"

    # Load existing index if it exists
    if index_path.exists():
        with open(index_path) as f:
            existing_index = json.load(f)
    else:
        existing_index = {}

    benchmark_dir = output_base / output_run

    # Build index for this benchmark run
    # Structure: index[benchmark_run][dataset][criterion][method] = {modes: [...], path: "..."}
    benchmark_index = {}

    # Scan for generated visualizations
    for task_name in task_methods:
        task_dir = benchmark_dir / task_name
        if not task_dir.exists():
            continue

        for method_name in task_methods[task_name]:
            method_dir = task_dir / method_name
            if not method_dir.exists():
                continue

            # Check for manifest.json
            manifest_path = method_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            # Read manifest to get available modes
            with open(manifest_path) as f:
                manifest = json.load(f)

            modes_set = set(manifest.get("layouts", {}).keys())

            # Always add heatmap and graph - they don't require layout coords
            modes_set.add("heatmap")
            modes_set.add("graph")

            # Add dendrogram if dendrogram image exists (even without layout coords)
            if manifest.get("dendrogram_image"):
                modes_set.add("dendrogram")

            # Add SOM if SOM grid image exists (even without layout coords)
            if manifest.get("som_grid_image"):
                modes_set.add("som")

            modes = sorted(modes_set)

            if not modes:
                continue

            # Add to index
            dataset = manifest["dataset"]
            criterion = manifest["criterion"]

            if dataset not in benchmark_index:
                benchmark_index[dataset] = {}
            if criterion not in benchmark_index[dataset]:
                benchmark_index[dataset][criterion] = {}

            benchmark_index[dataset][criterion][method_name] = {
                "modes": modes,
                "path": f"{output_run}/{task_name}/{method_name}",
            }

    # Copy results.json if it exists
    results_file = Path("outputs") / benchmark_run / "results" / "results.json"
    if results_file.exists():
        import shutil

        dest = benchmark_dir / "results.json"
        shutil.copy2(results_file, dest)
        logger.info(f"Copied results to: {dest}")

    # Merge into existing index
    existing_index[output_run] = benchmark_index

    # Save merged index
    with open(index_path, "w") as f:
        json.dump(existing_index, f, indent=2)

    logger.info(f"Updated visualization index: {index_path}")
    logger.info(f"  Total benchmark runs: {len(existing_index)}")
    return index_path


def generate_visualizations_for_benchmark(
    benchmark_run: str,
    reducers: list[str] | None = None,
    output_base: str | Path = "outputs/viz",
    output_benchmark_run: str | None = None,
    use_thumbnails: bool = True,
    include_triplets: bool = True,
    task_filter: list[str] | None = None,
    method_filter: list[str] | None = None,
    quiet: bool = False,
) -> tuple[int, int]:
    """Generate all visualizations for a benchmark run.

    Args:
        benchmark_run: Benchmark run name to read source artifacts from
        reducers: List of reducers to use (default: ['tsne', 'umap', 'pca', 'som', 'dendrogram'])
        output_base: Base output directory
        output_benchmark_run: Benchmark run name to write visualization artifacts under.
            Defaults to benchmark_run.
        use_thumbnails: Whether to use thumbnail markers
        include_triplets: Whether to copy triplet artifacts into viz outputs
        task_filter: Optional list of task names to include
        method_filter: Optional list of method names to include
        quiet: Suppress verbose output
    Returns:
        Tuple of (success_count, fail_count)
    """
    if reducers is None:
        reducers = ["tsne", "umap", "pca", "som", "dendrogram", "heatmap"]

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    output_run = output_benchmark_run or benchmark_run

    # Clear existing visualizations for this benchmark run
    # outputs/viz is derived from outputs/, so safe to delete stale artifacts
    # BUT preserve *_markers directories (expensive to regenerate thumbnails)
    benchmark_viz_dir = output_base / output_run
    if benchmark_viz_dir.exists():
        import shutil

        if not quiet:
            logger.info(
                f"Clearing stale visualizations (preserving thumbnails): {benchmark_viz_dir}"
            )
        # Save markers directories temporarily (task-level _markers directories)
        markers_dirs = list(benchmark_viz_dir.glob("*/_markers"))
        markers_backup = {}
        if markers_dirs:
            import tempfile

            backup_root = Path(tempfile.mkdtemp(prefix="viz_markers_"))
            for markers_dir in markers_dirs:
                rel_path = markers_dir.relative_to(benchmark_viz_dir)
                backup_path = backup_root / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(markers_dir), str(backup_path))
                markers_backup[rel_path] = backup_path
            if not quiet:
                logger.info(f"  Preserved {len(markers_dirs)} thumbnail directories")

        # Delete and recreate
        shutil.rmtree(benchmark_viz_dir)
        benchmark_viz_dir.mkdir(parents=True, exist_ok=True)

        # Restore markers directories
        for rel_path, backup_path in markers_backup.items():
            restore_path = benchmark_viz_dir / rel_path
            restore_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup_path), str(restore_path))

        # Clean up backup
        if markers_backup:
            shutil.rmtree(backup_root)

    # Find all tasks and methods
    if not quiet:
        logger.info(f"Scanning benchmark run: {benchmark_run}")

    task_methods = find_tasks_and_methods(benchmark_run)

    if not task_methods:
        logger.warning(f"No tasks found in {benchmark_run}")
        return 0, 0

    # Apply filters
    if task_filter:
        task_filter_set = set(task_filter)
        task_methods = {k: v for k, v in task_methods.items() if k in task_filter_set}

    if method_filter:
        method_filter_set = set(method_filter)
        task_methods = {
            k: [m for m in v if m in method_filter_set] for k, v in task_methods.items()
        }
        # Remove tasks with no matching methods
        task_methods = {k: v for k, v in task_methods.items() if v}

    if not quiet:
        logger.info("=" * 60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 60)
        logger.info(f"Benchmark: {benchmark_run}")
        if output_run != benchmark_run:
            logger.info(f"Output benchmark key: {output_run}")
        logger.info(f"Tasks: {len(task_methods)}")
        for task, methods in task_methods.items():
            logger.info(f"  {task}: {len(methods)} methods")
        logger.info(f"Reducers: {', '.join(reducers)}")
        logger.info(f"Output: {output_base}")
        logger.info(f"Thumbnails: {use_thumbnails}")
        logger.info(f"Include triplets: {include_triplets}")
        logger.info("")

    # Calculate total
    total = sum(len(methods) * len(reducers) for methods in task_methods.values())
    success_count = 0
    fail_count = 0
    skip_count = 0

    # Run all combinations
    combo_num = 0
    for task_name, methods in task_methods.items():
        for method_name in methods:
            preloaded_data: tuple[list[Any], list[str], str, np.ndarray, str] | None = (
                None
            )
            # Embedding methods share the same docs/embeddings across reducers.
            if is_embedding_method(method_name):
                try:
                    preferred_view = _infer_preferred_view(output_run)
                    preloaded_data = _load_from_benchmark(
                        benchmark_run,
                        task_name,
                        method_name,
                        preferred_view=preferred_view,
                    )
                except Exception as e:
                    if not quiet:
                        logger.warning(
                            f"Preload failed for {task_name}/{method_name}: {e}. "
                            "Falling back to per-reducer loading."
                        )

            for reducer in reducers:
                combo_num += 1
                if not quiet:
                    logger.info(
                        f"[{combo_num}/{total}] task={task_name} method={method_name} reducer={reducer}"
                    )

                result = run_visualization(
                    benchmark_run=benchmark_run,
                    output_benchmark_run=output_run,
                    task_name=task_name,
                    method_name=method_name,
                    reducer=reducer,
                    output_base=output_base,
                    use_thumbnails=use_thumbnails,
                    quiet=quiet,
                    preloaded_data=preloaded_data,
                )

                if result is True:
                    success_count += 1
                elif result is False:
                    fail_count += 1
                else:  # None - skipped
                    skip_count += 1

            # Copy triplet artifacts for sidebar interactions unless corpus-only mode is requested
            if include_triplets:
                copy_triplet_logs_for_method(
                    benchmark_run=benchmark_run,
                    output_benchmark_run=output_run,
                    task_name=task_name,
                    method_name=method_name,
                    output_base=output_base,
                )

    # Create index for web viewer
    if not quiet:
        logger.info("Creating web viewer index...")
    create_visualization_index(
        output_base=output_base,
        benchmark_run=benchmark_run,
        task_methods=task_methods,
        output_benchmark_run=output_run,
    )

    # Print summary
    if not quiet:
        logger.info("")
        logger.info("=" * 60)
        logger.info("VISUALIZATION GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total: {total}")
        logger.info(f"Success: {success_count}")
        logger.info(f"Skipped: {skip_count} (methods without embeddings)")
        if fail_count > 0:
            logger.info(f"Failed: {fail_count}")
        logger.info(f"Output: {output_base / output_run}")
        logger.info("")

    return success_count, fail_count
