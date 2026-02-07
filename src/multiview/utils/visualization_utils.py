"""Shared utilities for visualization scripts."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def visualize_benchmark_task(
    benchmark_run: str,
    task_name: str,
    method_name: str,
    reducer: str = "tsne",
    output_dir: str | Path | None = None,
    use_thumbnails: bool = False,
    quiet: bool = False,
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
        run_embedding_mode(args)
        return True
    except Exception as e:
        logger.error(f"Failed to visualize {task_name}/{method_name}: {e}")
        return False


def run_embedding_mode(args: Any) -> None:
    """Run embedding visualization mode using shared helpers from the CLI module."""
    quiet = getattr(args, "quiet", False)
    if not quiet:
        logger.debug(f"CORPUS VISUALIZATION: {args.reducer.upper()}")

    scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from analyze_corpus import (
        create_reducer,
        export_for_web_viewer,
        generate_embeddings,
        load_annotation_classes,
        load_documents,
        prepare_markers,
        print_summary,
        visualize,
    )

    # Step 1: Load documents (and possibly embeddings from benchmark)
    documents, doc_texts, dataset_name, embeddings, loaded_criterion = load_documents(
        args
    )

    # Set criterion from loaded data if not already set
    if loaded_criterion and not args.criterion:
        args.criterion = loaded_criterion

    # Step 2: Generate embeddings if not already loaded
    if embeddings is None:
        if not quiet:
            logger.debug("Generating fresh embeddings")
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
        if not quiet:
            logger.debug(f"Loading annotations from {args.annotations_file}...")
        classes = load_annotation_classes(args.annotations_file, args.color_by)
        classes = classes[: len(documents)]
    elif (
        hasattr(args, "from_benchmark")
        and args.from_benchmark
        and hasattr(args, "color_by_benchmark_annotations")
        and args.color_by_benchmark_annotations
    ):
        # Load annotations from benchmark artifacts
        output_base = Path("outputs") / args.from_benchmark
        annotations_file = output_base / "annotations" / f"{args.task}.jsonl"
        if annotations_file.exists():
            classes = load_annotation_classes(str(annotations_file), args.color_by)
            classes = classes[: len(documents)]
            if not quiet:
                logger.debug(
                    f"Loaded benchmark annotations from {annotations_file} "
                    f"(color_by={args.color_by})"
                )
        elif not quiet:
            logger.warning(
                f"Benchmark annotations not found: {annotations_file}. "
                "Proceeding without color classes."
            )

    # Step 4: Prepare markers
    image_paths, labels = prepare_markers(documents, dataset_name, args)

    # Step 5: Create reducer
    if not quiet:
        logger.debug(f"Initializing {args.reducer} reducer...")
    reducer = create_reducer(args, n_samples=len(embeddings))

    # Determine export mode
    export_format = args.export_format or args.format
    should_export_plot = (
        export_format in ["png", "svg", "all"] or not args.export_format
    )
    should_export_web = export_format in ["web", "all"]

    # Step 6: Visualize (if generating plots)
    coords_2d = None
    linkage_matrix = None
    dendrogram_image_path = None
    som_grid_image_path = None
    if should_export_plot:
        coords_2d, linkage_matrix, dendrogram_image_path, som_grid_image_path = (
            visualize(
                embeddings, documents, classes, image_paths, labels, reducer, args
            )
        )
    else:
        # For web-only export, still need to compute coordinates
        # For dendrogram/SOM with images, we also need to generate the matplotlib image for web viewer
        if args.reducer == "dendrogram" and should_export_web:
            if not quiet:
                logger.debug("Generating dendrogram matplotlib image for web export...")
            coords_2d, linkage_matrix, dendrogram_image_path, som_grid_image_path = (
                visualize(
                    embeddings, documents, classes, image_paths, labels, reducer, args
                )
            )
        elif args.reducer == "som" and image_paths and should_export_web:
            if not quiet:
                logger.debug("Generating SOM grid composite for web export...")
            coords_2d, linkage_matrix, dendrogram_image_path, som_grid_image_path = (
                visualize(
                    embeddings, documents, classes, image_paths, labels, reducer, args
                )
            )
        else:
            if args.reducer == "heatmap":
                if not quiet:
                    logger.debug("Heatmap mode - skipping dimensionality reduction")
                coords_2d = None
            else:
                if not quiet:
                    logger.debug(f"Reducing {len(embeddings)} embeddings to 2D...")
                embeddings_arr = np.array(embeddings, dtype=np.float32)
                coords_2d = reducer.fit_transform(embeddings_arr)
                if args.reducer == "dendrogram" and hasattr(reducer, "linkage_matrix"):
                    linkage_matrix = reducer.linkage_matrix

    # Step 7: Export for web viewer if requested
    if should_export_web:
        if coords_2d is None and args.reducer != "heatmap":
            logger.error("No 2D coordinates available for web export")
        else:
            criterion = loaded_criterion or args.criterion or "default"
            if not dataset_name:
                dataset_name = Path(args.output).stem

            web_output_dir = Path(args.output)
            method = getattr(args, "method", None)
            export_for_web_viewer(
                documents=doc_texts,
                embeddings=embeddings,
                coords_2d=coords_2d,
                reducer_name=args.reducer,
                output_dir=web_output_dir,
                dataset_name=dataset_name,
                criterion=criterion,
                method=method,
                linkage_matrix=linkage_matrix,
                image_paths=image_paths,
                dendrogram_image=dendrogram_image_path,
                som_grid_image=som_grid_image_path,
                quiet=quiet,
            )

    # Step 8: Print summary
    if should_export_plot:
        output_dir = (
            Path(args.output).parent / f"{Path(args.output).stem}_markers"
            if image_paths
            else None
        )
        print_summary(args, image_paths, output_dir)
    elif should_export_web and not quiet:
        logger.debug(f"Output: {args.output}/")


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


def has_embeddings(benchmark_run: str, task_name: str, method_name: str) -> bool:
    """Check if embeddings exist for a task/method combination.

    Args:
        benchmark_run: Benchmark run name
        task_name: Task name
        method_name: Method name

    Returns:
        True if embeddings file exists
    """
    embeddings_file = (
        Path("outputs")
        / benchmark_run
        / "embeddings"
        / task_name
        / f"{method_name}.npy"
    )
    return embeddings_file.exists()


def copy_triplet_logs_for_method(
    benchmark_run: str,
    task_name: str,
    method_name: str,
    output_base: Path,
) -> bool:
    """Copy triplet files (triplets.json and triplet_logs.jsonl) to visualization directory.

    Args:
        benchmark_run: Benchmark run name
        task_name: Task name
        method_name: Method name
        output_base: Base output directory

    Returns:
        True if at least one file was copied successfully
    """
    import shutil

    output_dir = output_base / benchmark_run / task_name / method_name
    output_dir.mkdir(parents=True, exist_ok=True)

    success = False

    # Copy triplet_logs.jsonl (per-method evaluation results with margin/correctness)
    logs_source = (
        Path("outputs")
        / benchmark_run
        / "method_logs"
        / task_name
        / f"{method_name}.jsonl"
    )

    if logs_source.exists():
        logs_dest = output_dir / "triplet_logs.jsonl"
        shutil.copy2(logs_source, logs_dest)
        success = True

    # Copy triplets.json (shared across all methods for a task)
    triplets_source = (
        Path("outputs") / benchmark_run / "triplets" / task_name / "triplets.json"
    )

    if triplets_source.exists():
        triplets_dest = output_dir / "triplets.json"
        shutil.copy2(triplets_source, triplets_dest)
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
    task_name: str,
    method_name: str,
    reducer: str,
    output_base: Path,
    use_thumbnails: bool = False,
    quiet: bool = False,
) -> bool | None:
    """Run visualization for a single task/method/reducer combination.

    Args:
        benchmark_run: Benchmark run name
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

    # Check if embeddings exist first
    if not has_embeddings(benchmark_run, task_name, method_name):
        # Don't log as warning for methods that don't produce embeddings
        if not is_embedding_method(method_name):
            # Silently skip methods that don't produce embeddings
            return None
        else:
            # Warn about missing embeddings for methods that should have them
            if not quiet:
                logger.warning(
                    f"Skipping {task_name}/{method_name}/{reducer} - no embeddings found"
                )
            return None

    # Construct output path: outputs/viz/{benchmark}/{task}/{method}/
    output_dir = output_base / benchmark_run / task_name / method_name

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
            use_thumbnails=use_thumbnails_for_task,
            quiet=quiet,
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
    output_base: Path, benchmark_run: str, task_methods: dict
) -> Path:
    """Create/update index.json for web viewer.

    Supports multiple benchmark runs by merging into existing index.

    Args:
        output_base: Base output directory (e.g., outputs/viz/)
        benchmark_run: Benchmark run name
        task_methods: Dict of task -> methods

    Returns:
        Path to created/updated index.json
    """
    index_path = output_base / "index.json"

    # Load existing index if it exists
    if index_path.exists():
        with open(index_path) as f:
            existing_index = json.load(f)
    else:
        existing_index = {}

    benchmark_dir = output_base / benchmark_run

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
                "path": f"{benchmark_run}/{task_name}/{method_name}",
            }

    # Copy results.json if it exists
    results_file = Path("outputs") / benchmark_run / "results" / "results.json"
    if results_file.exists():
        import shutil

        dest = benchmark_dir / "results.json"
        shutil.copy2(results_file, dest)
        logger.info(f"Copied results to: {dest}")

    # Merge into existing index
    existing_index[benchmark_run] = benchmark_index

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
    use_thumbnails: bool = True,
    include_triplets: bool = True,
    task_filter: list[str] | None = None,
    method_filter: list[str] | None = None,
    quiet: bool = False,
) -> tuple[int, int]:
    """Generate all visualizations for a benchmark run.

    Args:
        benchmark_run: Benchmark run name
        reducers: List of reducers to use (default: ['tsne', 'umap', 'pca', 'som', 'dendrogram'])
        output_base: Base output directory
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

    # Clear existing visualizations for this benchmark run
    # outputs/viz is derived from outputs/, so safe to delete stale artifacts
    # BUT preserve *_markers directories (expensive to regenerate thumbnails)
    benchmark_viz_dir = output_base / benchmark_run
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
            for reducer in reducers:
                combo_num += 1
                if not quiet:
                    logger.info(f"\n[{combo_num}/{total}] Processing...")

                result = run_visualization(
                    benchmark_run=benchmark_run,
                    task_name=task_name,
                    method_name=method_name,
                    reducer=reducer,
                    output_base=output_base,
                    use_thumbnails=use_thumbnails,
                    quiet=quiet,
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
                    task_name=task_name,
                    method_name=method_name,
                    output_base=output_base,
                )

    # Create index for web viewer
    if not quiet:
        logger.info("\nCreating web viewer index...")
    create_visualization_index(output_base, benchmark_run, task_methods)

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
        logger.info(f"Output: {output_base / benchmark_run}")
        logger.info("")

    return success_count, fail_count
