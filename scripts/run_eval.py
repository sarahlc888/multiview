"""Run evaluation on pre-generated triplets."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from multiview.benchmark.artifacts import (
    load_documents_from_jsonl,
    load_schema_from_json,
)
from multiview.benchmark.benchmark import Benchmark
from multiview.benchmark.task import Task
from multiview.inference.cost_tracker import print_summary as print_cost_summary
from multiview.utils.script_utils import (
    save_benchmark_results,
    setup_benchmark_config,
)
from multiview.utils.visualization_utils import generate_visualizations_for_benchmark

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="benchmark", version_base=None)
def main(cfg: DictConfig):
    seed, output_base = setup_benchmark_config(cfg)
    logger.info(f"Running evaluation: {cfg.run_name} with {seed=}")

    # Setup output directories
    results_dir = output_base / "results"
    method_logs_dir = output_base / "method_logs"
    embeddings_dir = output_base / "triples" / "embeddings"
    results_dir.mkdir(parents=True, exist_ok=True)
    method_logs_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached triplets first
    logger.info(f"Looking for cached triplets in {output_base}")

    triplets_dir = output_base / "triplets"
    documents_dir = output_base / "documents"
    if not triplets_dir.exists() or not documents_dir.exists():
        logger.error(f"✗ Evaluation artifacts not found in {output_base}")
        logger.error("")
        logger.error("To generate evaluation artifacts, run:")
        logger.error(f"  python scripts/create_eval.py --config-name {cfg.run_name}")
        logger.error("")
        raise FileNotFoundError(
            "Missing evaluation artifacts. Run create_eval.py first."
        )

    # Load tasks from saved artifacts
    # create_eval.py saves artifacts to subdirectories (triplets/, documents/, etc.)
    triplets_cache_dir = str(output_base / "triplets")
    documents_cache_dir = str(output_base / "documents")

    tasks = []
    for task_spec in cfg.tasks.task_list:
        # Set triplet_cache_dir to point to the triplets subdirectory
        cur_task = Task(
            config={
                **cfg.tasks.defaults,
                **task_spec,
                "run_name": cfg.run_name,
                "triplet_cache_dir": triplets_cache_dir,
                "reuse_cached_triplets": cfg.get("reuse_cached_triplets", True),
            }
        )

        task_name = cur_task.get_task_name()

        # Check if cached triplets are available
        if not cur_task.can_use_cached_triplets():
            logger.error(f"✗ No cached triplets found for {task_name}")
            logger.error("")
            logger.error("To generate evaluation artifacts, run:")
            logger.error(
                f"  python scripts/create_eval.py --config-name {cfg.run_name}"
            )
            logger.error("")
            raise FileNotFoundError(
                f"Missing cached triplets for {task_name}. Run create_eval.py first."
            )

        logger.info(f"✓ Loading cached artifacts for {task_name}")
        # Load triplets directly without re-checking cache
        if not cur_task.try_load_cached_triplets(triplets_cache_dir):
            raise RuntimeError(f"Failed to load cached triplets for {task_name}")
        # Load cached documents (includes synthetic docs from previous run)
        # The cached triplet indices reference these exact documents
        cur_task.documents = load_documents_from_jsonl(
            output_dir=documents_cache_dir,
            task_name=task_name,
        )
        logger.info(
            f"  Loaded {len(cur_task.documents)} cached documents (including synthetic)"
        )

        # Load schema if available (needed for use_oracle_schema in evaluation)
        schema = load_schema_from_json(
            output_dir=triplets_cache_dir,
            task_name=task_name,
        )
        if schema:
            # Create minimal annotation list with just the schema in first element
            cur_task.document_annotations = [schema]
            logger.info("  Loaded schema for oracle-based methods")

        tasks.append(cur_task)

    # create benchmark object
    benchmark = Benchmark(
        tasks,
        method_log_output_dir=str(method_logs_dir),
        embeddings_output_dir=str(embeddings_dir),
    )

    # evaluate multiview representation methods
    results, instruction_sensitivity = benchmark.evaluate(cfg.methods_to_evaluate)

    # Save results in multiple formats and log summary
    save_benchmark_results(results, results_dir, cfg.run_name, instruction_sensitivity)

    # Print API cost summary
    print_cost_summary()

    # Add embedding visualizations to triplets
    if cfg.get("auto_visualize_triplets", True):
        logger.info("")
        logger.info("=" * 60)
        logger.info("ADDING EMBEDDING VISUALIZATIONS TO TRIPLETS")
        logger.info("=" * 60)

        import json

        import numpy as np

        from multiview.visualization.reducers import UMAPReducer
        from multiview.visualization.triplet_viewer import generate_triplet_viewer

        reducer_type = cfg.get("triplet_viz_reducer", "umap")
        # Use a method that generates embeddings (not BM25 or LM judge)
        embedding_method = cfg.get(
            "triplet_viz_embedding_method", "qwen3_8b_no_instructions"
        )

        for task in tasks:
            task_name = task.get_task_name()
            triplets_path = triplets_dir / task_name / "triplets.json"
            embeddings_path = embeddings_dir / task_name / f"{embedding_method}.npy"
            doc_ids_path = (
                embeddings_dir / task_name / f"{embedding_method}_doc_ids.npy"
            )

            if not triplets_path.exists():
                logger.warning(f"Triplets file not found: {triplets_path}")
                continue

            if not embeddings_path.exists():
                logger.warning(
                    f"Embeddings not found for {task_name}/{embedding_method}, skipping triplet viz"
                )
                continue

            try:
                logger.info(f"Processing {task_name}...")
                logger.info(f"  Using embeddings from: {embedding_method}")
                logger.info(f"  Reducer: {reducer_type}")

                # Load embeddings and doc IDs
                embeddings = np.load(embeddings_path)
                doc_ids = np.load(doc_ids_path)
                logger.info(f"  Loaded {len(embeddings)} embeddings")

                # Reduce to 2D
                if reducer_type == "umap":
                    from multiview.visualization.reducers import UMAPReducer

                    reducer = UMAPReducer(n_neighbors=min(15, len(embeddings) - 1))
                elif reducer_type == "tsne":
                    from multiview.visualization.reducers import TSNEReducer

                    reducer = TSNEReducer(perplexity=min(30, len(embeddings) / 3))
                elif reducer_type == "pca":
                    from multiview.visualization.reducers import PCAReducer

                    reducer = PCAReducer()
                else:
                    raise ValueError(f"Unknown reducer: {reducer_type}")

                coords_2d = reducer.fit_transform(embeddings)
                logger.info(f"  Reduced to 2D: {coords_2d.shape}")

                # Load triplets and add coordinates
                with open(triplets_path) as f:
                    triplets = json.load(f)

                # Create doc_id -> coordinates mapping
                doc_id_to_coords = {
                    int(doc_id): coords_2d[i] for i, doc_id in enumerate(doc_ids)
                }

                # Add embedding_viz to each document in triplets
                for triplet in triplets:
                    for role in ["anchor", "positive", "negative"]:
                        doc_id = triplet[f"{role}_id"]
                        if doc_id not in doc_id_to_coords:
                            logger.warning(f"Doc {doc_id} not found in embeddings")
                            continue

                        x, y = doc_id_to_coords[doc_id]

                        # Add embedding_viz to document
                        if isinstance(triplet[role], dict):
                            triplet[role]["embedding_viz"] = {
                                "x": float(x),
                                "y": float(y),
                                "reducer": reducer_type,
                                "embedding_method": embedding_method,
                            }
                        else:
                            # Convert string to dict
                            triplet[role] = {
                                "text": triplet[role],
                                "embedding_viz": {
                                    "x": float(x),
                                    "y": float(y),
                                    "reducer": reducer_type,
                                    "embedding_method": embedding_method,
                                },
                            }

                # Save updated triplets
                with open(triplets_path, "w") as f:
                    json.dump(triplets, f, indent=2)

                logger.info("  ✓ Added embedding coordinates to triplets")

                # Regenerate viewer with embedding visualization
                viewer_path = generate_triplet_viewer(triplets_path)
                logger.info(f"  ✓ Regenerated viewer: {viewer_path}")

            except Exception as e:
                logger.warning(f"✗ Failed to add embeddings to {task_name}: {e}")
                import traceback

                traceback.print_exc()

        logger.info("")
        logger.info("=" * 60)
        logger.info("TRIPLET VISUALIZATION COMPLETE")
        logger.info("=" * 60)

    # Auto-generate visualizations if enabled
    if cfg.get("auto_visualize", True):
        logger.info("")
        logger.info("=" * 60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 60)

        viz_config = cfg.get("visualization", {})
        reducers = viz_config.get("reducers", ["tsne"])
        use_thumbnails = viz_config.get("thumbnails", True)
        output_base = viz_config.get("output_dir", "outputs/viz")

        # Collect task names from evaluated tasks to filter visualizations
        evaluated_task_names = [task.get_task_name() for task in tasks]

        # Extract method names from config to filter visualizations
        # Only visualize methods that are actually enabled in the config
        evaluated_method_names = []
        if cfg.get("methods_to_evaluate"):
            for _method_type, method_list in cfg.methods_to_evaluate.items():
                if method_list:
                    for method_config in method_list:
                        method_name = method_config.get("name")
                        if method_name:
                            # Handle trial-based methods (num_trials > 1)
                            num_trials = method_config.get("num_trials", 1)
                            if num_trials > 1:
                                # Add all trial variants: method_trial1, method_trial2, etc.
                                for trial_num in range(1, num_trials + 1):
                                    evaluated_method_names.append(
                                        f"{method_name}_trial{trial_num}"
                                    )
                            else:
                                evaluated_method_names.append(method_name)

        if evaluated_method_names:
            logger.info(
                f"Filtering visualizations to {len(evaluated_method_names)} enabled methods:"
            )
            for method_name in sorted(evaluated_method_names):
                logger.info(f"  - {method_name}")
        else:
            logger.info(
                "No method filter applied - will visualize all methods with embeddings"
            )

        try:
            success_count, fail_count = generate_visualizations_for_benchmark(
                benchmark_run=cfg.run_name,
                reducers=reducers,
                output_base=output_base,
                output_benchmark_run=f"{cfg.run_name}/triples",
                use_thumbnails=use_thumbnails,
                task_filter=evaluated_task_names,
                method_filter=evaluated_method_names
                if evaluated_method_names
                else None,
                quiet=False,
            )

            if fail_count > 0:
                logger.warning(
                    f"Some visualizations failed ({fail_count}/{success_count + fail_count})"
                )
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            logger.error(
                "You can try running the evaluation again, or check the visualization settings in your config."
            )


if __name__ == "__main__":
    main()
