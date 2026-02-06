#!/usr/bin/env python3
"""Add embedding visualization to existing triplet JSON files.

This script augments triplets.json files with 2D embedding coordinates
for visualization in the HTML triplet viewer.

Usage:
    # Add to a single triplets.json file
    python scripts/add_triplet_embeddings.py \\
        outputs/benchmark/triplets/haiku__poem_composition__tag__50/triplets.json

    # Use a different embedding preset
    python scripts/add_triplet_embeddings.py \\
        outputs/benchmark/triplets/gsm8k__problem_type__tag__5/triplets.json \\
        --embedding-preset hf_qwen3_embedding_8b

    # Use t-SNE instead of UMAP
    python scripts/add_triplet_embeddings.py \\
        outputs/benchmark/triplets/haiku__imagery__tag__50/triplets.json \\
        --reducer tsne

    # Process all triplet files in a directory
    python scripts/add_triplet_embeddings.py \\
        "outputs/benchmark/triplets/*/triplets.json" \\
        --glob
"""

import argparse
import glob
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multiview.utils.logging_utils import setup_logging  # noqa: E402
from multiview.visualization.triplet_embedding_viz import (  # noqa: E402
    add_embedding_viz_to_triplets,
)


def main():
    parser = argparse.ArgumentParser(
        description="Add embedding visualization to triplet JSON files"
    )
    parser.add_argument(
        "triplets_path",
        help="Path to triplets.json file (or glob pattern if --glob is used)",
    )
    parser.add_argument(
        "--embedding-preset",
        default="openai_embedding_small",
        help="Embedding preset to use (default: openai_embedding_small)",
    )
    parser.add_argument(
        "--reducer",
        default="umap",
        choices=["umap", "tsne", "pca"],
        help="Dimensionality reduction method (default: umap)",
    )
    parser.add_argument(
        "--cache-alias",
        default="triplet_embedding_viz",
        help="Cache alias for embeddings (default: triplet_embedding_viz)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embedding coordinates",
    )
    parser.add_argument(
        "--glob",
        action="store_true",
        help="Treat triplets_path as a glob pattern",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO" if args.verbose else "WARNING")

    # Get list of files to process
    if args.glob:
        files = glob.glob(args.triplets_path)
        if not files:
            print(f"No files found matching pattern: {args.triplets_path}")
            sys.exit(1)
        print(f"Found {len(files)} triplet files to process")
    else:
        files = [args.triplets_path]

    # Process each file
    success_count = 0
    error_count = 0

    for file_path in files:
        try:
            print(f"\nProcessing: {file_path}")
            add_embedding_viz_to_triplets(
                file_path,
                embedding_preset=args.embedding_preset,
                reducer=args.reducer,
                cache_alias=args.cache_alias,
                force=args.force,
            )
            success_count += 1

            # Regenerate viewer
            from multiview.visualization.triplet_viewer import generate_triplet_viewer

            viewer_path = Path(file_path).parent / "viewer.html"
            if viewer_path.exists():
                viewer_path.unlink()
            generate_triplet_viewer(file_path)
            print(f"✓ Regenerated viewer: {viewer_path}")

        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
            error_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}")

    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
