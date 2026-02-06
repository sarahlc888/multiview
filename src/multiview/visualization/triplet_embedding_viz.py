"""Add embedding visualization data to triplet JSON files.

This module provides utilities to augment triplet JSON files with 2D embedding
coordinates for visualization in the HTML triplet viewer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from multiview.inference import run_inference

logger = logging.getLogger(__name__)


def add_embedding_viz_to_triplets(
    triplets_json_path: str | Path,
    embedding_preset: str = "openai_embedding_small",
    reducer: str = "umap",
    cache_alias: str | None = None,
    force: bool = False,
) -> Path:
    """Add 2D embedding coordinates to a triplets.json file.

    This function:
    1. Loads the triplets.json file
    2. Extracts all documents
    3. Computes embeddings for all documents
    4. Reduces embeddings to 2D using UMAP (or other reducer)
    5. Adds coordinates to each document in the JSON
    6. Saves the updated JSON

    Args:
        triplets_json_path: Path to triplets.json file
        embedding_preset: Inference preset to use for embeddings
        reducer: Dimensionality reduction method ('umap', 'tsne', 'pca')
        cache_alias: Optional cache identifier for embeddings
        force: If True, overwrite existing coordinates

    Returns:
        Path to the updated triplets.json file
    """
    triplets_path = Path(triplets_json_path)

    if not triplets_path.exists():
        raise FileNotFoundError(f"Triplets file not found: {triplets_path}")

    logger.info(f"Adding embedding visualization to: {triplets_path}")

    # Load triplets
    with open(triplets_path) as f:
        triplets = json.load(f)

    if not triplets:
        logger.warning("No triplets found in file")
        return triplets_path

    # Check if coordinates already exist
    if not force and "embedding_viz" in triplets[0].get("anchor", {}):
        logger.info("Embedding coordinates already exist. Use force=True to overwrite.")
        return triplets_path

    # Extract documents and create document list
    documents = []
    doc_id_to_idx = {}

    for triplet in triplets:
        for role in ["anchor", "positive", "negative"]:
            doc_id = triplet[f"{role}_id"]
            if doc_id not in doc_id_to_idx:
                doc_id_to_idx[doc_id] = len(documents)
                # Get document content
                doc_content = triplet[role]
                if isinstance(doc_content, dict):
                    doc_text = doc_content.get("text", str(doc_content))

                    # Handle image documents with generic "<image>" text
                    # Add doc_id to make each document unique for embedding
                    if doc_text == "<image>" or doc_text.strip() == "":
                        # Build text from metadata fields
                        metadata_fields = [
                            k
                            for k in doc_content.keys()
                            if k
                            not in ["text", "image_path", "embedding_viz", "_metadata"]
                        ]
                        if metadata_fields:
                            metadata_text = " ".join(
                                f"{k}: {doc_content[k]}"
                                for k in sorted(metadata_fields)
                                if doc_content.get(k)
                            )
                            doc_text = (
                                f"Document {doc_id}: {metadata_text}"
                                if metadata_text
                                else f"Document {doc_id}"
                            )
                        else:
                            doc_text = f"Document {doc_id}"
                else:
                    doc_text = str(doc_content)
                documents.append(doc_text)

    logger.info(f"Computing embeddings for {len(documents)} unique documents")

    # Compute embeddings
    inputs = {"document": documents}
    embeddings = run_inference(
        inputs=inputs,
        config=embedding_preset,
        cache_alias=cache_alias,
        verbose=False,
    )

    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    logger.info(f"Embeddings shape: {embeddings_array.shape}")

    # Reduce to 2D
    logger.info(f"Reducing to 2D using {reducer}")
    if reducer == "umap":
        from multiview.visualization.reducers import UMAPReducer

        reducer_obj = UMAPReducer(n_neighbors=min(15, len(documents) - 1))
    elif reducer == "tsne":
        from multiview.visualization.reducers import TSNEReducer

        reducer_obj = TSNEReducer(perplexity=min(30, len(documents) / 3))
    elif reducer == "pca":
        from multiview.visualization.reducers import PCAReducer

        reducer_obj = PCAReducer()
    else:
        raise ValueError(f"Unknown reducer: {reducer}")

    coords_2d = reducer_obj.fit_transform(embeddings_array)
    logger.info(f"2D coordinates shape: {coords_2d.shape}")

    # Add coordinates to triplets
    for triplet in triplets:
        for role in ["anchor", "positive", "negative"]:
            doc_id = triplet[f"{role}_id"]
            doc_idx = doc_id_to_idx[doc_id]
            x, y = coords_2d[doc_idx]

            # Add embedding_viz to document
            if isinstance(triplet[role], dict):
                triplet[role]["embedding_viz"] = {
                    "x": float(x),
                    "y": float(y),
                    "reducer": reducer,
                    "embedding_preset": embedding_preset,
                }
            else:
                # If document is a string, we need to convert to dict first
                triplet[role] = {
                    "text": triplet[role],
                    "embedding_viz": {
                        "x": float(x),
                        "y": float(y),
                        "reducer": reducer,
                        "embedding_preset": embedding_preset,
                    },
                }

    # Save updated triplets
    with open(triplets_path, "w") as f:
        json.dump(triplets, f, indent=2)

    logger.info(f"✓ Added embedding visualization data to {triplets_path}")
    logger.info(f"  Reducer: {reducer}")
    logger.info(f"  Embedding preset: {embedding_preset}")
    logger.info(f"  Documents: {len(documents)}")

    return triplets_path
