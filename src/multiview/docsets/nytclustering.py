"""NYTClustering dataset loader from InBedder.

Loads New York Times articles with topic and location clusters from Hugging Face.
Dataset from: https://huggingface.co/datasets/BrandonZYW/NYTClustering
Paper: "Answer is All You Need" (https://arxiv.org/abs/2402.09642)
"""

from __future__ import annotations

from multiview.docsets.inbedder_clustering import InBedderClusteringDocSet


class NYTClusteringDocSet(InBedderClusteringDocSet):
    """NYTClustering dataset from InBedder.

    Dataset contains 4,320 New York Times articles (2.16k per configuration).
    Two configurations with different clustering criteria:
    - topic: Topic-based clustering (10 categories)
    - location: Location-based clustering (10 categories)

    Each row contains: text, cluster, split
    Uses prelabeled format for evaluation.

    Config parameters:
        subset (str): "topic" or "location" (required)
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")

    Usage:
        tasks:
          - document_set: nytclustering
            criterion: topic
            triplet_style: prelabeled
            config:
              subset: topic
            max_docs: 500
    """

    DATASET_PATH = "BrandonZYW/NYTClustering"
    DESCRIPTION = "NYT articles with topic and location clusters from InBedder"
    SUBSETS = ["topic", "location"]

    # Both criteria are known (pre-labeled)
    KNOWN_CRITERIA = ["topic", "location"]

    # Metadata for LM-based criteria (descriptions, hints, etc.)
    DATASET_NAME = "inb_nytclustering"
