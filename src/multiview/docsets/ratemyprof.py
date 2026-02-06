"""RateMyProfClustering dataset loader from InBedder.

Loads professor reviews with thematic clusters from Hugging Face.
Dataset from: https://huggingface.co/datasets/BrandonZYW/RateMyProfClustering
Paper: "Answer is All You Need" (https://arxiv.org/abs/2402.09642)
"""

from __future__ import annotations

from multiview.docsets.inbedder_clustering import InBedderClusteringDocSet


class RateMyProfClusteringDocSet(InBedderClusteringDocSet):
    """RateMyProfClustering dataset from InBedder.

    Dataset contains 2,296 professor reviews with thematic clusters.
    Reviews are clustered by what aspect they highlight:
    - Demeanor/attitude/personal qualities
    - Teaching style/methods
    - Course difficulty/workload
    - Grading practices/fairness
    - Clarity/communication

    Each row contains: text, cluster, split
    Uses prelabeled format for evaluation.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")

    Usage:
        tasks:
          - document_set: ratemyprof
            criterion: cluster
            triplet_style: prelabeled
            config:
              max_docs: 500
    """

    DATASET_PATH = "BrandonZYW/RateMyProfClustering"
    DESCRIPTION = "Professor reviews with thematic clusters from InBedder"
    SUBSETS = None  # No subsets for this dataset

    # Cluster criterion is known (pre-labeled)
    KNOWN_CRITERIA = ["cluster"]

    # Metadata for LM-based criteria (descriptions, hints, etc.)
    DATASET_NAME = "inb_ratemyprof"
