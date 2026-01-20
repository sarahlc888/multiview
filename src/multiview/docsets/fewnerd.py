"""FewNerdClustering dataset loader from InBedder.

Loads sentences with named entity type clusters from Hugging Face.
Dataset from: https://huggingface.co/datasets/BrandonZYW/FewNerdClustering
Paper: "Answer is All You Need" (https://arxiv.org/abs/2402.09642)
"""

from __future__ import annotations

from multiview.docsets.criteria_metadata import FEWNERD_CRITERIA
from multiview.docsets.inbedder_clustering import InBedderClusteringDocSet


class FewNerdClusteringDocSet(InBedderClusteringDocSet):
    """FewNerdClustering dataset from InBedder.

    Dataset contains 3,789 sentences with named entity type clusters.
    Sentences are clustered by the type of named entity featured:
    - Person
    - Organization
    - Location
    - Building/Facility
    - Geo-Political Entity
    - Art/Creative Work
    - Product
    - Event
    - And more entity types

    Each row contains: text, cluster, split
    Uses prelabeled format for evaluation.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")

    Usage:
        tasks:
          - document_set: fewnerd
            criterion: cluster
            triplet_style: prelabeled
            config:
              max_docs: 500
    """

    DATASET_PATH = "BrandonZYW/FewNerdClustering"
    DESCRIPTION = "Sentences with named entity type clusters from InBedder"
    SUBSETS = None  # No subsets for this dataset

    # Cluster criterion is known (pre-labeled)
    KNOWN_CRITERIA = ["cluster"]

    # Metadata for LM-based criteria (descriptions, hints, etc.)
    CRITERION_METADATA = FEWNERD_CRITERIA
