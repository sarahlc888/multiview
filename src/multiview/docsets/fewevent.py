"""FewEventClustering dataset loader from InBedder.

Loads biographical text with event type clusters from Hugging Face.
Dataset from: https://huggingface.co/datasets/BrandonZYW/FewEventClustering
Paper: "Answer is All You Need" (https://arxiv.org/abs/2402.09642)
"""

from __future__ import annotations

from multiview.docsets.criteria_metadata import FEWEVENT_CRITERIA
from multiview.docsets.inbedder_clustering import InBedderClusteringDocSet


class FewEventClusteringDocSet(InBedderClusteringDocSet):
    """FewEventClustering dataset from InBedder.

    Dataset contains 4,742 biographical texts with event type clusters.
    Texts are clustered by the type of life event or biographical category:
    - Education
    - Career
    - Personal Life
    - Achievements
    - Conflicts
    - Birth/Death
    - And more event types

    Each row contains: text, cluster, split
    Uses prelabeled format for evaluation.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")

    Usage:
        tasks:
          - document_set: fewevent
            criterion: cluster
            triplet_style: prelabeled
            config:
              max_docs: 500
    """

    DATASET_PATH = "BrandonZYW/FewEventClustering"
    DESCRIPTION = "Biographical text with event type clusters from InBedder"
    SUBSETS = None  # No subsets for this dataset

    # Cluster criterion is known (pre-labeled)
    KNOWN_CRITERIA = ["cluster"]

    # Metadata for LM-based criteria (descriptions, hints, etc.)
    CRITERION_METADATA = FEWEVENT_CRITERIA
