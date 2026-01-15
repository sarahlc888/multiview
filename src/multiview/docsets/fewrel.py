"""FewRelClustering dataset loader from InBedder.

Loads sentences with relation type clusters from Hugging Face.
Dataset from: https://huggingface.co/datasets/BrandonZYW/FewRelClustering
Paper: "Answer is All You Need" (https://arxiv.org/abs/2402.09642)
"""

from __future__ import annotations

from multiview.docsets.criteria_metadata import FEWREL_CRITERIA
from multiview.docsets.inbedder_clustering import InBedderClusteringDocSet


class FewRelClusteringDocSet(InBedderClusteringDocSet):
    """FewRelClustering dataset from InBedder.

    Dataset contains 4,480 sentences with relation type clusters.
    Sentences are clustered by the type of relationship expressed between entities:
    - Field of work
    - Record label
    - Place of birth
    - Occupation
    - Nationality
    - Location
    - Affiliation
    - And more relation types

    Each row contains: text, cluster, split
    Uses prelabeled format for evaluation.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")

    Usage:
        tasks:
          - document_set: fewrel
            criterion: cluster
            triplet_style: prelabeled
            config:
              max_docs: 500
    """

    DATASET_PATH = "BrandonZYW/FewRelClustering"
    DESCRIPTION = "Sentences with relation type clusters from InBedder"
    SUBSETS = None  # No subsets for this dataset

    # Cluster criterion is known (pre-labeled)
    KNOWN_CRITERIA = ["cluster"]

    # Metadata for LM-based criteria (descriptions, hints, etc.)
    CRITERION_METADATA = FEWREL_CRITERIA
