"""FeedbacksClustering dataset loader from InBedder.

Loads summary feedback with feedback type clusters from Hugging Face.
Dataset from: https://huggingface.co/datasets/BrandonZYW/FeedbacksClustering
Paper: "Answer is All You Need" (https://arxiv.org/abs/2402.09642)
"""

from __future__ import annotations

from multiview.docsets.inbedder_clustering import InBedderClusteringDocSet


class FeedbacksClusteringDocSet(InBedderClusteringDocSet):
    """FeedbacksClustering dataset from InBedder.

    Dataset contains 756 feedback comments on summaries with feedback type clusters.
    Feedback is clustered by the type of quality dimension addressed:
    - Inclusion of main points and necessary details
    - Accuracy and correctness of information
    - Coherence and logical flow of ideas

    Each row contains: text, cluster, split
    Uses prelabeled format for evaluation.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")

    Usage:
        tasks:
          - document_set: feedbacks
            criterion: cluster
            triplet_style: prelabeled
            config:
              max_docs: 500
    """

    DATASET_PATH = "BrandonZYW/FeedbacksClustering"
    DESCRIPTION = "Summary feedback with feedback type clusters from InBedder"
    SUBSETS = None  # No subsets for this dataset

    # Cluster criterion is known (pre-labeled)
    KNOWN_CRITERIA = ["cluster"]

    # Metadata for LM-based criteria (descriptions, hints, etc.)
    DATASET_NAME = "inb_feedbacks"
