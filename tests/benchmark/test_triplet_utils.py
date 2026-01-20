"""Tests for triplet utilities."""

import pytest

from multiview.benchmark.task import Task
from multiview.benchmark.triplets.utils import build_triplet_dicts


@pytest.mark.parametrize("dataset", ["abstractsim", "arxiv_abstract_sentences", "gsm8k", "rocstories", "crossword_clues", "hackernews", "analogies", "infinite_prompts", "infinite_chats", "dickinson", "moralfables", "onion_headlines", "goodreads_quotes", "arxiv_cs", "ut_zappos50k"])
def test_create_random_triplets(dataset):
    """Test random triplet creation across different datasets."""
    # Create a task with random triplets
    task = Task(
        config={
            "document_set": dataset,
            "criterion": "word_count",  # Not used for random style, but required
            "max_docs": 10,
            "max_triplets": 5,
            "triplet_style": "random",
        }
    )

    task.load_documents()
    # Skip annotation for random style
    if task.triplet_style != "random":
        task.annotate_documents()
    task.create_triplets()

    print(f"\n=== Testing {dataset} ===")
    print(f"Created {len(task.triplets)} triplets:")

    def format_doc(doc):
        """Format document for display, handling both string and dict formats."""
        if isinstance(doc, str):
            return doc
        elif isinstance(doc, dict):
            return str(doc)
        else:
            return str(doc)

    def get_text(doc):
        """Extract just the text content from document."""
        if isinstance(doc, str):
            return doc
        elif isinstance(doc, dict):
            return doc.get('text', str(doc))
        else:
            return str(doc)

    for i, triplet in enumerate(task.triplets):
        anchor_idx, positive_idx, negative_idx = triplet
        print(f"\nTriplet {i} (full):")
        print(f"  [Anchor {anchor_idx}] {format_doc(task.documents[anchor_idx])}")
        print(f"  [Positive {positive_idx}] {format_doc(task.documents[positive_idx])}")
        print(f"  [Negative {negative_idx}] {format_doc(task.documents[negative_idx])}")

        print(f"\nTriplet {i} (text only):")
        print(f"  [Anchor {anchor_idx}] {get_text(task.documents[anchor_idx])}")
        print(f"  [Positive {positive_idx}] {get_text(task.documents[positive_idx])}")
        print(f"  [Negative {negative_idx}] {get_text(task.documents[negative_idx])}")

    # Basic assertions
    assert len(task.triplets) == 5
    assert all(len(triplet) == 3 for triplet in task.triplets)
    # Check that each triplet has distinct indices
    assert all(len(set(triplet)) == 3 for triplet in task.triplets)


@pytest.mark.parametrize("dataset,criterion,docset_config", [
    ("abstractsim", "abstract_similarity", {"split": "validation"}),
    ("abstractsim", "abstraction_level", {"split": "validation"}),
    ("inb_nytclustering", "topic", {"subset": "topic"}),
    ("inb_ratemyprof", "cluster", {}),
    ("inb_feedbacks", "cluster", {}),
    ("inb_fewrel", "cluster", {}),
    ("inb_fewnerd", "cluster", {}),
    ("inb_fewevent", "cluster", {}),
    ("inb_intent_emotion", "intent_similarity", {"subset": "intent"}),
    ("inspired", "movie_recommendation", {"split": "train"}),
    ("bills", "topic", {"text_field": "summary"}),
    ("bills", "subtopic", {"text_field": "summary"}),
    ("bills", "topic", {"text_field": "tokenized_text"}),
    ("trex", "relation", {"split": "validation", "min_relation_freq": 3}),
    ("infinite_prompts", "categories", {}),
    ("triz40", "triz_principle", {}),
])
def test_create_prelabeled_triplets(dataset, criterion, docset_config):
    """Test prelabeled triplet creation across different datasets with known labels."""
    # All datasets now use the same prelabeled style
    # The task implementation will detect the dataset type and use appropriate logic
    triplet_style = "prelabeled"

    config = {
        "document_set": dataset,
        "criterion": criterion,
        "max_docs": 100,
        "max_triplets": 10,
        "triplet_style": triplet_style,
        "selection_strategy": "hard_negatives",
        "seed": 42,
        "config": docset_config,  # Pass docset-specific config
    }

    # Create a task with prelabeled triplets
    task = Task(config=config)

    task.load_documents()

    # Dataset-specific smoke checks that used to live in docset-only tests.
    # (We keep these here to reduce duplication while still validating schema + labels.)
    if dataset == "inspired":
        assert len(task.documents) > 0, "Should load at least some Inspired documents"
        for doc in task.documents:
            assert isinstance(doc, dict), "Inspired documents should be dicts"
            assert "text" in doc, "Inspired documents should have 'text' field"
            assert "movie_recommendation" in doc, "Inspired documents should have 'movie_recommendation' field"
            assert isinstance(doc["text"], str) and len(doc["text"]) > 0, "Inspired text should be non-empty"
            assert isinstance(doc["movie_recommendation"], str), "Inspired movie_recommendation should be a string"
    elif dataset == "bills":
        assert len(task.documents) > 0, "Should load at least some Bills documents"
        for doc in task.documents:
            assert isinstance(doc, dict), "Bills documents should be dicts"
            assert "text" in doc, "Bills documents should have 'text' field"
            assert "topic" in doc, "Bills documents should have 'topic' field"
            assert "subtopic" in doc, "Bills documents should have 'subtopic' field"
            assert isinstance(doc["text"], str) and len(doc["text"]) > 0, "Bills text should be non-empty"
            assert isinstance(doc["topic"], str), "Bills topic should be a string"
            assert isinstance(doc["subtopic"], str), "Bills subtopic should be a string"

    task.annotate_documents()
    task.create_triplets()

    print(f"\n=== Testing {dataset} with criterion={criterion} ===")
    print(f"Created {len(task.triplets)} triplets from {len(task.documents)} documents")

    # Should create triplets
    assert len(task.triplets) > 0, f"Should create at least some triplets for {dataset}"
    assert len(task.triplets) <= 10, f"Should respect max_triplets limit for {dataset}"

    # Verify triplet structure - different logic for KGC and IntentEmotion
    if dataset == "inb_intent_emotion":
        # IntentEmotion uses pre-made triplets - just verify they're valid
        for i, triplet in enumerate(task.triplets[:3]):  # Show first 3
            anchor_idx, positive_idx, negative_idx = triplet

            # Documents are now strings, not dicts
            anchor_text = task.document_set.get_document_text(task.documents[anchor_idx])
            positive_text = task.document_set.get_document_text(task.documents[positive_idx])
            negative_text = task.document_set.get_document_text(task.documents[negative_idx])

            print(f"\nTriplet {i}:")
            print(f"  Anchor: {anchor_text[:60]}...")
            print(f"  Positive: {positive_text[:60]}...")
            print(f"  Negative: {negative_text[:60]}...")

            # Verify indices are distinct
            assert anchor_idx != positive_idx, f"{dataset}: Anchor and positive should be different"
            assert anchor_idx != negative_idx, f"{dataset}: Anchor and negative should be different"
            assert positive_idx != negative_idx, f"{dataset}: Positive and negative should be different"

    elif dataset == "trex":
        # KGC-specific validation - documents are now strings, metadata in _triplet_metadata
        metadata_lookup = task.document_set._triplet_metadata

        for i, triplet in enumerate(task.triplets[:3]):  # Show first 3
            anchor_idx, positive_idx, negative_idx = triplet

            anchor_text = task.document_set.get_document_text(task.documents[anchor_idx])
            positive_text = task.document_set.get_document_text(task.documents[positive_idx])
            negative_text = task.document_set.get_document_text(task.documents[negative_idx])

            anchor_meta = metadata_lookup[anchor_text]
            positive_meta = metadata_lookup[positive_text]
            negative_meta = metadata_lookup[negative_text]

            print(f"\nTriplet {i}:")
            print(f"  Anchor: {anchor_text} (entity_type={anchor_meta['entity_type']}, relation={anchor_meta['relation']})")
            print(f"  Positive: {positive_text} (entity_type={positive_meta['entity_type']}, relation={positive_meta['relation']})")
            print(f"  Negative: {negative_text} (entity_type={negative_meta['entity_type']}, relation={negative_meta['relation']})")

            # Verify entity types
            assert anchor_meta["entity_type"] == "head", f"{dataset}: Anchor should be HEAD entity"
            assert positive_meta["entity_type"] == "tail", f"{dataset}: Positive should be TAIL entity"
            assert negative_meta["entity_type"] == "tail", f"{dataset}: Negative should be TAIL entity"

            # Verify all share same relation
            assert anchor_meta["relation"] == positive_meta["relation"], f"{dataset}: Anchor and positive should share relation"
            assert anchor_meta["relation"] == negative_meta["relation"], f"{dataset}: Anchor and negative should share relation"

            # Verify positive is correct for anchor
            assert positive_meta["entity_id"] == anchor_meta["correct_tail_id"], f"{dataset}: Positive should match anchor's correct tail"

            # Verify negative is incorrect for anchor
            assert negative_meta["entity_id"] != anchor_meta["correct_tail_id"], f"{dataset}: Negative should not match anchor's correct tail"

            # All indices should be distinct
            assert anchor_idx != positive_idx, f"{dataset}: Anchor and positive should be different documents"
            assert anchor_idx != negative_idx, f"{dataset}: Anchor and negative should be different documents"
            assert positive_idx != negative_idx, f"{dataset}: Positive and negative should be different documents"
    else:
        # If a docset declares precomputed annotations for the criterion, verify those
        # annotations line up with the loaded documents (text -> prelabel).
        if task.document_set.has_precomputed_annotations(criterion):
            precomputed = task.document_set.get_precomputed_annotations(criterion)
            assert len(precomputed) == len(task.documents), (
                f"{dataset}: Expected one precomputed annotation per document "
                f"for criterion={criterion}"
            )
            for doc in task.documents:
                # Handle both string documents (like infinite_prompts) and dict documents
                doc_text = doc if isinstance(doc, str) else doc["text"]
                doc_criterion = None if isinstance(doc, str) else doc.get(criterion)

                assert doc_text in precomputed, (
                    f"{dataset}: Document text missing from precomputed annotations: "
                    f"{doc_text[:80]}"
                )

                # For string documents, annotations are stored separately, not in the doc itself
                if isinstance(doc, dict):
                    assert precomputed[doc_text]["prelabel"] == doc_criterion, (
                        f"{dataset}: Precomputed prelabel should match doc[{criterion}]"
                    )

        # Standard prelabeled validation
        def get_label(doc, crit):
            """Get criterion label from document."""
            if isinstance(doc, dict):
                return doc.get(crit, "unknown")
            # For string documents, look up label from precomputed annotations
            elif isinstance(doc, str) and task.document_set.has_precomputed_annotations(crit):
                precomputed = task.document_set.get_precomputed_annotations(crit)
                return precomputed.get(doc, {}).get("prelabel", "unknown")
            return "string_doc"

        def get_label_set(label):
            """Parse label into set of values (handles comma-separated multi-labels)."""
            if isinstance(label, str) and ", " in label:
                # Handle comma-separated multi-label strings
                return {v.strip() for v in label.split(",")}
            elif isinstance(label, list):
                return set(label)
            else:
                return {label}

        for i, triplet in enumerate(task.triplets[:3]):  # Show first 3
            anchor_idx, positive_idx, negative_idx = triplet

            # Get labels
            anchor_label = get_label(task.documents[anchor_idx], criterion)
            pos_label = get_label(task.documents[positive_idx], criterion)
            neg_label = get_label(task.documents[negative_idx], criterion)

            print(f"\nTriplet {i} labels:")
            print(f"  Anchor: {anchor_label}")
            print(f"  Positive: {pos_label}")
            print(f"  Negative: {neg_label}")
            print(f"  Anchor: {task.documents[anchor_idx]['text']=}")
            print(f"  Positive: {task.documents[positive_idx]['text']=}")
            print(f"  Negative: {task.documents[negative_idx]['text']=}")

            # Parse labels into sets for comparison (handles multi-label datasets)
            anchor_labels = get_label_set(anchor_label)
            pos_labels = get_label_set(pos_label)
            neg_labels = get_label_set(neg_label)

            # Verify: anchor and positive should share at least one label
            assert len(anchor_labels & pos_labels) > 0, \
                f"{dataset}: Anchor and positive should share at least one label. Got {anchor_label} and {pos_label}"
            # Negative should share no labels with anchor
            assert len(anchor_labels & neg_labels) == 0, \
                f"{dataset}: Anchor and negative should have no shared labels. Got {anchor_label} and {neg_label}"

            # All indices should be distinct
            assert anchor_idx != positive_idx, f"{dataset}: Anchor and positive should be different documents"
            assert anchor_idx != negative_idx, f"{dataset}: Anchor and negative should be different documents"
            assert positive_idx != negative_idx, f"{dataset}: Positive and negative should be different documents"

    print(f"\n✓ All {len(task.triplets)} triplets validated for {dataset}")


def test_inspired_load_test_split_smoke():
    """Smoke test: Inspired can load test split (may be too small for triplets)."""
    task = Task(
        config={
            "document_set": "inspired",
            "criterion": "movie_recommendation",
            "max_docs": 10,
            "triplet_style": "prelabeled",
            "split": "test",
        }
    )
    task.load_documents()

    assert len(task.documents) > 0, "Should load Inspired test split documents"
    assert len(task.documents) <= 10, "Should respect max_docs limit for test split"


@pytest.mark.parametrize("authors,description", [
    (["Schopenhauer", "Pascal"], "Pessimism vs Faith"),
    (["C.S. Lewis", "Hermann Hesse"], "Christian vs Eastern"),
    (["Tolstoy"], "Russian Moralist"),
    (["Arthur Schopenhauer", "Blaise Pascal", "Leo Tolstoy"], "Three Philosophers"),
])
def test_goodreads_author_filtering(authors, description):
    """Test Goodreads quotes with specific author filtering.

    This test demonstrates the positive_sum criterion with curated author lists.
    Different author combinations create different philosophical dialogues.
    """
    # Create a task with author filtering
    task = Task(
        config={
            "document_set": "goodreads_quotes",
            "criterion": "positive_sum",
            "max_docs": 60,
            "max_triplets": 3,
            "triplet_style": "random",
            "config": {
                "authors": authors,
                "min_likes": 5,
            }
        }
    )

    print(f"\n{'=' * 80}")
    print(f"TESTING: {description}")
    print(f"Authors: {', '.join(authors)}")
    print('=' * 80)

    task.load_documents()
    print(f"\nLoaded {len(task.documents)} quotes")

    # Count quotes per author
    author_counts = {}
    for doc in task.documents:
        author = doc['author']
        author_counts[author] = author_counts.get(author, 0) + 1

    print("Author distribution:")
    for author, count in sorted(author_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {author}: {count} quotes")

    # Verify we got quotes
    assert len(task.documents) > 0, f"Should load quotes for {authors}"

    # Verify we only got requested authors
    for doc in task.documents:
        doc_author = doc['author'].lower()
        matches = any(filter_author.lower() in doc_author for filter_author in authors)
        assert matches, f"Got unexpected author: {doc['author']} (not in {authors})"

    # Create triplets
    task.create_triplets()
    print(f"\nCreated {len(task.triplets)} triplets")

    # Show example triplets
    for i, (anchor_idx, pos_idx, neg_idx) in enumerate(task.triplets, 1):
        print(f"\n{'-' * 80}")
        print(f"Triplet {i}:")
        print(f"{'-' * 80}")

        anchor = task.documents[anchor_idx]
        positive = task.documents[pos_idx]
        negative = task.documents[neg_idx]

        # Print anchor
        print(f"\nANCHOR ({anchor['author']}):")
        print(f'  "{anchor["text"][:100]}..."' if len(anchor["text"]) > 100 else f'  "{anchor["text"]}"')
        print(f"  Likes: {anchor['likes']}")

        # Print positive
        print(f"\nPOSITIVE ({positive['author']}):")
        print(f'  "{positive["text"][:100]}..."' if len(positive["text"]) > 100 else f'  "{positive["text"]}"')
        print(f"  Likes: {positive['likes']}")

        # Print negative
        print(f"\nNEGATIVE ({negative['author']}):")
        print(f'  "{negative["text"][:100]}..."' if len(negative["text"]) > 100 else f'  "{negative["text"]}"')
        print(f"  Likes: {negative['likes']}")

        # Analyze cross-author pairings
        if len(authors) > 1:
            anchor_matches = [a for a in authors if a.lower() in anchor['author'].lower()]
            pos_matches = [a for a in authors if a.lower() in positive['author'].lower()]

            if anchor_matches and pos_matches and anchor_matches[0] != pos_matches[0]:
                print(f"\n  → CROSS-AUTHOR: {anchor_matches[0]} ↔ {pos_matches[0]}")
                print(f"    Could these create an interesting dialogue?")
            else:
                print(f"\n  → SAME-AUTHOR PAIRING")
                print(f"    Do these quotes complement each other?")

    # Basic assertions
    assert len(task.triplets) > 0
    assert all(len(triplet) == 3 for triplet in task.triplets)

    print(f"\n{'=' * 80}")
    print(f"✓ Successfully tested {description}")
    print('=' * 80)


def test_bm25_heuristic_category_triplets():
    """Test BM25 heuristic for category-based triplet selection.

    This test verifies that:
    1. BM25 heuristic mode skips LM judge calls
    2. Positives share the same category as anchor
    3. Negatives have different categories from anchor
    4. All triplet indices are distinct
    """
    from multiview.benchmark.triplets.triplet_utils import create_lm_triplets_category

    # Create mock documents with clear categories
    documents = [
        "The cat sleeps on the mat",  # category: animals
        "Dogs are loyal companions",  # category: animals
        "Puppies play in the yard",   # category: animals
        "The Python programming language is powerful",  # category: tech
        "JavaScript frameworks are popular",  # category: tech
        "Machine learning models need data",  # category: tech
        "Pizza is a delicious food",  # category: food
        "Pasta recipes are diverse",  # category: food
    ]

    # Create mock annotations with categories
    annotations = [
        {"category": "animals"},
        {"category": "animals"},
        {"category": "animals"},
        {"category": "tech"},
        {"category": "tech"},
        {"category": "tech"},
        {"category": "food"},
        {"category": "food"},
    ]

    # Test BM25 heuristic mode
    triplets = create_lm_triplets_category(
        documents=documents,
        annotations=annotations,
        max_triplets=5,
        use_bm25_heuristic=True,  # Enable BM25 heuristic
        criterion="category",
        criterion_description="topic category",
    )

    print(f"\n=== BM25 Heuristic Test ===")
    print(f"Created {len(triplets)} triplets using BM25 heuristic")

    # Should create some triplets
    assert len(triplets) > 0, "Should create at least some triplets"
    assert len(triplets) <= 5, "Should respect max_triplets limit"

    # Verify each triplet
    for i, (anchor_idx, pos_idx, neg_idx) in enumerate(triplets):
        anchor_cat = annotations[anchor_idx]["category"]
        pos_cat = annotations[pos_idx]["category"]
        neg_cat = annotations[neg_idx]["category"]

        print(f"\nTriplet {i}:")
        print(f"  Anchor [{anchor_idx}] ({anchor_cat}): {documents[anchor_idx]}")
        print(f"  Positive [{pos_idx}] ({pos_cat}): {documents[pos_idx]}")
        print(f"  Negative [{neg_idx}] ({neg_cat}): {documents[neg_idx]}")

        # Verify category constraints
        assert anchor_cat == pos_cat, \
            f"Anchor and positive should share category: {anchor_cat} vs {pos_cat}"
        assert anchor_cat != neg_cat, \
            f"Anchor and negative should have different categories: {anchor_cat} vs {neg_cat}"

        # Verify all indices are distinct
        assert anchor_idx != pos_idx, "Anchor and positive should be different"
        assert anchor_idx != neg_idx, "Anchor and negative should be different"
        assert pos_idx != neg_idx, "Positive and negative should be different"

    print(f"\n✓ All {len(triplets)} BM25 heuristic triplets validated")


def test_document_deduplication():
    """Test that duplicate documents are removed during loading."""
    # Create a task with a dataset that might have duplicates
    task = Task(
        config={
            "document_set": "arxiv_cs",
            "criterion": "core_contribution",
            "max_docs": 50,
            "triplet_style": "random",
        }
    )

    task.load_documents()

    # Extract text from all documents
    doc_texts = []
    for doc in task.documents:
        if isinstance(doc, dict):
            text = doc.get("text", str(doc))
        elif isinstance(doc, str):
            text = doc
        else:
            text = str(doc)
        doc_texts.append(text)

    # Verify no duplicates
    assert len(doc_texts) == len(set(doc_texts)), \
        "Documents should be deduplicated (no duplicate text content)"

    print(f"\n✓ Verified {len(doc_texts)} unique documents (no duplicates)")


def test_prelabeled_multiple_triplets_per_anchor():
    """Test that prelabeled triplet creation can create multiple triplets per anchor.

    When there are fewer anchors than requested triplets, the system should
    cycle through anchors to create multiple triplets per anchor without
    reusing the same positive/negative for a given anchor.
    """
    from multiview.benchmark.triplets.triplet_utils import create_prelabeled_triplets

    # Create 5 anchors, each with 3 positives and 3 negatives
    documents = []
    annotations = []

    for i in range(5):
        # Add anchor
        documents.append({'text': f'anchor_{i}', 'is_anchor': True})
        annotations.append({'prelabel': f'class_{i}'})

        # Add 3 positives (same class as anchor)
        for j in range(3):
            documents.append({'text': f'pos_{i}_{j}'})
            annotations.append({'prelabel': f'class_{i}'})

        # Add 3 negatives (different class)
        for j in range(3):
            documents.append({'text': f'neg_{i}_{j}'})
            annotations.append({'prelabel': f'other_class_{i}'})

    print(f"\n=== Multi-Triplet Per Anchor Test ===")
    print(f"Created {len(documents)} documents: 5 anchors + 15 positives + 15 negatives")

    # Request 15 triplets (3x more than number of anchors)
    triplets = create_prelabeled_triplets(
        documents=documents,
        annotations=annotations,
        max_triplets=15,
        selection_strategy='random',
        seed=42
    )

    print(f"Requested: 15 triplets")
    print(f"Created: {len(triplets)} triplets")

    # Should create 15 triplets (3 per anchor since each anchor has 3 pos + 3 neg)
    assert len(triplets) == 15, f"Should create 15 triplets (3 per anchor), got {len(triplets)}"

    # Count triplets per anchor
    anchor_counts = {}
    for anchor_idx, pos_idx, neg_idx in triplets:
        anchor_counts[anchor_idx] = anchor_counts.get(anchor_idx, 0) + 1

    print(f"\nTriplets per anchor:")
    for anchor_idx, count in sorted(anchor_counts.items()):
        print(f"  Anchor {anchor_idx}: {count} triplets")

    # All 5 anchors should be used
    assert len(anchor_counts) == 5, "All 5 anchors should be used"

    # Each anchor should have 3 triplets
    for anchor_idx, count in anchor_counts.items():
        assert count == 3, f"Anchor {anchor_idx} should have 3 triplets, got {count}"

    # Verify no duplicate pos/neg for same anchor
    anchor_pairs = {}
    for anchor_idx, pos_idx, neg_idx in triplets:
        if anchor_idx not in anchor_pairs:
            anchor_pairs[anchor_idx] = {'positives': set(), 'negatives': set()}

        # Check for duplicates
        assert pos_idx not in anchor_pairs[anchor_idx]['positives'], \
            f"Duplicate positive {pos_idx} for anchor {anchor_idx}"
        assert neg_idx not in anchor_pairs[anchor_idx]['negatives'], \
            f"Duplicate negative {neg_idx} for anchor {anchor_idx}"

        anchor_pairs[anchor_idx]['positives'].add(pos_idx)
        anchor_pairs[anchor_idx]['negatives'].add(neg_idx)

    print(f"\n✓ All triplets validated - no duplicate pos/neg per anchor")

    # Verify all triplets have distinct indices
    for anchor_idx, pos_idx, neg_idx in triplets:
        assert anchor_idx != pos_idx, "Anchor and positive should be different"
        assert anchor_idx != neg_idx, "Anchor and negative should be different"
        assert pos_idx != neg_idx, "Positive and negative should be different"

    print(f"✓ All triplets have distinct indices")


def test_build_triplet_dicts_validation():
    """Test that build_triplet_dicts filters out invalid triplets."""
    documents = [
        "Document A",
        "Document B",
        "Document C",
        "Document D",
    ]

    # Mix of valid and invalid triplets
    triplet_ids = [
        (0, 1, 2),  # Valid
        (0, 0, 2),  # Invalid: anchor == positive
        (1, 2, 3),  # Valid
        (2, 3, 2),  # Invalid: anchor == negative
        (3, 1, 1),  # Invalid: positive == negative
    ]

    # Build triplets (should filter out invalid ones)
    triplet_dicts = build_triplet_dicts(documents, triplet_ids)

    print(f"\n=== Triplet Validation Test ===")
    print(f"Input: {len(triplet_ids)} triplet IDs")
    print(f"Output: {len(triplet_dicts)} valid triplet dicts")

    # Should only keep the 2 valid triplets
    assert len(triplet_dicts) == 2, "Should filter out 3 invalid triplets"

    # Verify all remaining triplets have distinct indices
    for triplet_dict in triplet_dicts:
        anchor_id = triplet_dict["anchor_id"]
        positive_id = triplet_dict["positive_id"]
        negative_id = triplet_dict["negative_id"]

        assert anchor_id != positive_id, "Anchor and positive should be different"
        assert anchor_id != negative_id, "Anchor and negative should be different"
        assert positive_id != negative_id, "Positive and negative should be different"

    print(f"\n✓ Successfully validated triplet filtering")
