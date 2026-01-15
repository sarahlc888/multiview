"""Tests for triplet utilities."""

import pytest

from multiview.benchmark.task import Task


@pytest.mark.parametrize("dataset", ["abstractsim", "gsm8k", "rocstories", "crossword_clues", "hackernews", "analogies", "infinite_prompts", "infinite_chats", "dickinson", "moralfables", "onion_news", "goodreads_quotes"])
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

    for i, triplet in enumerate(task.triplets):
        anchor_idx, positive_idx, negative_idx = triplet
        print(f"\nTriplet {i}:")
        print(f"  [Anchor {anchor_idx}] {format_doc(task.documents[anchor_idx])}")
        print(f"  [Positive {positive_idx}] {format_doc(task.documents[positive_idx])}")
        print(f"  [Negative {negative_idx}] {format_doc(task.documents[negative_idx])}")

    # Basic assertions
    assert len(task.triplets) == 5
    assert all(len(triplet) == 3 for triplet in task.triplets)
    # Check that each triplet has distinct documents
    assert all(len({id(doc) for doc in triplet}) == 3 for triplet in task.triplets)


@pytest.mark.parametrize("dataset,criterion,docset_config", [
    ("abstractsim", "abstract_similarity", {"split": "validation"}),
    ("nytclustering", "topic", {"subset": "topic"}),
    ("ratemyprof", "cluster", {}),
    ("feedbacks", "cluster", {}),
    ("fewrel", "cluster", {}),
    ("fewnerd", "cluster", {}),
    ("fewevent", "cluster", {}),
    ("intent_emotion", "intent_similarity", {"subset": "intent"}),
    ("inspired", "movie_recommendation", {"split": "train"}),
    ("bills", "topic", {"text_field": "summary"}),
    ("bills", "subtopic", {"text_field": "summary"}),
    ("bills", "topic", {"text_field": "tokenized_text"}),
    ("trex", "relation", {"split": "validation", "min_relation_freq": 3}),
])
def test_create_prelabeled_triplets(dataset, criterion, docset_config):
    """Test prelabeled triplet creation across different datasets with known labels."""
    # Different datasets use different triplet styles for pre-made triplets
    if dataset == "trex":
        triplet_style = "kgc"
    elif dataset == "intent_emotion":
        triplet_style = "intent_emotion"
    else:
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
    if dataset == "intent_emotion":
        # IntentEmotion uses pre-made triplets - just verify they're valid
        for i, triplet in enumerate(task.triplets[:3]):  # Show first 3
            anchor_idx, positive_idx, negative_idx = triplet

            print(f"\nTriplet {i}:")
            print(f"  Anchor: {task.documents[anchor_idx]['text'][:60]}...")
            print(f"  Positive: {task.documents[positive_idx]['text'][:60]}...")
            print(f"  Negative: {task.documents[negative_idx]['text'][:60]}...")

            # Verify indices are distinct
            assert anchor_idx != positive_idx, f"{dataset}: Anchor and positive should be different"
            assert anchor_idx != negative_idx, f"{dataset}: Anchor and negative should be different"
            assert positive_idx != negative_idx, f"{dataset}: Positive and negative should be different"

    elif dataset == "trex":
        # KGC-specific validation
        for i, triplet in enumerate(task.triplets[:3]):  # Show first 3
            anchor_idx, positive_idx, negative_idx = triplet

            anchor = task.documents[anchor_idx]
            positive = task.documents[positive_idx]
            negative = task.documents[negative_idx]

            print(f"\nTriplet {i}:")
            print(f"  Anchor: {anchor['text']} (entity_type={anchor['entity_type']}, relation={anchor['relation']})")
            print(f"  Positive: {positive['text']} (entity_type={positive['entity_type']}, relation={positive['relation']})")
            print(f"  Negative: {negative['text']} (entity_type={negative['entity_type']}, relation={negative['relation']})")

            # Verify entity types
            assert anchor["entity_type"] == "head", f"{dataset}: Anchor should be HEAD entity"
            assert positive["entity_type"] == "tail", f"{dataset}: Positive should be TAIL entity"
            assert negative["entity_type"] == "tail", f"{dataset}: Negative should be TAIL entity"

            # Verify all share same relation
            assert anchor["relation"] == positive["relation"], f"{dataset}: Anchor and positive should share relation"
            assert anchor["relation"] == negative["relation"], f"{dataset}: Anchor and negative should share relation"

            # Verify positive is correct for anchor
            assert positive["entity_id"] == anchor["correct_tail_id"], f"{dataset}: Positive should match anchor's correct tail"

            # Verify negative is incorrect for anchor
            assert negative["entity_id"] != anchor["correct_tail_id"], f"{dataset}: Negative should not match anchor's correct tail"

            # All indices should be distinct
            assert anchor_idx != positive_idx, f"{dataset}: Anchor and positive should be different documents"
            assert anchor_idx != negative_idx, f"{dataset}: Anchor and negative should be different documents"
            assert positive_idx != negative_idx, f"{dataset}: Positive and negative should be different documents"
    else:
        # If a docset declares precomputed annotations for the criterion, verify those
        # annotations line up with the loaded documents (text -> criterion_value).
        if task.document_set.has_precomputed_annotations(criterion):
            precomputed = task.document_set.get_precomputed_annotations(criterion)
            assert len(precomputed) == len(task.documents), (
                f"{dataset}: Expected one precomputed annotation per document "
                f"for criterion={criterion}"
            )
            for doc in task.documents:
                assert doc["text"] in precomputed, (
                    f"{dataset}: Document text missing from precomputed annotations: "
                    f"{doc['text'][:80]}"
                )
                assert precomputed[doc["text"]]["criterion_value"] == doc.get(criterion), (
                    f"{dataset}: Precomputed criterion_value should match doc[{criterion}]"
                )

        # Standard prelabeled validation
        def get_label(doc, crit):
            """Get criterion label from document."""
            if isinstance(doc, dict):
                return doc.get(crit, "unknown")
            return "string_doc"

        for i, triplet in enumerate(task.triplets[:3]):  # Show first 3
            anchor_idx, positive_idx, negative_idx = triplet

            # Get labels
            anchor_label = get_label(task.documents[anchor_idx], criterion)
            pos_label = get_label(task.documents[positive_idx], criterion)
            neg_label = get_label(task.documents[negative_idx], criterion)

            print(f"\nTriplet {i}:")
            print(f"  Anchor: {anchor_label}")
            print(f"  Positive: {pos_label}")
            print(f"  Negative: {neg_label}")

            # Verify: anchor and positive should share same label
            assert anchor_label == pos_label, \
                f"{dataset}: Anchor and positive should have same label. Got {anchor_label} != {pos_label}"
            # Negative should have different label
            assert anchor_label != neg_label, \
                f"{dataset}: Anchor and negative should have different labels. Got {anchor_label} == {neg_label}"

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
