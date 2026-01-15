"""High-signal smoke tests for the example image docset.

This docset is intentionally a placeholder. These tests aim to validate the
`BaseDocSet` contract and a couple of config toggles, without locking us into
the exact demo schema (which is expected to change).
"""

from multiview.docsets import DOCSETS


def test_example_images_smoke_load_and_accessors():
    docset = DOCSETS["example_images"]({"max_docs": 3})
    docs = docset.load_documents()

    assert 0 < len(docs) <= 3

    for doc in docs:
        assert isinstance(doc, dict)
        assert isinstance(doc.get("text"), str) and doc["text"]
        assert isinstance(doc.get("image_path"), str) and doc["image_path"]
        assert isinstance(doc.get("category"), str) and doc["category"]

        assert docset.get_document_text(doc) == doc["text"]
        assert docset.get_document_image(doc) == doc["image_path"]

        # This method exists on the docset; keep behavior stable without
        # asserting that "category" must be in KNOWN_CRITERIA.
        assert docset.get_known_criterion_value(doc, "category") == doc["category"]
        assert docset.get_known_criterion_value(doc, "does_not_exist") is None


def test_example_images_images_dir_and_use_urls_toggle():
    custom_dir = "custom/path/to/images"
    docset_paths = DOCSETS["example_images"]({"max_docs": 2, "images_dir": custom_dir})
    docs_paths = docset_paths.load_documents()
    assert 0 < len(docs_paths) <= 2
    for doc in docs_paths:
        assert doc["image_path"].startswith(custom_dir)

    docset_urls = DOCSETS["example_images"]({"max_docs": 2, "use_urls": True})
    docs_urls = docset_urls.load_documents()
    assert 0 < len(docs_urls) <= 2
    for doc in docs_urls:
        # In URL mode, the demo uses bare filenames.
        assert doc["image_path"] in {"sample1.jpg", "sample2.jpg", "sample3.jpg", "sample4.jpg", "sample5.jpg"}
