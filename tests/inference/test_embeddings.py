"""Tests for embedding model inference.

Tests cover:
- OpenAI embeddings
- HuggingFace API embeddings (Qwen3)
- Embedding instructions (query/doc)
- Vector parsing
- Caching for embeddings
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from multiview.inference import InferenceConfig, run_inference


@pytest.mark.external
class TestOpenAIEmbeddings:
    """Test OpenAI embedding models."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_openai_embedding_basic(self):
        """Test basic OpenAI embedding."""
        config = InferenceConfig(
            provider="openai",
            model_name="text-embedding-3-small",
            prompt_template="{text}",
            is_embedding=True,
            parser="vector",
        )

        results = run_inference(
            inputs={"text": ["Hello world"]},
            config=config,
        )

        assert len(results) == 1
        vector = results[0]
        # OpenAI text-embedding-3-small has 1536 dimensions
        assert isinstance(vector, list)
        assert len(vector) == 1536
        assert all(isinstance(x, (int, float)) for x in vector)

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_openai_embedding_multiple_texts(self):
        """Test OpenAI embeddings with multiple texts."""
        results = run_inference(
            inputs={"document": ["First text", "Second text", "Third text"]},
            config="openai_embedding_small",
        )

        assert len(results) == 3
        for vector in results:
            assert isinstance(vector, list)
            assert len(vector) == 1536

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_openai_embedding_large(self):
        """Test OpenAI large embedding model."""
        results = run_inference(
            inputs={"documents": ["Test text"]},
            config="openai_embedding_large",
        )

        assert len(results) == 1
        vector = results[0]
        # text-embedding-3-large has 3072 dimensions
        assert isinstance(vector, list)
        assert len(vector) == 3072


@pytest.mark.external
class TestHuggingFaceEmbeddings:
    """Test HuggingFace API embedding models."""

    @pytest.mark.skipif(
        not (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
            or os.getenv("HUGGINGFACE_API_KEY")
        ),
        reason="HF_TOKEN not set",
    )
    def test_hf_qwen3_8b_embedding(self):
        """Test Qwen3-Embedding-8B via HF API."""
        config = InferenceConfig(
            provider="hf_api",
            model_name="Qwen/Qwen3-Embedding-8B",
            prompt_template="{text}",
            is_embedding=True,
            parser="vector",
        )

        results = run_inference(
            inputs={"text": ["Hello world"]},
            config=config,
        )

        assert len(results) == 1
        vector = results[0]
        assert isinstance(vector, list)
        # Qwen3-Embedding-8B has 8192 dimensions
        assert len(vector) > 1000  # At least verify it's a large vector
        assert all(isinstance(x, (int, float)) for x in vector)

    @pytest.mark.skipif(
        not (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
            or os.getenv("HUGGINGFACE_API_KEY")
        ),
        reason="HF_TOKEN not set",
    )
    def test_hf_qwen3_4b_embedding(self):
        """Test Qwen3-Embedding-8B via HF API (4B not available)."""
        # Note: Using 8B because 4B is not available via HF Inference API
        results = run_inference(
            inputs={"documents": ["Test document"]},
            config="hf_qwen3_embedding_8b",  # Using 8B instead of 4B
        )

        assert len(results) == 1
        vector = results[0]
        assert isinstance(vector, list)
        assert len(vector) > 1000

    @pytest.mark.skipif(
        not (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
            or os.getenv("HUGGINGFACE_API_KEY")
        ),
        reason="HF_TOKEN not set",
    )
    def test_hf_embedding_with_query_instruction(self):
        """Test HF embeddings with query instruction."""
        config = InferenceConfig(
            provider="hf_api",
            model_name="Qwen/Qwen3-Embedding-8B",
            prompt_template="{text}",
            embed_query_instr_template="Represent this query for retrieval: ",
            is_embedding=True,
            parser="vector",
        )

        results = run_inference(
            inputs={"text": ["What is machine learning?"]},
            config=config,
        )

        assert len(results) == 1
        vector = results[0]
        assert isinstance(vector, list)
        assert len(vector) > 1000


@pytest.mark.external
class TestEmbeddingCaching:
    """Test caching behavior for embeddings."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_embedding_caching_works(self):
        """Test that embeddings are cached on second request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "embedding_cache.json"

            config = InferenceConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                prompt_template="{text}",
                is_embedding=True,
                parser="vector",
            )

            inputs = {"text": ["Embedding test text"]}

            # First request
            results1 = run_inference(
                inputs=inputs,
                config=config,
                cache_path=str(cache_path),
            )

            # Check cache file exists
            assert cache_path.exists()
            with open(cache_path) as f:
                cache_data = json.load(f)
            assert "completions" in cache_data
            initial_size = len(cache_data["completions"])

            # Second request - should use cache
            results2 = run_inference(
                inputs=inputs,
                config=config,
                cache_path=str(cache_path),
            )

            # Results should be identical
            assert results1 == results2

            # Cache size should not have grown
            with open(cache_path) as f:
                cache_data = json.load(f)
            assert len(cache_data["completions"]) == initial_size

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_embedding_deduplication(self):
        """Test that duplicate texts are deduplicated for embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "dedup_cache.json"

            config = InferenceConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                prompt_template="{text}",
                is_embedding=True,
                parser="vector",
            )

            # 5 inputs with only 2 unique values
            inputs = {
                "text": [
                    "unique text one",
                    "unique text two",
                    "unique text one",
                    "unique text two",
                    "unique text one",
                ]
            }

            results = run_inference(
                inputs=inputs,
                config=config,
                cache_path=str(cache_path),
            )

            # Should return 5 results
            assert len(results) == 5

            # But cache should only have 2 entries
            with open(cache_path) as f:
                cache_data = json.load(f)
            assert len(cache_data["completions"]) == 2

            # Verify results at indices 0, 2, 4 are identical (all "unique text one")
            assert results[0] == results[2] == results[4]
            # Verify results at indices 1, 3 are identical (all "unique text two")
            assert results[1] == results[3]
            # Verify the two unique texts have different embeddings
            assert results[0] != results[1]


@pytest.mark.external
class TestEmbeddingInstructions:
    """Test embedding instructions (query/doc side)."""

    @pytest.mark.skipif(
        not (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
            or os.getenv("HUGGINGFACE_API_KEY")
        ),
        reason="HF_TOKEN not set",
    )
    def test_query_instruction_applied(self):
        """Test that query instruction is properly applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "instr_cache.json"

            config = InferenceConfig(
                provider="hf_api",
                model_name="Qwen/Qwen3-Embedding-8B",
                prompt_template="{text}",
                embed_query_instr_template="Query: ",
                is_embedding=True,
                parser="vector",
            )

            results = run_inference(
                inputs={"text": ["search query"]},
                config=config,
                cache_path=str(cache_path),
            )

            assert len(results) == 1

            # Check the cache to verify the prompt includes instruction
            with open(cache_path) as f:
                cache_data = json.load(f)

            # The stored prompt (in cache values) should include the instruction
            cache_entries = cache_data["completions"]
            assert len(cache_entries) == 1
            # Get the first (and only) cached entry
            cached_prompt = list(cache_entries.values())[0]["prompt"]
            # Should have "Query: " prepended to "search query"
            assert "Query: " in cached_prompt
            assert "search query" in cached_prompt

    @pytest.mark.skipif(
        not (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
            or os.getenv("HUGGINGFACE_API_KEY")
        ),
        reason="HF_TOKEN not set",
    )
    def test_doc_instruction_applied(self):
        """Test that document instruction is properly applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "doc_instr_cache.json"

            config = InferenceConfig(
                provider="hf_api",
                model_name="Qwen/Qwen3-Embedding-8B",
                prompt_template="{text}",
                embed_doc_instr_template="Document: ",
                is_embedding=True,
                parser="vector",
            )

            results = run_inference(
                inputs={"text": ["document text"]},
                config=config,
                cache_path=str(cache_path),
            )

            assert len(results) == 1

            # Check the cached prompt includes doc instruction
            with open(cache_path) as f:
                cache_data = json.load(f)

            cache_entries = cache_data["completions"]
            cached_prompt = list(cache_entries.values())[0]["prompt"]
            assert "Document: " in cached_prompt
            assert "document text" in cached_prompt

    @pytest.mark.skipif(
        not (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
            or os.getenv("HUGGINGFACE_API_KEY")
        ),
        reason="HF_TOKEN not set",
    )
    def test_different_instructions_different_cache_keys(self):
        """Test that different instructions create different cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "multi_instr_cache.json"

            # Same text, different instructions
            base_text = "machine learning"

            # Query side
            config_query = InferenceConfig(
                provider="hf_api",
                model_name="Qwen/Qwen3-Embedding-8B",
                prompt_template="{text}",
                embed_query_instr_template="Query: ",
                is_embedding=True,
                parser="vector",
            )

            results_query = run_inference(
                inputs={"text": [base_text]},
                config=config_query,
                cache_path=str(cache_path),
            )

            # Doc side
            config_doc = InferenceConfig(
                provider="hf_api",
                model_name="Qwen/Qwen3-Embedding-8B",
                prompt_template="{text}",
                embed_doc_instr_template="Document: ",
                is_embedding=True,
                parser="vector",
            )

            results_doc = run_inference(
                inputs={"text": [base_text]},
                config=config_doc,
                cache_path=str(cache_path),
            )

            # Should have 2 cache entries (different packed prompts)
            with open(cache_path) as f:
                cache_data = json.load(f)
            assert len(cache_data["completions"]) == 2

            # Embeddings should be different (different instructions)
            # Note: May be similar but shouldn't be identical
            assert results_query[0] != results_doc[0]


class TestVectorParser:
    """Test that vector parser correctly extracts embeddings."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    @pytest.mark.external
    def test_vector_parser_returns_list(self):
        """Test vector parser returns a list of numbers."""
        results = run_inference(
            inputs={"documents": ["test"]},
            config="openai_embedding_small",
        )

        vector = results[0]
        assert isinstance(vector, list)
        assert all(isinstance(x, (int, float)) for x in vector)

    def test_vector_parser_with_dict_input(self):
        """Test vector parser can handle dict with 'vector' key."""
        from multiview.inference.parsers import vector_parser

        test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        completion_dict = {"vector": test_vector}

        result = vector_parser(completion_dict)
        assert result == test_vector

    def test_vector_parser_with_raw_vector(self):
        """Test vector parser can handle raw vector input."""
        from multiview.inference.parsers import vector_parser

        test_vector = [0.1, 0.2, 0.3]
        result = vector_parser(test_vector)
        assert result == test_vector
