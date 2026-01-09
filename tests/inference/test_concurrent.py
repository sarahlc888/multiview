"""Tests for concurrent API request execution.

Tests ThreadPoolExecutor integration in provider functions.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from multiview.inference.providers.anthropic import (
    _anthropic_single_completion,
    anthropic_completions,
)
from multiview.inference.providers.openai import (
    _openai_single_completion,
    openai_completions,
)
from multiview.inference.providers.gemini import (
    _gemini_single_completion,
    gemini_completions,
)


class TestConcurrentExecution:
    """Test concurrent execution with ThreadPoolExecutor."""

    def test_openai_sequential_vs_concurrent(self):
        """Test that concurrent execution is faster than sequential."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]

        # Add delay to simulate API latency
        def slow_create(*args, **kwargs):
            time.sleep(0.1)  # 100ms per call
            return mock_response

        mock_client.chat.completions.create = slow_create

        # Test sequential (max_workers=1)
        prompts = ["test"] * 5
        with patch("multiview.inference.providers.openai.openai.OpenAI", return_value=mock_client):
            with patch("multiview.inference.providers.openai.OPENAI_API_KEYS", ["test-key"]):
                start = time.time()
                result_seq = openai_completions(
                    prompts=prompts,
                    model_name="gpt-4.1-mini",
                    max_workers=1,
                    max_retries=1,
                )
                time_seq = time.time() - start

        # Test concurrent (max_workers=5)
        with patch("multiview.inference.providers.openai.openai.OpenAI", return_value=mock_client):
            with patch("multiview.inference.providers.openai.OPENAI_API_KEYS", ["test-key"]):
                start = time.time()
                result_conc = openai_completions(
                    prompts=prompts,
                    model_name="gpt-4.1-mini",
                    max_workers=5,
                    max_retries=1,
                )
                time_conc = time.time() - start

        # Verify results are correct
        assert len(result_seq["completions"]) == 5
        assert len(result_conc["completions"]) == 5
        assert all(c["text"] == "test response" for c in result_seq["completions"])
        assert all(c["text"] == "test response" for c in result_conc["completions"])

        # Concurrent should be significantly faster
        # Sequential: ~500ms (5 * 100ms), Concurrent: ~100ms (1 * 100ms)
        # Allow some overhead, but concurrent should be at least 2x faster
        assert time_conc < time_seq / 2, f"Concurrent ({time_conc:.2f}s) not faster than sequential ({time_seq:.2f}s)"

    def test_anthropic_concurrent_execution(self):
        """Test that Anthropic provider supports concurrent execution."""
        # Mock Anthropic client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="anthropic response")]

        def slow_create(*args, **kwargs):
            time.sleep(0.05)
            return mock_response

        mock_client.messages.create = slow_create

        prompts = ["test"] * 4
        with patch("multiview.inference.providers.anthropic.anthropic.Anthropic", return_value=mock_client):
            with patch("multiview.inference.providers.anthropic.ANTHROPIC_API_KEY", "test-key"):
                start = time.time()
                result = anthropic_completions(
                    prompts=prompts,
                    model_name="claude-3-5-sonnet-20241022",
                    max_workers=4,
                    max_retries=1,
                )
                elapsed = time.time() - start

        # Verify results
        assert len(result["completions"]) == 4
        assert all(c["text"] == "anthropic response" for c in result["completions"])

        # Should be faster than sequential (4 * 50ms = 200ms)
        # With 4 workers, should be ~50ms plus overhead
        assert elapsed < 0.15, f"Concurrent execution too slow: {elapsed:.2f}s"

    def test_gemini_concurrent_execution(self):
        """Test that Gemini provider supports concurrent execution."""
        # Mock Gemini client and response
        mock_client = MagicMock()
        mock_response = MagicMock(text="gemini response")

        def slow_generate(*args, **kwargs):
            time.sleep(0.05)
            return mock_response

        mock_client.models.generate_content = slow_generate

        prompts = ["test"] * 4
        with patch("multiview.inference.providers.gemini.genai.Client", return_value=mock_client):
            with patch("multiview.inference.providers.gemini.GEMINI_API_KEY", "test-key"):
                start = time.time()
                result = gemini_completions(
                    prompts=prompts,
                    model_name="gemini-2.5-flash-lite",
                    max_workers=4,
                    max_retries=1,
                )
                elapsed = time.time() - start

        # Verify results
        assert len(result["completions"]) == 4
        assert all(c["text"] == "gemini response" for c in result["completions"])

        # Should be faster than sequential
        assert elapsed < 0.15, f"Concurrent execution too slow: {elapsed:.2f}s"

    def test_order_preserved_with_concurrency(self):
        """Test that output order matches input order with concurrent execution."""
        # Mock client that returns the prompt as the response
        mock_client = MagicMock()

        def echo_prompt(*args, messages=None, **kwargs):
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content=messages[0]["content"]))]
            time.sleep(0.01)  # Small delay
            return response

        mock_client.chat.completions.create = echo_prompt

        # Use distinct prompts to verify order
        prompts = [f"prompt_{i}" for i in range(10)]

        with patch("multiview.inference.providers.openai.openai.OpenAI", return_value=mock_client):
            with patch("multiview.inference.providers.openai.OPENAI_API_KEYS", ["test-key"]):
                result = openai_completions(
                    prompts=prompts,
                    model_name="gpt-4.1-mini",
                    max_workers=5,
                )

        # Verify order is preserved
        assert len(result["completions"]) == 10
        for i, completion in enumerate(result["completions"]):
            assert completion["text"] == f"prompt_{i}", f"Order not preserved at index {i}"

    def test_prefills_with_concurrency(self):
        """Test that prefills work correctly with concurrent execution."""
        mock_client = MagicMock()

        def mock_create(*args, messages=None, **kwargs):
            response = MagicMock()
            # Return a completion that we can verify was prefilled
            response.choices = [MagicMock(message=MagicMock(content=" completed"))]
            return response

        mock_client.chat.completions.create = mock_create

        prompts = ["prompt1", "prompt2", "prompt3"]
        prefills = ["pre1", "pre2", "pre3"]

        with patch("multiview.inference.providers.openai.openai.OpenAI", return_value=mock_client):
            with patch("multiview.inference.providers.openai.OPENAI_API_KEYS", ["test-key"]):
                result = openai_completions(
                    prompts=prompts,
                    model_name="gpt-4.1-mini",
                    force_prefills=prefills,
                    max_workers=3,
                )

        # Verify prefills were applied
        assert len(result["completions"]) == 3
        assert result["completions"][0]["text"] == "pre1 completed"
        assert result["completions"][1]["text"] == "pre2 completed"
        assert result["completions"][2]["text"] == "pre3 completed"


class TestRetryLogic:
    """Test exponential backoff retry logic."""

    def test_openai_retry_with_backoff(self):
        """Test that OpenAI retries with exponential backoff on rate limit."""
        mock_client = MagicMock()

        # Fail twice, then succeed
        call_count = 0
        call_times = []

        def rate_limited_create(*args, **kwargs):
            nonlocal call_count
            call_times.append(time.time())
            call_count += 1

            if call_count <= 2:
                # Import the real exception class
                import openai
                raise openai.RateLimitError("Rate limit", response=MagicMock(), body=None)

            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="success"))]
            return response

        mock_client.chat.completions.create = rate_limited_create

        with patch("multiview.inference.providers.openai.openai.OpenAI", return_value=mock_client):
            with patch("multiview.inference.providers.openai.OPENAI_API_KEYS", ["test-key"]):
                result = _openai_single_completion(
                    client=mock_client,
                    prompt="test",
                    prefill=None,
                    model_name="gpt-4.1-mini",
                    temperature=0.0,
                    max_tokens=100,
                    max_retries=5,
                    initial_retry_delay=0.1,  # Short delay for testing
                    retry_backoff_factor=2.0,
                )

        # Verify success after retries
        assert result["text"] == "success"
        assert call_count == 3  # 2 failures + 1 success

        # Verify exponential backoff timing
        # First call -> wait 0.1s -> second call -> wait 0.2s -> third call
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            # Allow some timing variance
            assert 0.08 < delay1 < 0.15, f"First delay should be ~0.1s, got {delay1:.3f}s"
            assert 0.18 < delay2 < 0.25, f"Second delay should be ~0.2s, got {delay2:.3f}s"

    def test_max_retries_exhausted(self):
        """Test that function returns empty completion after exhausting retries."""
        mock_client = MagicMock()

        # Always fail
        def always_fail(*args, **kwargs):
            import openai
            raise openai.RateLimitError("Rate limit", response=MagicMock(), body=None)

        mock_client.chat.completions.create = always_fail

        result = _openai_single_completion(
            client=mock_client,
            prompt="test",
            prefill=None,
            model_name="gpt-4.1-mini",
            temperature=0.0,
            max_tokens=100,
            max_retries=2,  # Only 2 retries
            initial_retry_delay=0.01,  # Very short for testing
            retry_backoff_factor=2.0,
        )

        # Should return empty completion after exhausting retries
        assert result["text"] == ""

    def test_gemini_quota_vs_rate_limit(self):
        """Test that Gemini distinguishes between quota exhaustion and rate limits."""
        mock_client = MagicMock()

        # Test quota exhaustion (should not retry)
        def quota_exceeded(*args, **kwargs):
            raise Exception("quota exceeded for the day")

        mock_client.models.generate_content = quota_exceeded

        result = _gemini_single_completion(
            client=mock_client,
            prompt="test",
            prefill=None,
            model_name="gemini-2.5-flash-lite",
            temperature=0.0,
            max_tokens=100,
            max_retries=5,
            initial_retry_delay=0.01,
        )

        # Should return empty immediately without retrying
        assert result["text"] == ""

        # Test rate limit (should retry)
        call_count = 0

        def rate_limited(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise Exception("429 rate limit error")
            response = MagicMock(text="success")
            return response

        mock_client.models.generate_content = rate_limited

        result = _gemini_single_completion(
            client=mock_client,
            prompt="test",
            prefill=None,
            model_name="gemini-2.5-flash-lite",
            temperature=0.0,
            max_tokens=100,
            max_retries=5,
            initial_retry_delay=0.01,
        )

        # Should succeed after retry
        assert result["text"] == "success"
        assert call_count == 2  # 1 failure + 1 success
