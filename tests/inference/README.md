# Inference Tests

Comprehensive tests for the multiview inference system.

## Setup

Set API keys as environment variables:

```bash
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export HF_API_KEY="your-huggingface-key"  # For HF embedding tests
```

Note: Tests will be skipped if the relevant API key is not set.

## Running Tests

Run all inference tests:
```bash
uv run pytest tests/inference/ -v
```

Run specific test files:
```bash
# Test LM inference (Gemini, Anthropic, OpenAI)
uv run pytest tests/inference/test_inference.py -v

# Test embedding models
uv run pytest tests/inference/test_embeddings.py -v
```

Run specific test classes:
```bash
# Test Gemini inference
uv run pytest tests/inference/test_inference.py::TestGeminiInference -v

# Test caching behavior
uv run pytest tests/inference/test_inference.py::TestCaching -v

# Test OpenAI embeddings
uv run pytest tests/inference/test_embeddings.py::TestOpenAIEmbeddings -v

# Test HuggingFace embeddings
uv run pytest tests/inference/test_embeddings.py::TestHuggingFaceEmbeddings -v
```

Run a specific test:
```bash
uv run pytest tests/inference/test_inference.py::TestCaching::test_caching_on_second_request -v
uv run pytest tests/inference/test_embeddings.py::TestEmbeddingCaching::test_embedding_caching_works -v
```

## Test Coverage

### LM Inference Tests (`test_inference.py`)

#### TestPresets
- ✅ List all available presets
- ✅ Get preset by name
- ✅ Error handling for invalid presets

#### TestGeminiInference
- ✅ Basic Gemini LM inference works
- ✅ JSON parsing from Gemini responses

#### TestCaching
- ✅ **Caching on second request**: Verifies identical requests use cache
- ✅ **Force refresh bypasses cache**: Verifies `force_refresh=True` ignores cache
- ✅ **Deduplication**: Verifies duplicate prompts are deduped before API calls

#### TestMultipleProviders
- ✅ Anthropic provider works
- ✅ OpenAI provider works

#### TestPresetUsage
- ✅ Using preset by string name
- ✅ Using preset with overrides

### Embedding Tests (`test_embeddings.py`)

#### TestOpenAIEmbeddings
- ✅ Basic OpenAI embeddings return correct dimensions (1536 for small, 3072 for large)
- ✅ Multiple texts are embedded correctly
- ✅ Both small and large models work

#### TestHuggingFaceEmbeddings
- ✅ Qwen3-Embedding-8B works via HF API
- ✅ Qwen3-Embedding-4B works via HF API
- ✅ Query instructions are properly applied

#### TestEmbeddingCaching
- ✅ **Embeddings are cached**: Second request uses cache
- ✅ **Embedding deduplication**: Duplicate texts deduplicated before API calls

#### TestEmbeddingInstructions
- ✅ **Query instruction applied**: Instruction prepended to queries
- ✅ **Doc instruction applied**: Instruction prepended to documents
- ✅ **Different instructions = different cache**: Same text with different instructions creates separate cache entries

#### TestVectorParser
- ✅ Vector parser returns list of numbers
- ✅ Parser handles dict with 'vector' key
- ✅ Parser handles raw vector input

## Key Test Behaviors

### Caching Test (`test_caching_on_second_request`)
1. Makes first request → hits API, creates cache file
2. Verifies cache file exists and has 1 entry
3. Makes identical second request → uses cache (no API call)
4. Verifies results are identical
5. Verifies cache size didn't grow (proves cache was used)

### Force Refresh Test (`test_force_refresh_bypasses_cache`)
1. Makes first request → populates cache
2. Manually modifies cache to contain "CACHED_VALUE"
3. Request without `force_refresh` → returns "CACHED_VALUE" (proves cache works)
4. Request with `force_refresh=True` → returns fresh result (proves cache bypassed)

### Deduplication Test (`test_deduplication`)
1. Submits 4 inputs with only 2 unique values: ["hello", "hello", "world", "hello"]
2. Verifies 4 results returned (matching input length)
3. Verifies cache has only 2 entries (proves deduplication worked)

## Expected Behavior

- Tests requiring API keys will be **skipped** if keys are not set (not failed)
- All tests use temporary directories for cache files (cleaned up automatically)
- Tests should complete in ~30-60 seconds (depending on API latency)
- Cache files are verified by reading JSON directly

## Troubleshooting

**Tests skipped?**
- Check that API keys are set as environment variables
- Use `export` in bash/zsh, not just `KEY=value pytest`

**Rate limit errors?**
- Wait a few minutes and retry
- Tests use temperature=0 and small max_tokens to minimize cost

**Import errors?**
- Make sure dependencies are installed: `uv sync`
- Install provider packages: `pip install google-genai anthropic openai`
