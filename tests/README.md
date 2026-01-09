# Tests

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/benchmark/test_annotations.py -v

# Run without API keys (local tests only)
pytest tests/benchmark/test_annotation_utils.py -v

# Run with coverage
pytest tests/ --cov=multiview --cov-report=html
```

## Test Files

### Benchmark Tests
- **test_annotation_utils.py** - Helper utilities (runs locally, no API needed)
- **test_annotations.py** - Schema generation and annotation functions (needs API)
- **test_annotation_integration.py** - End-to-end Task workflows (needs API)
- **test_class_schema.py** - Category schema tests
- **test_triplet_utils.py** - Triplet creation tests

### Inference Tests
- **test_inference.py** - Core inference system
- **test_annotator_presets.py** - Preset configurations
- **test_embeddings.py** - Embedding model tests
- **test_concurrent.py** - Concurrent execution
- **test_prompts.py** - Prompt formatting
- **test_parsers.py** - Output parsing

## API Keys

Some tests require API keys and are skipped by default:

```bash
export GEMINI_API_KEY=your_key
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

Tests that need API keys are marked with `@pytest.mark.skip`.
