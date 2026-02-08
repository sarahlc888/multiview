# Task: Add ChatBot Arena (Arena-Hard v2.0) Dataset

## Goal
Create a new docset for ChatBot Arena Hard v2.0 benchmark data, enabling multiview evaluation across different LLM response qualities, instruction-following abilities, and output styles.

## Type
ADD_DATASET

## Background
Arena-Hard v2.0 is an automatic LLM benchmark from LMSYS with 500 challenging real-world queries (software engineering, math, etc.) and 250 creative writing prompts sourced from Chatbot Arena. Multiple LLM responses per question enable comparison across models.

## Data source
- **Repository**: https://github.com/lmarena/arena-hard-auto (also on HuggingFace: https://huggingface.co/datasets/lmarena-ai/arena-hard-auto)
- **Structure**:
  - `data/arena-hard-v2.0/question.jsonl` — Questions with UIDs
  - `data/arena-hard-v2.0/model_answer/` — Model response JSONL files
  - `data/arena-hard-v2.0/model_judgment/` — Judgments (optional)
- **Size**: 1.89 GB total
- **License**: Apache 2.0

## Approach

### Document structure decision
Two options for how to structure documents:

**Option A: Flat (each answer = 1 document)** ✅ Recommended
- Each (question, answer) pair is a separate document
- Simpler, more documents (~thousands)
- Easier to evaluate individual responses
- Can add metadata for question_id, model, etc.

**Option B: Grouped (each question with all answers = 1 document)**
- Matches s3 repo pattern better
- More complex document structure (dict with outputs list)
- Fewer documents (~750)

### Criteria ideas
- `response_quality` — overall quality of the LLM response
- `instruction_following` — how well it follows the prompt
- `technical_accuracy` — correctness for technical/math questions
- `creativity` — for creative writing prompts
- `verbosity` — response length/detail level
- `tone` — formal, casual, technical, etc.
- `domain` — software engineering, math, creative writing, etc.

## Implementation steps

### 1. Setup data downloading (following AidanBench pattern)
- [ ] Add constants to `src/multiview/constants.py`:
  ```python
  CHATBOTARENA_REPO_URL = "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto"
  CHATBOTARENA_CACHE_DIR = CACHE_ROOT / "arena-hard-auto"
  CHATBOTARENA_DATA_DIR = CHATBOTARENA_CACHE_DIR / "data" / "arena-hard-v2.0"
  CHATBOTARENA_QUESTIONS_PATH = CHATBOTARENA_DATA_DIR / "question.jsonl"
  CHATBOTARENA_ANSWERS_DIR = CHATBOTARENA_DATA_DIR / "model_answer"
  ```

### 2. Create docset class
- [ ] Create `src/multiview/docsets/chatbotarena.py`
  - [ ] Inherit from `BaseDocSet`
  - [ ] Set class attributes: `DATASET_PATH`, `DESCRIPTION`, `DOCUMENT_TYPE`, `DATASET_NAME`, `KNOWN_CRITERIA`
  - [ ] Implement `_ensure_repo_cloned()` method (copy from aidanbench.py)
  - [ ] Implement `load_documents()`:
    - Load questions from `question.jsonl`
    - Load model answers from `model_answer/` directory
    - Group answers by UID
    - Flatten to (question, answer) pairs OR keep grouped (decide on structure)
    - Apply `max_docs` limit if specified
    - Call `_deduplicate()` before returning
  - [ ] Implement `get_document_text()` to extract text
  - [ ] Optional: `get_known_criterion_value()` for any deterministic criteria (e.g., domain from question metadata)
  - [ ] Optional: `get_document_metadata()` for visualization (question_id, model, etc.)

### 3. Register docset
- [ ] Import in `src/multiview/docsets/__init__.py`
- [ ] Add to `DOCSETS` dict: `"chatbotarena": ChatBotArenaDocSet`
- [ ] Add to `__all__`

### 4. Add criteria metadata
- [ ] Add `chatbotarena:` section to `configs/available_criteria.yaml`
- [ ] Define criteria with descriptions and hints (see criteria ideas above)

### 5. Testing
- [ ] Test data download/cloning works
- [ ] Test docset loads correctly with `max_docs` limit
- [ ] Test `python scripts/create_eval.py` with chatbotarena docset
- [ ] Test visualization in viewer

## Reference files
- **Pattern to copy**: `src/multiview/docsets/aidanbench.py` — has git clone pattern
- **Base class**: `src/multiview/docsets/base.py` — BaseDocSet abstract class
- **S3 implementation**: `/Users/sarahchen/code/pproj/project/src/s3/benchmark/utils/factory_utils/custom_tasks/chatbotarena.py` — original loading logic (108 lines)
- **Constants**: `src/multiview/constants.py` — add repo URLs and paths here
- **Registration**: `src/multiview/docsets/__init__.py` — register new docset
- **Criteria**: `configs/available_criteria.yaml` — add criteria definitions

## Complexity estimate
**2-4 hours total**:
- 1-2 hours: Core implementation (adapt s3 loading logic to BaseDocSet pattern)
- 0.5-1 hour: Data setup, download testing
- 0.5-1 hour: Criterion metadata configuration

## Exit criteria
- [ ] New file `src/multiview/docsets/chatbotarena.py` exists and follows BaseDocSet pattern
- [ ] Constants added to `src/multiview/constants.py`
- [ ] Registered in `src/multiview/docsets/__init__.py`
- [ ] Criteria defined in `configs/available_criteria.yaml`
- [ ] Data auto-downloads on first use (via git clone)
- [ ] Can run `python scripts/create_eval.py` with chatbotarena docset successfully
- [ ] Results visualize correctly in viewer

## Notes
- Consider train/test split if needed for future finetuning work (s3 code has this)
- May want to filter out very long responses (s3 uses max_char_len=8000)
- Could add config for `candidates_per_question` to limit answers per question
- Model judgment data available if we want to add "judge_preference" as a known criterion
