"""LM judge presets for pairwise, triplet, and quality rating."""

from __future__ import annotations

from ._base import InferenceConfig

LM_JUDGE_PRESETS = {
    # ========================================================================
    # LM JUDGE - PAIRWISE COMPARISON
    # ========================================================================
    # Likert scale judge (1-5)
    "lmjudge_pair_plaintext_likerthard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/pair_plaintext_likerthard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?2(?:\]|\*\*)?": 2,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?3(?:\]|\*\*)?": 3,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?4(?:\]|\*\*)?": 4,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?5(?:\]|\*\*)?": 5,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    # Binary judge (0=same, 1=different)
    "lmjudge_pair_norewrite_binaryhard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/pair_norewrite_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?0(?:\]|\*\*)?": 0,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
            }
        },
        temperature=0.7,
        max_tokens=4096,
    ),
    # Likert scale judge with annotations (1-5)
    "lmjudge_pair_with_annotation_likerthard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/pair_with_annotation_likerthard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?2(?:\]|\*\*)?": 2,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?3(?:\]|\*\*)?": 3,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?4(?:\]|\*\*)?": 4,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?5(?:\]|\*\*)?": 5,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    # Binary judge with annotations (0=same, 1=different)
    "lmjudge_pair_with_annotation_binaryhard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/pair_with_annotation_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?0(?:\]|\*\*)?": 0,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
            }
        },
        temperature=0.7,
        max_tokens=4096,
    ),
    # ========================================================================
    # LM JUDGE - TRIPLET COMPARISON
    # ========================================================================
    # Gemini Flash Lite (default)
    "lmjudge_triplet_plaintext_binaryhard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/triplet_plaintext_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    "lmjudge_triplet_with_annotation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    "lmjudge_triplet_plaintext_binaryhard_with_annotation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    # Gemini Flash
    "lmjudge_triplet_plaintext_binaryhard_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/lm_judge/triplet_plaintext_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    "lmjudge_triplet_plaintext_binaryhard_gemini_flash_lite": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/triplet_plaintext_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    "lmjudge_triplet_with_annotation_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    "lmjudge_triplet_plaintext_binaryhard_with_annotation_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    # Gemini Pro
    "lmjudge_triplet_plaintext_binaryhard_gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/lm_judge/triplet_plaintext_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    "lmjudge_triplet_with_annotation_gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    "lmjudge_triplet_plaintext_binaryhard_with_annotation_gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(b\)|\$\\boxed\{(?:\\text\{)?\(?b\)?\}?\}?|\(b\)\s*$)": 1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(c\)|\$\\boxed\{(?:\\text\{)?\(?c\)?\}?\}?|\(c\)\s*$)": -1,
                r"(?:Final\s+[Aa]nswer(?:\s+is)?:\s*\(d\)|\$\\boxed\{(?:\\text\{)?\(?d\)?\}?\}?|\(d\)\s*$)": 0.0,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    # ========================================================================
    # LM JUDGE - QUALITY RATING
    # ========================================================================
    "lmjudge_quality_rating_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/quality/quality_rating.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Rr]ating:\s*1": 1,
                r"Final\s+[Rr]ating:\s*2": 2,
                r"Final\s+[Rr]ating:\s*3": 3,
                r"Final\s+[Rr]ating:\s*4": 4,
                r"Final\s+[Rr]ating:\s*5": 5,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    "lmjudge_quality_rating_gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/quality/quality_rating.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Rr]ating:\s*1": 1,
                r"Final\s+[Rr]ating:\s*2": 2,
                r"Final\s+[Rr]ating:\s*3": 3,
                r"Final\s+[Rr]ating:\s*4": 4,
                r"Final\s+[Rr]ating:\s*5": 5,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
    "lmjudge_quality_rating_with_annotation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/quality/quality_rating_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Rr]ating:\s*1": 1,
                r"Final\s+[Rr]ating:\s*2": 2,
                r"Final\s+[Rr]ating:\s*3": 3,
                r"Final\s+[Rr]ating:\s*4": 4,
                r"Final\s+[Rr]ating:\s*5": 5,
            }
        },
        temperature=0.7,
        max_tokens=8192,
    ),
}
