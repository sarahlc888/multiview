#!/usr/bin/env python3
"""Generate a markdown table showing all available tasks with example triplets.

Scans existing triplet directories and extracts example triplets.
Falls back to available_criteria.yaml descriptions where triplets don't exist.
"""

import json
import re
from pathlib import Path
from typing import Any

import yaml

# Project root
project_root = Path(__file__).parent.parent


def load_criteria() -> dict[str, dict[str, Any]]:
    """Load available_criteria.yaml."""
    criteria_path = project_root / "configs" / "available_criteria.yaml"
    with open(criteria_path) as f:
        return yaml.safe_load(f)


def truncate_text(text: str, max_length: int = 300) -> str:
    """Truncate text to max_length characters."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def map_docset_key(yaml_key: str) -> str | None:
    """Map YAML keys to docset registry keys.

    Args:
        yaml_key: Key from available_criteria.yaml (e.g., 'gsm8k', 'crossword')

    Returns:
        Corresponding key in DOCSETS registry, or None if not found
    """
    mapping = {
        "crossword": "crossword_clues",
        # Add other mappings as needed
    }
    return mapping.get(yaml_key)


def find_triplet_files() -> dict[str, Path]:
    """Find all triplet JSON files in outputs directory.

    Returns:
        Dict mapping task_name to triplet file path (base name without suffixes)
    """
    triplet_files = {}
    outputs_dir = project_root / "outputs"

    if not outputs_dir.exists():
        return triplet_files

    # Find all triplets.json and triplets.jsonl files (prefer .json)
    for pattern in ["triplets.json", "triplets.jsonl"]:
        for triplet_file in outputs_dir.rglob(pattern):
            # Extract task name from path: outputs/run_name/triplets/task_name/triplets.json
            full_task_name = triplet_file.parent.name

            # Remove common suffixes like __pre__10, __tag__5, __class__10, etc.
            # These indicate prelabeled data, tag schemas, class schemas, or other variations
            base_task_name = full_task_name

            # Remove various suffix patterns
            for suffix_pattern in ["__pre__", "__tag__", "__class__"]:
                if suffix_pattern in base_task_name:
                    base_task_name = base_task_name.split(suffix_pattern)[0]
                    break  # Only remove first matching suffix

            # Only add if not already found (prefer .json over .jsonl, prefer non-suffixed over suffixed)
            if base_task_name not in triplet_files:
                triplet_files[base_task_name] = triplet_file

    return triplet_files


def load_quality_distribution(triplet_file: Path) -> str:
    """Load quality distribution from all triplets in a file.

    Returns:
        ASCII representation of quality distribution, e.g., "1:░ 2:░ 3:▓▓▓ 4:▓▓"
    """
    try:
        with open(triplet_file) as f:
            if triplet_file.suffix == ".json":
                data = json.load(f)
                if not data or not isinstance(data, list):
                    return ""
                triplets = data
            else:
                # JSONL format
                triplets = [json.loads(line) for line in f if line.strip()]

        if not triplets:
            return ""

        # Collect quality ratings
        ratings = []
        for triplet in triplets:
            # Try different quality rating locations
            rating = triplet.get("quality_rating")  # Direct field (JSONL format)

            if rating is None:
                # Try nested quality_assessment_without_annotations first (raw quality)
                qa = triplet.get("quality_assessment_without_annotations", {})
                rating = qa.get("rating")

            if rating is None:
                # Fall back to quality_assessment_with_annotations
                qa = triplet.get("quality_assessment_with_annotations", {})
                rating = qa.get("rating")

            if rating is None:
                # Try nested quality_assessment (JSON format)
                qa = triplet.get("quality_assessment", {})
                rating = qa.get("rating")

            if rating is not None:
                ratings.append(rating)

        if not ratings:
            return "-"

        # Calculate distribution
        total = len(ratings)
        counts = {}
        for rating in ratings:
            counts[rating] = counts.get(rating, 0) + 1

        # Create ultra-compact format with counts
        # Format: [7][0][19][10][2] n=38
        parts = []
        for rating in range(1, 6):  # Ratings 1-5
            count = counts.get(rating, 0)
            parts.append(f"[{count}]")

        return "".join(parts) + f" n={total}"

    except Exception as e:
        return f"Error: {e}"


def load_example_triplet(triplet_file: Path) -> dict[str, str] | None:
    """Load best quality triplet from JSON or JSONL file.

    Prefers triplets with quality rating 5 (exceptional), then 4 (ideal),
    then falls back to the first triplet.

    Returns:
        Dict with 'anchor', 'positive', 'negative' keys, or None if error
    """
    try:
        with open(triplet_file) as f:
            if triplet_file.suffix == ".json":
                # JSON format: array of triplets
                data = json.load(f)
                if not data or not isinstance(data, list):
                    return None
                triplets = data
            else:
                # JSONL format: newline-delimited
                triplets = [json.loads(line) for line in f if line.strip()]

        if not triplets:
            return None

        # Helper function to extract quality rating from a triplet
        def get_rating(triplet):
            # Try different quality rating locations
            rating = triplet.get("quality_rating")  # Direct field (JSONL format)

            if rating is None:
                # Try nested quality_assessment_without_annotations first (raw quality)
                qa = triplet.get("quality_assessment_without_annotations", {})
                rating = qa.get("rating")

            if rating is None:
                # Fall back to quality_assessment_with_annotations
                qa = triplet.get("quality_assessment_with_annotations", {})
                rating = qa.get("rating")

            if rating is None:
                # Try nested quality_assessment (JSON format)
                qa = triplet.get("quality_assessment", {})
                rating = qa.get("rating")

            return rating

        # Find best triplet: prefer rating 5, then 4, then first
        best_triplet = None
        best_rating = 0

        for triplet in triplets:
            rating = get_rating(triplet)
            if rating == 5:
                # Found exceptional triplet, use it immediately
                best_triplet = triplet
                break
            elif rating == 4 and best_rating < 4:
                # Found ideal triplet, remember it but keep looking for 5
                best_triplet = triplet
                best_rating = 4
            elif best_triplet is None:
                # No good triplet found yet, save this as fallback
                best_triplet = triplet
                best_rating = rating if rating else 0

        # Use best triplet found (or first if no ratings)
        triplet = best_triplet if best_triplet else triplets[0]

        # Extract fields and ensure they're strings
        anchor = triplet.get("anchor", "")
        positive = triplet.get("positive", "")
        negative = triplet.get("negative", "")

        # Convert to strings if they're not already
        if not isinstance(anchor, str):
            anchor = str(anchor) if anchor else ""
        if not isinstance(positive, str):
            positive = str(positive) if positive else ""
        if not isinstance(negative, str):
            negative = str(negative) if negative else ""

        return {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
        }
    except Exception as e:
        print(f"Error loading {triplet_file}: {e}")
        return None


def load_embedding_similarity(task_name: str) -> str:
    """Load embedding similarity results for a task from method_logs.

    Args:
        task_name: Task name (e.g., "gsm8k__arithmetic")

    Returns:
        Example result: "✓" or "✗" for first triplet, or ""
    """
    outputs_dir = project_root / "outputs"

    # Search for method_logs directories containing this task
    # Note: document rewrite often has suffix patterns like __pre__10
    for method_logs_dir in outputs_dir.rglob(f"method_logs/{task_name}*"):
        if not method_logs_dir.is_dir():
            continue

        # Look for specific embedding method JSONL file
        # Hardcode to qwen3_8b_with_instructions only
        method_file = method_logs_dir / "qwen3_8b_with_instructions.jsonl"

        if not method_file.exists():
            continue

        try:
            # Read JSONL and filter for embedding methods
            first_outcome = None
            outcomes = []

            with open(method_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)

                    # Only process embedding methods
                    if entry.get("method_type") != "embeddings":
                        continue

                    outcome = entry.get("outcome")
                    if outcome is not None:
                        outcomes.append(outcome)
                        # Store first triplet outcome (triplet_idx == 0)
                        if entry.get("triplet_idx") == 0 and first_outcome is None:
                            first_outcome = outcome

            if not outcomes:
                continue

            # Format first example result
            if first_outcome == 1:
                example_result = "✓"
            elif first_outcome == -1:
                example_result = "✗"
            else:
                example_result = "~"  # tie

            return example_result

        except Exception:
            continue

    return ""  # No results found


def load_bm25_results(task_name: str) -> str:
    """Load BM25 results for a task from method_logs.

    Args:
        task_name: Task name (e.g., "gsm8k__arithmetic")

    Returns:
        Example result: "✓" or "✗" for first triplet, or ""
    """
    outputs_dir = project_root / "outputs"

    # Search for method_logs directories containing this task
    for method_logs_dir in outputs_dir.rglob(f"method_logs/{task_name}*"):
        if not method_logs_dir.is_dir():
            continue

        # Look for specific BM25 method JSONL file
        method_file = method_logs_dir / "bm25_lexical.jsonl"

        if not method_file.exists():
            continue

        try:
            # Read JSONL and filter for bm25 methods
            first_outcome = None
            outcomes = []

            with open(method_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)

                    # Only process bm25 methods
                    if entry.get("method_type") != "bm25":
                        continue

                    outcome = entry.get("outcome")
                    if outcome is not None:
                        outcomes.append(outcome)
                        # Store first triplet outcome (triplet_idx == 0)
                        if entry.get("triplet_idx") == 0 and first_outcome is None:
                            first_outcome = outcome

            if not outcomes:
                continue

            # Format first example result
            if first_outcome == 1:
                example_result = "✓"
            elif first_outcome == -1:
                example_result = "✗"
            else:
                example_result = "~"  # tie

            return example_result

        except Exception:
            continue

    return ""  # No results found


def load_document_rewrite_results(task_name: str) -> str:
    """Load document rewrite results for a task from method_logs.

    Args:
        task_name: Task name (e.g., "analogies__analogy_type")

    Returns:
        Example result: "✓" or "✗" for first triplet, or ""
    """
    outputs_dir = project_root / "outputs"

    # Search for method_logs directories containing this task
    # Note: document rewrite often has suffix patterns like __pre__10
    for method_logs_dir in outputs_dir.rglob(f"method_logs/{task_name}*"):
        if not method_logs_dir.is_dir():
            continue

        # Look for specific document rewrite method JSONL file
        # Hardcode to dr_gemini_lite_openai_small
        method_file = method_logs_dir / "dr_gemini_lite_openai_small.jsonl"

        if not method_file.exists():
            continue

        try:
            # Read JSONL and filter for document_rewrite methods
            first_outcome = None
            outcomes = []

            with open(method_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)

                    # Only process document_rewrite methods
                    if entry.get("method_type") != "document_rewrite":
                        continue

                    outcome = entry.get("outcome")
                    if outcome is not None:
                        outcomes.append(outcome)
                        # Store first triplet outcome (triplet_idx == 0)
                        if entry.get("triplet_idx") == 0 and first_outcome is None:
                            first_outcome = outcome

            if not outcomes:
                continue

            # Format first example result
            if first_outcome == 1:
                example_result = "✓"
            elif first_outcome == -1:
                example_result = "✗"
            else:
                example_result = "~"  # tie

            return example_result

        except Exception:
            continue

    return ""  # No results found


def generate_task_report(task_name: str, method_logs_dir: Path) -> Path | None:
    """Generate a per-task markdown report from method logs.

    Args:
        task_name: Task name (e.g., "gsm8k__arithmetic")
        method_logs_dir: Path to the method_logs/task_name directory

    Returns:
        Path to generated report, or None if error
    """
    try:
        # Collect results from all method JSONL files
        method_results = {}

        for jsonl_file in method_logs_dir.glob("*.jsonl"):
            method_name = jsonl_file.stem
            outcomes = []

            with open(jsonl_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    outcome = entry.get("outcome")
                    if outcome is not None:
                        outcomes.append(outcome)

            if outcomes:
                correct = sum(1 for o in outcomes if o == 1)
                incorrect = sum(1 for o in outcomes if o == -1)
                ties = sum(1 for o in outcomes if o == 0)
                total = len(outcomes)

                # For LM judge triplet methods (bidirectional evaluation),
                # each triplet is evaluated twice, so divide by 2
                if "triplet" in method_name:
                    correct = correct / 2
                    incorrect = incorrect / 2
                    ties = ties / 2
                    total = total / 2

                accuracy = (correct / total * 100) if total > 0 else 0

                method_results[method_name] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "incorrect": incorrect,
                    "ties": ties,
                    "total": total,
                }

        if not method_results:
            return None

        # Create task_reports directory inside results
        run_dir = (
            method_logs_dir.parent.parent
        )  # Go up from method_logs/task_name to run dir
        task_reports_dir = run_dir / "results" / "task_reports"
        task_reports_dir.mkdir(parents=True, exist_ok=True)

        # Use the actual directory name for links (may have suffixes like __tag__5)
        actual_task_dir_name = method_logs_dir.name

        # Generate markdown report
        report_lines = [
            f"# {task_name}",
            "",
            "## Evaluation Results",
            "",
            "| Method | Accuracy | Correct | Incorrect | Ties | Total | Logs |",
            "|--------|----------|---------|-----------|------|-------|------|",
        ]

        # Sort by accuracy descending
        sorted_methods = sorted(
            method_results.items(), key=lambda x: x[1]["accuracy"], reverse=True
        )

        for method_name, results in sorted_methods:
            log_link = (
                f"[log](../../method_logs/{actual_task_dir_name}/{method_name}.jsonl)"
            )

            # Format counts as integers if whole numbers, otherwise as floats
            def format_count(val):
                return (
                    int(val)
                    if isinstance(val, int | float) and val % 1 == 0
                    else f"{val:.1f}"
                )

            report_lines.append(
                f"| {method_name} | {results['accuracy']:.2f}% | "
                f"{format_count(results['correct'])} | {format_count(results['incorrect'])} | "
                f"{format_count(results['ties'])} | {format_count(results['total'])} | {log_link} |"
            )

        # Write report
        report_path = task_reports_dir / f"{task_name}.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        return report_path

    except Exception as e:
        print(f"Error generating report for {task_name}: {e}")
        return None


def find_results_link(task_name: str) -> str:
    """Find or generate evaluation results for a task and return a markdown link.

    Args:
        task_name: Task name (e.g., "gsm8k__arithmetic")

    Returns:
        Markdown link to results, or empty string if no results found
    """
    outputs_dir = project_root / "outputs"

    # Search for method_logs directories containing this task
    for method_logs_dir in outputs_dir.rglob(f"method_logs/{task_name}*"):
        if not method_logs_dir.is_dir():
            continue

        # Check if any evaluation files exist
        if not list(method_logs_dir.glob("*.jsonl")):
            continue

        # Generate per-task report
        report_path = generate_task_report(task_name, method_logs_dir)

        if report_path:
            # Create relative path from project root
            rel_path = report_path.relative_to(project_root)
            return f"[Results]({rel_path})"
        else:
            # Fallback to method_logs directory if report generation fails
            rel_path = method_logs_dir.relative_to(project_root)
            return f"[Results]({rel_path})"

    return ""  # No results found


def generate_table() -> str:
    """Generate markdown table of all tasks, organized into three sections."""
    criteria_data = load_criteria()
    triplet_files = find_triplet_files()

    rows = []
    total_with_triplets = 0
    total_prelabeled = 0
    total_generated = 0

    for docset_name, criteria_dict in criteria_data.items():
        if not isinstance(criteria_dict, dict):
            continue

        for criterion_name, criterion_metadata in criteria_dict.items():
            if not isinstance(criterion_metadata, dict):
                continue

            # Build task name using registry key (not YAML key)
            # Map YAML key (e.g., "crossword") to registry key (e.g., "crossword_clues")
            docset_registry_key = map_docset_key(docset_name)
            if docset_registry_key is None:
                docset_registry_key = docset_name  # Fallback to YAML key if no mapping

            task_name = f"{docset_registry_key}__{criterion_name}"

            # Extract metadata
            description = criterion_metadata.get("description", "No description")
            criterion_source = criterion_metadata.get("source", "new")
            criterion_display = f"**{criterion_name}**<br/>{description}"

            # Try to find existing triplet
            triplet_file = triplet_files.get(task_name)
            emb_example = load_embedding_similarity(
                task_name
            )  # Load embedding similarity
            bm25_example = load_bm25_results(task_name)  # Load BM25 results
            dr_example = load_document_rewrite_results(
                task_name
            )  # Load document rewrite results
            results_link = find_results_link(task_name)  # Find results link

            # Generate triplet link if triplet exists
            triplet_link = ""
            if triplet_file:
                rel_triplet_path = triplet_file.relative_to(project_root)
                triplet_link = f"[Triplets]({rel_triplet_path})"

            is_prelabeled = False
            if triplet_file:
                example = load_example_triplet(triplet_file)
                quality_dist = load_quality_distribution(triplet_file)

                if example:
                    anchor_display = truncate_text(example["anchor"])
                    pos_display = truncate_text(example["positive"])
                    neg_display = truncate_text(example["negative"])

                    # Check if this is prelabeled data
                    parent_name = triplet_file.parent.name
                    is_prelabeled = (
                        "prelabeled" in str(triplet_file)
                        or "__pre__" in parent_name
                        or "__tag__" in parent_name
                        or "__class__" in parent_name
                    )
                    label = "**[PRELABELED]** " if is_prelabeled else ""
                    source = f"{label}From {triplet_file.parent.parent.parent.name}"
                    total_with_triplets += 1
                    if is_prelabeled:
                        total_prelabeled += 1
                    else:
                        total_generated += 1
                else:
                    anchor_display = pos_display = neg_display = "Error loading triplet"
                    source = "File exists but couldn't parse"
                    quality_dist = ""
            else:
                # Check for triplet_example_hint in metadata
                triplet_hint = criterion_metadata.get("triplet_example_hint", {})
                if triplet_hint and isinstance(triplet_hint, dict):
                    anchor = triplet_hint.get("anchor", "")
                    pos = triplet_hint.get("pos", "")
                    neg = triplet_hint.get("neg", "")

                    anchor_display = truncate_text(anchor)
                    pos_display = truncate_text(pos)
                    neg_display = truncate_text(neg)
                    source = "From triplet_example_hint"
                    quality_dist = ""
                else:
                    anchor_display = pos_display = neg_display = ""
                    source = "No triplets generated yet"
                    quality_dist = ""

            rows.append(
                {
                    "docset": docset_name,
                    "criterion": criterion_display,
                    "criterion_source": criterion_source,
                    "anchor": anchor_display,
                    "pos": pos_display,
                    "neg": neg_display,
                    "source": source,
                    "quality": quality_dist,
                    "emb_example": emb_example,
                    "bm25_example": bm25_example,
                    "dr_example": dr_example,
                    "results_link": results_link,
                    "triplet_link": triplet_link,
                    "is_prelabeled": is_prelabeled,
                }
            )

    # Categorize rows into three sections
    # Use criterion_source metadata to determine categorization
    generated_triplets = []  # New tasks with generated triplets (source="new")
    existing_adapted = []  # Pre-existing or adapted data (source="pre-existing" or "adapted")
    tasks_under_development = []  # No triplets yet OR moved from generated

    # Define the 6 tasks that should stay in "Generated triplets"
    GENERATED_WHITELIST = {
        ("gsm8k", "arithmetic"),
        ("gsm8k", "problem_type"),
        ("crossword", "topic"),
        ("crossword", "clue_type"),
        ("crossword", "type_of_challenge"),
        ("haiku", "imagery"),
        ("haiku", "meaning_evoked"),
        ("haiku", "poem_composition"),
    }

    for row in rows:
        docset = row["docset"]
        # Extract criterion name from criterion display (strip markdown and description)
        criterion_raw = row["criterion"]
        criterion_name = criterion_raw.split("**")[1] if "**" in criterion_raw else ""

        # Check if this task is in the whitelist
        is_whitelisted = (docset, criterion_name) in GENERATED_WHITELIST

        # No triplets yet → Tasks under development
        if (
            row["source"] == "No triplets generated yet"
            or row["source"] == "From triplet_example_hint"
        ):
            tasks_under_development.append(row)
        # Has triplets and criterion_source is "new" → Check whitelist
        elif row["criterion_source"] == "new" and row["anchor"]:
            if is_whitelisted:
                generated_triplets.append(row)
            else:
                # Move to tasks under development
                tasks_under_development.append(row)
        # Has triplets and criterion_source is "pre-existing" or "adapted" → Existing/adapted
        elif row["criterion_source"] in ["pre-existing", "adapted"] and row["anchor"]:
            existing_adapted.append(row)
        # Fallback: if it has triplets but source is unclear, check if prelabeled
        elif row["anchor"] and row["pos"] and row["neg"]:
            if row["is_prelabeled"]:
                existing_adapted.append(row)
            else:
                if is_whitelisted:
                    generated_triplets.append(row)
                else:
                    tasks_under_development.append(row)
        else:
            tasks_under_development.append(row)

    # Sort each section by docset name, then criterion
    for section in [generated_triplets, existing_adapted, tasks_under_development]:
        section.sort(key=lambda r: (r["docset"], r["criterion"]))

    # Helper function to escape pipe characters and newlines
    def escape(s: str) -> str:
        # Normalize line endings (handle \r\n, \r, \n)
        result = s.replace("\r\n", "\n").replace("\r", "\n")
        # Replace newlines with <br/>
        result = result.replace("\n", "<br/>")
        # Collapse multiple consecutive <br/> into single space + <br/>
        result = re.sub(r"(<br/>){2,}", " <br/>", result)
        # Escape pipe characters
        result = result.replace("|", "\\|")
        return result

    # Helper function to generate table rows
    def generate_rows(section_rows, include_source=True):
        table_rows = []
        for row in section_rows:
            if include_source:
                table_rows.append(
                    f"| {escape(row['docset'])} "
                    f"| {escape(row['criterion'])} "
                    f"| {escape(row['anchor'])} "
                    f"| {escape(row['pos'])} "
                    f"| {escape(row['neg'])} "
                    f"| {escape(row['source'])} "
                    f"| {escape(row['quality'])} "
                    f"| {escape(row['bm25_example'])} "
                    f"| {escape(row['emb_example'])} "
                    f"| {escape(row['dr_example'])} "
                    f"| {escape(row['results_link'])} "
                    f"| {escape(row['triplet_link'])} |"
                )
            else:
                table_rows.append(
                    f"| {escape(row['docset'])} "
                    f"| {escape(row['criterion'])} "
                    f"| {escape(row['anchor'])} "
                    f"| {escape(row['pos'])} "
                    f"| {escape(row['neg'])} "
                    f"| {escape(row['quality'])} "
                    f"| {escape(row['bm25_example'])} "
                    f"| {escape(row['emb_example'])} "
                    f"| {escape(row['dr_example'])} "
                    f"| {escape(row['results_link'])} "
                    f"| {escape(row['triplet_link'])} |"
                )
        return table_rows

    # Generate markdown with three sections (reordered)
    md_lines = [
        "# Available Tasks",
        "",
        f"Total tasks: {len(rows)}",
        f"Tasks with triplets: {total_with_triplets}",
        f"  - Generated: {len(generated_triplets)}",
        f"  - Existing/Adapted: {len(existing_adapted)}",
        f"Tasks with example hints: {sum(1 for r in rows if r['source'] == 'From triplet_example_hint')}",
        f"Tasks without triplets: {len(rows) - total_with_triplets}",
        "",
        "Quality distribution: `[r1][r2][r3][r4][r5] n=total` where r1=invalid, r2=ambiguous, r3=trivial, r4=ideal, r5=exceptional",
        "",
        "Evaluation methods:",
        "- **BM25?**: ✓/✗ for example triplet using BM25 lexical search (bm25_lexical)",
        "- **Emb?**: ✓/✗ for example triplet using embedding similarity (qwen3_8b_with_instructions)",
        "- **DR?**: ✓/✗ for example triplet using document rewrite (dr_gemini_lite_openai_small)",
        "- **Results**: Link to per-task evaluation report",
        "- **Triplets**: Link to triplet data file",
        "",
    ]

    # Section 1: Generated triplets (NEW - first section)
    md_lines.extend(
        [
            "## Generated triplets",
            "",
            f"Tasks with generated triplets (source=new): {len(generated_triplets)}",
            "",
            "| Document Set | Criterion | Anchor | Positive | Negative | Quality Distribution | BM25? | Emb? | DR? | Results | Triplets |",
            "|--------------|-----------|--------|----------|----------|----------------------|-------|------|-----|---------|----------|",
        ]
    )
    md_lines.extend(generate_rows(generated_triplets, include_source=False))
    md_lines.append("")

    # Section 2: Existing or adapted data
    md_lines.extend(
        [
            "## Existing or adapted data",
            "",
            f"Tasks with existing/adapted triplets (source=pre-existing or adapted): {len(existing_adapted)}",
            "",
            "| Document Set | Criterion | Anchor | Positive | Negative | Source | Quality Distribution | BM25? | Emb? | DR? | Results | Triplets |",
            "|--------------|-----------|--------|----------|----------|--------|----------------------|-------|------|-----|---------|----------|",
        ]
    )
    md_lines.extend(generate_rows(existing_adapted, include_source=True))
    md_lines.append("")

    # Section 3: Tasks under development
    md_lines.extend(
        [
            "## Tasks under development",
            "",
            f"Tasks without triplets yet (plus tasks moved from generated triplets): {len(tasks_under_development)}",
            "",
            "| Document Set | Criterion | Anchor | Positive | Negative | Quality Distribution | BM25? | Emb? | DR? | Results | Triplets |",
            "|--------------|-----------|--------|----------|----------|----------------------|-------|------|-----|---------|----------|",
        ]
    )
    md_lines.extend(generate_rows(tasks_under_development, include_source=False))

    return "\n".join(md_lines)


def main():
    """Generate and save the task table."""
    print("Generating task table...")
    table = generate_table()

    # Save to file
    output_path = project_root / "TASKS_TABLE.md"
    with open(output_path, "w") as f:
        f.write(table)

    print(f"\nTable saved to: {output_path}")
    print("\nPreview (first 15 lines):")
    print("\n".join(table.split("\n")[:15]))


if __name__ == "__main__":
    main()
