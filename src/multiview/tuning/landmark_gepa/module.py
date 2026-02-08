"""DSPy signature, module, and wrapper for the landmark query-generation workflow."""

from __future__ import annotations

import json
import logging
import re

import dspy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSPy signature & module
# ---------------------------------------------------------------------------


class QueryGenerationSignature(dspy.Signature):
    """Generate search queries that map out the semantic space of a criterion."""

    criteria: str = dspy.InputField(
        desc="User criteria for what they're looking for (e.g., 'religious symbolism')"
    )
    domain: str = dspy.InputField(
        desc="Domain context (e.g., 'fine art museum collection', 'cartoons')"
    )
    num_queries: int = dspy.InputField(desc="Number of queries to generate")
    queries: str = dspy.OutputField(
        desc="JSON array of query strings that cover different facets"
    )


class QueryGeneratorModule(dspy.Module):
    """DSPy module for generating queries via ``ChainOfThought``."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(QueryGenerationSignature)

    def forward(self, criteria: str, domain: str, num_queries: int) -> dspy.Prediction:
        return self.generate(criteria=criteria, domain=domain, num_queries=num_queries)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_queries(queries_json: str) -> list[str]:
    """Parse a list of query strings from JSON / markdown / raw text."""
    text = queries_json.strip()

    # Strip markdown code fences
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()

    # Extract JSON array boundaries
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]

    # Try JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            queries: list[str] = []
            for item in data:
                if isinstance(item, str):
                    queries.append(item.strip())
                elif isinstance(item, dict):
                    query = item.get("query", "").strip()
                    if query:
                        queries.append(query)
            return queries
    except json.JSONDecodeError:
        pass

    # Fallback: quoted strings
    matches = re.findall(r'"([^"]+)"', text)
    if matches:
        return [m.strip() for m in matches if len(m.strip()) > 10]

    # Last resort: line split
    lines = text.strip().split("\n")
    queries = []
    for line in lines:
        line = line.strip()
        if not line or len(line) <= 10 or line.startswith(("{", "[")):
            continue
        if line[0].isdigit() and "." in line[:5]:
            line = line.split(".", 1)[1].strip()
        elif line.startswith(("-", "*")):
            line = line[1:].strip()
        queries.append(line)
    return queries


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


class DSPyQueryGenerator:
    """Drop-in replacement for ``LLMQueryGenerator`` backed by a DSPy module.

    Unlike the source implementation, this version does **not** configure DSPy
    internally – the caller is expected to set up ``dspy.configure(lm=...)``
    before instantiation.
    """

    def __init__(
        self,
        optimized_prompt_path: str | None = None,
    ):
        self.module = QueryGeneratorModule()
        if optimized_prompt_path:
            self.load_optimized_prompt(optimized_prompt_path)

    # -- public interface --------------------------------------------------

    def generate_queries(
        self,
        criteria: str,
        num_queries: int = 50,
        domain: str = "general",
        verbose: bool = True,
    ) -> list[str]:
        if verbose:
            logger.info(
                "Generating %d queries for criteria='%s' domain='%s'",
                num_queries,
                criteria,
                domain,
            )

        result = self.module(criteria=criteria, domain=domain, num_queries=num_queries)
        queries = parse_queries(result.queries)
        queries = self._postprocess_queries(queries, num_queries)

        if verbose:
            logger.info("Generated %d queries", len(queries))
        return queries

    # -- persistence -------------------------------------------------------

    def load_optimized_prompt(self, path: str) -> None:
        self.module.load(path)
        logger.info("Loaded optimized prompt from: %s", path)

    def save_prompt(self, path: str) -> None:
        self.module.save(path)
        logger.info("Saved prompt to: %s", path)

    def get_module(self) -> QueryGeneratorModule:
        return self.module

    # -- internal ----------------------------------------------------------

    def _postprocess_queries(self, queries: list[str], limit: int) -> list[str]:
        filtered: list[str] = []
        seen: set[str] = set()
        banned = {
            "artwork",
            "art piece",
            "piece of art",
            "important artwork",
            "famous artwork",
        }
        for query in queries:
            norm = self._normalize(query)
            if not norm or norm in seen:
                continue
            if any(tok in norm for tok in banned):
                continue
            if len(norm.split()) < 3:
                continue
            seen.add(norm)
            filtered.append(query.strip())
            if len(filtered) >= limit:
                break
        if len(filtered) < limit:
            logger.warning(
                "Only %d unique queries after filtering (requested %d)",
                len(filtered),
                limit,
            )
        return filtered

    @staticmethod
    def _normalize(query: str) -> str:
        text = query.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()
