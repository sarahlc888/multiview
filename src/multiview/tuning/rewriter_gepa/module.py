"""DSPy signature and module for the rewriter GEPA workflow."""

from __future__ import annotations

import logging

import dspy
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class RewriteQuery(dspy.Signature):
    """Rewrite *input_text* to surface the salient aspect described by *aspect*."""

    input_text = dspy.InputField()
    aspect = dspy.InputField()
    aspect_of_input_text = dspy.OutputField()


class CheckTripletSimple(dspy.Module):
    """Determine whether A is closer to B or C after criteria-driven rewriting.

    A single learned prompt is applied independently to A, B, and C.  The
    rewritten texts are then embedded and compared via cosine distance.

    Args:
        embedder: A ``dspy.Embedder`` (or any callable that maps a list of
            strings to a list of vectors).
        fallback_criteria: Default criteria string when none is provided per-call.
    """

    def __init__(
        self,
        embedder: dspy.Embedder,
        fallback_criteria: str | None = None,
    ):
        self.embedder = embedder
        self.fallback_criteria = fallback_criteria
        self.rewrite_query = dspy.ChainOfThought(RewriteQuery)

    def _rewrite(self, text: str, criteria: str) -> str:
        out = self.rewrite_query(input_text=text, aspect=criteria)
        return out.aspect_of_input_text

    def forward(
        self,
        A: str,
        B: str,
        C: str,
        criteria: str | None = None,
    ) -> dspy.Prediction:
        crit = (
            criteria
            if criteria is not None
            else (self.fallback_criteria if self.fallback_criteria is not None else "")
        )

        A_rewritten = self._rewrite(A, crit)
        B_rewritten = self._rewrite(B, crit)
        C_rewritten = self._rewrite(C, crit)

        embeds = self.embedder([A_rewritten, B_rewritten, C_rewritten])
        A_embed, B_embed, C_embed = embeds[0], embeds[1], embeds[2]

        cos_AB = cosine(A_embed, B_embed)  # cosine *distance* (lower = closer)
        cos_AC = cosine(A_embed, C_embed)

        logger.debug("criteria=%s", crit)
        logger.debug("A=%s -> %s", A, A_rewritten)
        logger.debug("B=%s -> %s", B, B_rewritten)
        logger.debug("C=%s -> %s", C, C_rewritten)
        logger.debug("cos_AB=%.6f  cos_AC=%.6f", cos_AB, cos_AC)

        # 1 if C is closer (cos_AB > cos_AC), 0 if B is closer
        closer = int(cos_AB > cos_AC)

        return dspy.Prediction(
            closer=closer,
            A_rewritten=A_rewritten,
            B_rewritten=B_rewritten,
            C_rewritten=C_rewritten,
            cos_AB=cos_AB,
            cos_AC=cos_AC,
        )
