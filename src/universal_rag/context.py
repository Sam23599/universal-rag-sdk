"""Evidence deduplication and context budgeting."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import SearchHit
from .prompting import GroundedPromptBuilder


class ContextBuilder:
    def __init__(
        self, prompt_builder: GroundedPromptBuilder | None = None, *, duplicate_similarity: float = 0.9
    ) -> None:
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()
        self.duplicate_similarity = duplicate_similarity

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.casefold()))

    def _similar(self, left: str, right: str) -> float:
        left_tokens, right_tokens = self._tokens(left), self._tokens(right)
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 1.0

    def build(
        self,
        evidence: Sequence[SearchHit],
        *,
        max_characters: int,
    ) -> tuple[str, dict[str, SearchHit], tuple[SearchHit, ...]]:
        selected: list[SearchHit] = []
        seen_ids: set[str] = set()
        for hit in sorted(evidence, key=lambda value: -value.score):
            if hit.document.id in seen_ids:
                continue
            if any(
                self._similar(hit.document.content, existing.document.content) >= self.duplicate_similarity
                for existing in selected
            ):
                continue
            selected.append(hit)
            seen_ids.add(hit.document.id)
        context, mapping = self.prompt_builder.evidence_context(selected, max_characters=max_characters)
        included = tuple(mapping[citation_id] for citation_id in mapping)
        return context, mapping, included
