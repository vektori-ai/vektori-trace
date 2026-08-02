"""The teacher interface, in one place.

There were three overlapping Protocols before this module: `distill.IdScoringPool`
(`score_ids`), `distill.TopKScoringPool` (`score_ids_topk`), and a *second*
`IdScoringPool` in `cross` (`score_ids` + `provenance`) — same name, different
shape, which made "does this pool satisfy IdScoringPool" a question you could
not answer without knowing which import you were looking at.

These are Protocols rather than an ABC on purpose: the four pools are
`@dataclass` classes that inherit nothing, and structural typing keeps it that
way. Nothing has to be edited for a new backend to qualify.

Note that no type checker runs in CI, so these annotations are documentation.
The enforcement lives in `tests/test_teacher_protocol.py`, which asserts every
pool really does implement `Teacher` with matching signatures. If you add a
backend, that test is what will fail.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IdScoringPool(Protocol):
    """A teacher that can score ids the caller supplies."""

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]: ...


@runtime_checkable
class TopKScoringPool(Protocol):
    """A teacher that also returns its top-K at each scored position."""

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]: ...


@runtime_checkable
class Teacher(Protocol):
    """Everything a teacher pool offers.

    The full surface `CrossTokenizerTeacherPool` needs from the pool it wraps —
    it calls all four. `VllmTeacherPool`, `BedrockTeacherPool`,
    `FireworksTeacherPool`, `CrossTokenizerTeacherPool`, and
    `InMemoryIdScoringPool` all satisfy this.

    `prompt_logprobs` is deliberately absent: the direct-endpoint pools have it,
    but `CrossTokenizerTeacherPool` cannot, since the prompt it would score is
    in the student's tokenisation rather than the teacher's.
    """

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]: ...

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]: ...

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str: ...

    def provenance(self) -> dict[str, Any]: ...
