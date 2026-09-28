"""Scoring. No I/O, so it can be tested on hand-written rankings.

Grading is at chapter level: an exercise question is labelled only with its
chapter, and raw windows have no section to compare against.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Label = tuple[str, int]  # (book slug, chapter number)

KS = (1, 5, 10)

# Two-sided 95%, for the interval on a paired difference in `paired_margin`.
#
# There is deliberately no per-arm interval any more. An independent-sample
# margin printed beside eight rows invites exactly the comparison it cannot
# support, needed a paragraph of table footnote telling readers not to make it,
# and this eval published a wrong conclusion by making it anyway.
CONFIDENCE = 1.96


def first_hit(ranked: Sequence[Label], gold: Label) -> int | None:
    """1-based rank of the first chunk from the gold chapter, if any."""
    return next(
        (pos for pos, label in enumerate(ranked, start=1) if label == gold),
        None,
    )


@dataclass(frozen=True, slots=True)
class Score:
    n: int
    recall: dict[int, float]
    mrr: float

    def row(self) -> str:
        cells = " ".join(f"{self.recall[k]:>7.1%}" for k in KS)
        return f"{cells} {self.mrr:>7.3f} {self.n:>6}"


def summarize(ranks: Iterable[int | None]) -> Score:
    ranks = list(ranks)
    if not ranks:
        return Score(0, dict.fromkeys(KS, 0.0), 0.0)

    return Score(
        n=len(ranks),
        recall={
            k: sum(1 for r in ranks if r is not None and r <= k) / len(ranks)
            for k in KS
        },
        # truncated at the deepest cutoff, so MRR and recall@10 agree about
        # what counts as found
        mrr=sum(1.0 / r for r in ranks if r is not None and r <= max(KS)) / len(ranks),
    )


def hits_at(ranks: Iterable[int | None], k: int = 5) -> list[bool]:
    """Per-question hit/miss, in question order, for a paired comparison."""
    return [rank is not None and rank <= k for rank in ranks]


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Discordant counts (a-only, b-only) and a two-sided p on the difference.

    Every arm answers the same questions, so `Score.margin` is the wrong test
    for comparing two of them: an independent-sample interval throws the
    pairing away and calls real differences noise. Two arms agreeing on 250
    questions and splitting 30 carry the evidence in those 30, not in all 282.

    Normal approximation with a continuity correction, which is close enough at
    the discordant counts this eval produces (20-65 questions).
    """
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    discordant = only_a + only_b
    if not discordant:
        return 0, 0, 1.0
    z = max(abs(only_b - only_a) - 1, 0) / math.sqrt(discordant)
    return only_a, only_b, math.erfc(z / math.sqrt(2))


def paired_margin(only_a: int, only_b: int, n: int) -> float:
    """Half-width of the 95% interval on the paired R@5 difference, in points.

    `mcnemar` answers "can this question set tell the two arms apart"; this
    answers "how large a difference could it still be hiding". The pair must be
    read together. A high p with a wide interval means *we cannot tell*, which
    is not the same claim as *the arms are equal* -- and the difference matters
    most exactly when a null result is being used to argue two arms equivalent,
    as it is for the parser. 12 discordant questions against 11 is p=1.0 by the
    continuity correction whatever the sample size, so p alone says nothing
    there; the interval is what carries the information.
    """
    if n <= 0:
        return 0.0
    difference = (only_b - only_a) / n
    variance = max((only_a + only_b) / n**2 - difference**2 / n, 0.0)
    return CONFIDENCE * math.sqrt(variance) * 100


HEADER = f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'n':>6}"
