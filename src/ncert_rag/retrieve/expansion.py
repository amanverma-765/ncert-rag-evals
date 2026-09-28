"""Rewrite the question into textbook vocabulary, then search with it.

Students and textbooks use different words for the same idea. Asked "why do we
feel tired after running fast", BM25 returns Computer Science chapters, since
"running" and "fast" are the only terms it can match; the book says "anaerobic
respiration" and "lactic acid".

The rewrite does not care what searches afterwards, so BASES lets it feed BM25,
the embeddings, or both.
"""

import sqlite3
from collections.abc import Callable

from ncert_rag.retrieve.base import BaseRetriever, Retriever
from ncert_rag.retrieve.bm25 import Bm25
from ncert_rag.retrieve.hybrid import Hybrid
from ncert_rag.retrieve.vector import Vector

# Terms only. Having the model draft a hypothetical answer passage retrieves
# better but invents textbook prose that was never in the books, which a study
# tool should not put in front of retrieval. See EVALUATION.md.
_INSTRUCTIONS = (
    "You rewrite a student's question into the vocabulary an NCERT science or "
    "mathematics textbook would use for the same idea. Reply with search terms "
    "only: the technical nouns and phrases likely to appear in the relevant "
    "passage, separated by spaces. Keep any term the student already used. No "
    "punctuation, no explanation, at most 25 words."
)

# Fixed across every expansion arm; swapping it would confound them all at once.
QUERY_REWRITER = "ag/gemini-3.7-flash-medium"

# What the rewritten query is handed to.
BASES: dict[str, Callable[[sqlite3.Connection], Retriever]] = {
    "bm25": Bm25,
    "vector": Vector,
    "raw": lambda conn: Vector(conn, source="raw"),
    "hybrid": Hybrid,
}

# Shared across arms: they ask the same model the same questions and differ
# only in what they search with, so one rewrite covers all of them.
_CACHE: dict[str, str] = {}


def clear_cache() -> None:
    """Drop cached rewrites so the next query pays the model call again.

    The cost benchmark calls this between arms; without it every arm after the
    first measures a cache hit instead of its real latency.
    """
    _CACHE.clear()


class Expansion(BaseRetriever):
    def __init__(self, conn: sqlite3.Connection, base: str = "bm25"):
        self.conn = conn
        self.name = f"expansion_{base}"
        self.model = QUERY_REWRITER
        self.base = BASES[base](conn)
        # imported lazily so building the corpus needs no model credentials
        from ncert_rag.services import llm

        self.agent = llm.agent(_INSTRUCTIONS, model=self.model)
        self._llm = llm

    def expand(self, question: str) -> str:
        """Question plus its textbook-vocabulary restatement.

        The original terms stay in the query, so a drifting rewrite has the
        student's own words to fall back on. That bounds the damage; it does
        not rule it out. Retrieval returns the top k by score, so added terms
        reshuffle every rank and a chunk sitting at k can be pushed past it.
        Measured over 282 questions, the rewrite wins 21 and loses 3.
        """
        if question not in _CACHE:
            _CACHE[question] = self._llm.ask(self.agent, question)
        return f"{question} {_CACHE[question]}"

    def search(self, question: str, k: int) -> list[tuple[int, float]]:
        return self.base.search(self.expand(question), k)
