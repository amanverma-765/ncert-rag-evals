"""The retriever interface and the rank fusion the arms share."""

import sqlite3
from collections.abc import Sequence
from typing import Protocol

from ncert_rag.core.models import RetrievalHit
from ncert_rag.store import db

# Reciprocal rank fusion's damping constant. 60 is the value from the original
# paper and nothing here is tuned against it.
RRF_K = 60


class Retriever(Protocol):
    """What every arm implements.

    `search` is part of the contract, not an implementation detail: Hybrid
    fuses two arms' ranked ids and Expansion puts a rewrite in front of one, so
    both compose on ids and scores rather than on hydrated hits.
    """

    name: str

    def search(self, question: str, k: int) -> list[tuple[int, float]]: ...

    def retrieve(self, question: str, k: int) -> list[RetrievalHit]: ...


class BaseRetriever:
    conn: sqlite3.Connection

    def search(self, question: str, k: int) -> list[tuple[int, float]]:
        raise NotImplementedError

    def retrieve(self, question: str, k: int) -> list[RetrievalHit]:
        return db.hits(self.conn, self.search(question, k))


def rrf(rankings: Sequence[Sequence[int]], k: int) -> list[tuple[int, float]]:
    """Fuse ranked id lists by reciprocal rank.

    Only rank position carries over. BM25 scores and cosine similarities are
    not on a comparable scale, so the scores themselves cannot be combined.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + position)
    ordered = sorted(scores.items(), key=lambda pair: -pair[1])
    return ordered[:k]
