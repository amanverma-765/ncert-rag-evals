"""BM25 and dense retrieval fused by reciprocal rank."""

import sqlite3

from ncert_rag.retrieve.base import BaseRetriever, rrf
from ncert_rag.retrieve.bm25 import Bm25
from ncert_rag.retrieve.vector import Vector

# Fuse deeper than we return, so a chunk ranked 15th by one method and 3rd by
# the other can still surface.
_DEPTH = 30


class Hybrid(BaseRetriever):
    name = "hybrid"

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.lexical = Bm25(conn)
        self.dense = Vector(conn)

    def search(self, question: str, k: int) -> list[tuple[int, float]]:
        rankings = [
            [cid for cid, _ in self.lexical.search(question, _DEPTH)],
            [cid for cid, _ in self.dense.search(question, _DEPTH)],
        ]
        return rrf(rankings, k)
