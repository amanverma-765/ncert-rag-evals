"""Dense retrieval over one chunk set.

`vector` searches the section-aware chunks and `vector_raw` the fixed windows.
Same model and same cleaned text, so the gap between them measures chunking.
"""

import sqlite3

from ncert_rag.core.models import ChunkSource
from ncert_rag.retrieve.base import BaseRetriever
from ncert_rag.services import embedder
from ncert_rag.store import vectors


class Vector(BaseRetriever):
    def __init__(self, conn: sqlite3.Connection, source: ChunkSource = "parsed"):
        self.conn = conn
        self.source = source
        self.name = "vector" if source == "parsed" else "vector_raw"

    def search(self, question: str, k: int) -> list[tuple[int, float]]:
        return vectors.top_k(self.source, embedder.encode_query(question), k)
