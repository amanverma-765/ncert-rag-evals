"""Full-text search over the FTS5 index.

Textbook questions are dense with exact terminology (phenolphthalein,
seminiferous, SN2), which BM25 matches better than any embedding.
"""

import re
import sqlite3

from ncert_rag.retrieve.base import BaseRetriever

_WORD = re.compile(r"[A-Za-z0-9]+")

# Exercise lists are wall-to-wall question words, so leaving these in ranks
# "What is X?" against every exercise block in the corpus instead of against
# the passage defining X. FTS5 ships no stopword list of its own.
_STOP_WORDS = (
    "a an the is are was were be been being am do does did doing have has had "
    "of in on at to for from by with about into over under and or but if then "
    "than that this these those it its as we you your they them their he she "
    "his her i what why how when where which who whom whose can could should "
    "would will shall may might must not no nor so such only own same too very "
    "give given write explain define describe name state list mention discuss "
    "following each other any all some many much more most"
)
_STOP = frozenset(_STOP_WORDS.split())


def fts_query(question: str) -> str:
    """Rewrite free text as an FTS5 OR query.

    Quoting each term keeps punctuation from being read as query syntax, and OR
    beats FTS5's implicit AND here: a question rarely shares every term with the
    passage that answers it, and bm25 ranks the overlap anyway.
    """
    terms = [
        w for w in _WORD.findall(question) if len(w) > 1 and w.lower() not in _STOP
    ]
    return " OR ".join(f'"{term}"' for term in terms)


class Bm25(BaseRetriever):
    name = "bm25"

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def search(self, question: str, k: int) -> list[tuple[int, float]]:
        query = fts_query(question)
        if not query:
            return []
        rows = self.conn.execute(
            "SELECT f.rowid AS id, bm25(chunks_fts) AS score "
            "FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
            "WHERE chunks_fts MATCH ? AND c.source = 'parsed' "
            "ORDER BY score LIMIT ?",
            (query, k),
        ).fetchall()
        # bm25() is negative with better matches more negative; flip it so every
        # arm reports higher-is-better
        return [(row["id"], -row["score"]) for row in rows]
