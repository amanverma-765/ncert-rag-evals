"""One SQLite file holds the corpus text and its full-text index.

Vectors live in Chroma (`store/vectors.py`), not here, so there is one store
that can be behind rather than two that can disagree.

Both chunk sets live in the same table under different `source` values so
every arm searches identical text and any difference between them comes from
chunking strategy alone.
"""

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from ncert_rag.core.models import BookSpec, Chunk, RetrievalHit
from ncert_rag.core.paths import DB_PATH
from ncert_rag.store import vectors

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    slug    TEXT PRIMARY KEY,
    code    TEXT NOT NULL,
    klass   INTEGER NOT NULL,
    subject TEXT NOT NULL,
    tier    TEXT NOT NULL,
    digest  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    book    TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    section TEXT,
    page    INTEGER NOT NULL,
    source  TEXT NOT NULL,
    text    TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text);
CREATE TABLE IF NOT EXISTS exercises (
    id       INTEGER PRIMARY KEY,
    book     TEXT NOT NULL,
    chapter  INTEGER NOT NULL,
    question TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_book_source ON chunks(book, source);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    """Open the corpus.

    The default same-thread guard stays on. One connection cannot serve
    concurrent queries. Turning the guard off does not make it safe; it just
    trades a clear error for `InterfaceError: bad parameter or other API
    misuse`. Callers that use threads should read the database first and hand
    the results to their workers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def stored_digest(conn: sqlite3.Connection, slug: str) -> str | None:
    row = conn.execute("SELECT digest FROM books WHERE slug = ?", (slug,)).fetchone()
    return row["digest"] if row else None


def clear_book(conn: sqlite3.Connection, slug: str) -> None:
    """Drop everything derived from one book so a rebuild cannot leave orphans.

    Chroma goes first, because it is outside this transaction and it needs the
    chunk rows to find the ids. A crash between the two leaves chunks without
    vectors, which the next build re-embeds; the other order would leave
    vectors under ids SQLite is free to hand to a different book.
    """
    vectors.drop_book(conn, slug)
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE book = ?)",
        (slug,),
    )
    conn.execute("DELETE FROM chunks WHERE book = ?", (slug,))
    conn.execute("DELETE FROM exercises WHERE book = ?", (slug,))
    conn.execute("DELETE FROM books WHERE slug = ?", (slug,))


def add_book(conn: sqlite3.Connection, book: BookSpec, digest: str) -> None:
    conn.execute(
        "INSERT INTO books (slug, code, klass, subject, tier, digest) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (book.slug, book.code, book.klass, book.subject, book.tier, digest),
    )


def add_chunks(conn: sqlite3.Connection, chunks: Sequence[Chunk]) -> None:
    for chunk in chunks:
        cur = conn.execute(
            "INSERT INTO chunks (book, chapter, section, page, source, text) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk.book,
                chunk.chapter,
                chunk.section,
                chunk.page,
                chunk.source,
                chunk.text,
            ),
        )
        # fts rowid mirrors the chunk id, which is what search results join on
        conn.execute(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
            (cur.lastrowid, chunk.text),
        )


def add_exercises(
    conn: sqlite3.Connection, book: str, chapter: int, questions: Iterable[str]
) -> None:
    conn.executemany(
        "INSERT INTO exercises (book, chapter, question) VALUES (?, ?, ?)",
        [(book, chapter, q) for q in questions],
    )


def hits(
    conn: sqlite3.Connection, scored: Sequence[tuple[int, float]]
) -> list[RetrievalHit]:
    """Turn (chunk_id, score) pairs into hits, preserving the given order."""
    if not scored:
        return []
    placeholders = ",".join("?" * len(scored))
    rows = {
        row["id"]: row
        for row in conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",
            [cid for cid, _ in scored],
        )
    }
    return [
        RetrievalHit(
            chunk_id=cid,
            book=rows[cid]["book"],
            chapter=rows[cid]["chapter"],
            section=rows[cid]["section"],
            page=rows[cid]["page"],
            text=rows[cid]["text"],
            score=score,
        )
        for cid, score in scored
        if cid in rows
    ]
