"""Build the corpus: PDFs in, searchable SQLite out.

Idempotent per book: one whose PDFs hash the same as last time is skipped,
so a rebuild after touching one book does not re-parse the other sixteen.
"""

import hashlib
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from ncert_rag.core.models import BookSpec
from ncert_rag.core.paths import BOOKS_DIR
from ncert_rag.core.registry import BOOKS
from ncert_rag.ingest.download import chapter_files, provision
from ncert_rag.ingest.pdf import (
    book_offset,
    induce_profile,
    page_lines,
    parse_chapter,
)
from ncert_rag.ingest.text import from_pages, from_sections
from ncert_rag.services import embedder
from ncert_rag.store import db, vectors

_PROFILE_SAMPLE = 4  # chapters used to induce the heading gate


def _digest(files: list[Path]) -> str:
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _build_book(conn: sqlite3.Connection, book: BookSpec) -> list[int]:
    """Parse one book into chunks and exercises. Returns printed chapter numbers."""
    files = chapter_files(BOOKS_DIR / book.slug)
    chapters = [page_lines(path) for path in files]
    profile = induce_profile(chapters[:_PROFILE_SAMPLE])

    numbers = []
    for position, (path, lines) in enumerate(zip(files, chapters, strict=True), 1):
        parsed = parse_chapter(path, lines, profile, book.slug, position)
        numbers.append(parsed.chapter)

        db.add_exercises(conn, book.slug, parsed.chapter, parsed.exercises)
        db.add_chunks(conn, from_sections(parsed.sections))
        db.add_chunks(conn, from_pages(book.slug, parsed.chapter, parsed.pages))

    return numbers


def _embed_pending(conn: sqlite3.Connection) -> int:
    """Embed every chunk that has no vector yet, one chunk set at a time.

    The two stores cannot be joined, so the pending set is a difference: chunk
    ids SQLite knows about, minus the ids Chroma already holds. A build killed
    part way leaves the finished batches in Chroma and the rest pending, which
    is what the old per-batch commit bought.
    """
    total = 0
    for source in ("parsed", "raw"):
        done = vectors.stored_ids(source)
        rows = [
            row
            for row in conn.execute(
                "SELECT id, text FROM chunks WHERE source = ? ORDER BY id", (source,)
            )
            if row["id"] not in done
        ]
        if not rows:
            continue
        ids = [row["id"] for row in rows]
        texts = [row["text"] for row in rows]
        for start in range(0, len(ids), 512):
            window = slice(start, start + 512)
            vectors.save(source, ids[window], embedder.encode_documents(texts[window]))
            print(f"  {source}: embedded {min(start + 512, len(ids))}/{len(ids)}")
        total += len(ids)
    return total


def build(books: Iterable[BookSpec] = BOOKS, rebuild: bool = False) -> None:
    books = list(books)
    provision(books)
    conn = db.connect()

    for book in books:
        files = chapter_files(BOOKS_DIR / book.slug)
        digest = _digest(files)
        if not rebuild and db.stored_digest(conn, book.slug) == digest:
            print(f"{book.slug}: unchanged, skipping")
            continue

        db.clear_book(conn, book.slug)
        numbers = _build_book(conn, book)
        db.add_book(conn, book, digest)
        conn.commit()

        offset = book_offset(numbers)
        note = f"chapters {numbers[0]}-{numbers[-1]}"
        if offset is None:
            note += " (numbering inconsistent, using file order where unresolved)"
        elif offset != 0:
            note += f" (offset {offset:+d} from filenames)"
        print(f"{book.slug}: {len(numbers)} files, {note}")

    print("Embedding...")
    print(f"  {_embed_pending(conn)} new vectors")
    _report(conn, books)
    conn.close()


def _report(conn: sqlite3.Connection, books: list[BookSpec]) -> None:
    """Invariants worth seeing every build."""
    print("\nCorpus")
    for source in ("parsed", "raw"):
        n = conn.execute(
            "SELECT COUNT(*) n FROM chunks WHERE source = ?", (source,)
        ).fetchone()["n"]
        stored = vectors.count(source)
        print(f"  {source:>6} chunks: {n} ({stored} vectors)")
        # More vectors than chunks cannot happen in a healthy store. It means
        # data/chroma survived a corpus.db that did not: chunk ids restart at
        # 1, so old vectors now sit under ids belonging to different chapters.
        if stored > n:
            raise SystemExit(
                f"  ! data/chroma holds {stored} {source} vectors but corpus.db "
                f"has {n} chunks. Delete data/chroma and rebuild."
            )

    total_q = conn.execute("SELECT COUNT(*) n FROM exercises").fetchone()["n"]
    print(f"  exercise questions: {total_q}")

    for book in books:
        found = conn.execute(
            "SELECT COUNT(DISTINCT chapter) n FROM chunks WHERE book = ?",
            (book.slug,),
        ).fetchone()["n"]
        if found != book.chapters:
            print(f"  ! {book.slug}: {found} chapters parsed, {book.chapters} expected")
