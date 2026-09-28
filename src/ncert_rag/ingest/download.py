"""Put one PDF per chapter under data/books/<slug>/.

NCERT publishes chapters as separate files and ncert-cli keeps them that way,
which hands us chapter count, order and page ranges for free. Never merge them.
"""

import subprocess
from collections.abc import Iterable
from pathlib import Path

from ncert_rag.core.models import BookSpec
from ncert_rag.core.paths import BOOKS_DIR
from ncert_rag.core.registry import BOOKS


def chapter_files(slug_dir: Path) -> list[Path]:
    """chapter_NN.pdf, in printed order. prelims.pdf is not a chapter."""
    return sorted(slug_dir.glob("chapter_*.pdf"))


def provision(books: Iterable[BookSpec] = BOOKS) -> None:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    fetch = [
        book
        for book in books
        if len(chapter_files(BOOKS_DIR / book.slug)) != book.chapters
    ]

    if fetch:
        print(f"Downloading {len(fetch)} books: {' '.join(b.code for b in fetch)}")
        # ncert-cli exits 0 even when a book has no PDFs published, so the
        # count check below is what actually decides success
        subprocess.run(
            ["ncert", *(b.code for b in fetch), "--chapters", "-o", str(BOOKS_DIR)],
            check=False,
        )

    short = [
        f"{b.slug} ({len(chapter_files(BOOKS_DIR / b.slug))}/{b.chapters})"
        for b in books
        if len(chapter_files(BOOKS_DIR / b.slug)) != b.chapters
    ]
    if short:
        raise FileNotFoundError("incomplete books: " + ", ".join(short))
