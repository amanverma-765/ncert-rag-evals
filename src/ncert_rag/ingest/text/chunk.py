"""Two chunkings of the same text: section-aware, and fixed windows.

Both use the same window size and the same cleaned text, so the eval can
attribute any difference between them to the boundaries alone.

Windows are counted in tokens rather than words, so a chunk's size means the
same thing here as it does in the prompt it eventually lands in.
"""

from collections.abc import Iterator

import tiktoken

from ncert_rag.core.models import Chunk, Section

TOKENS = 512
OVERLAP = 64

# cl100k is not the serving model's tokenizer, but it is stable and offline,
# and a window only has to be a consistent size, not an exactly billed one.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _windows(tokens: list[int]) -> Iterator[tuple[int, list[int]]]:
    """(start index, window) pairs covering the whole token list."""
    if not tokens:
        return
    start = 0
    while True:
        end = min(start + TOKENS, len(tokens))
        yield start, tokens[start:end]
        if end >= len(tokens):
            return
        start = end - OVERLAP


def from_sections(sections: list[Section]) -> list[Chunk]:
    """Chunk within section boundaries; a window never spans two sections."""
    return [
        Chunk(
            book=section.book,
            chapter=section.chapter,
            section=section.number,
            page=section.page,
            text=_ENCODING.decode(window),
            source="parsed",
        )
        for section in sections
        for _start, window in _windows(_ENCODING.encode(section.text))
    ]


def from_pages(book: str, chapter: int, pages: list[str]) -> list[Chunk]:
    """Chunk the chapter as one flat stream, structure ignored."""
    tokens: list[int] = []
    page_of: list[int] = []  # page each token came from, for citations
    for number, text in enumerate(pages, start=1):
        # the trailing newline keeps the last token of one page from running
        # into the first of the next, which " ".join used to do for free
        page_tokens = _ENCODING.encode(text + "\n")
        tokens.extend(page_tokens)
        page_of.extend([number] * len(page_tokens))

    return [
        Chunk(
            book=book,
            chapter=chapter,
            section=None,
            page=page_of[start],
            text=_ENCODING.decode(window),
            source="raw",
        )
        for start, window in _windows(tokens)
    ]
