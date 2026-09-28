"""Value types shared across the pipeline. No internal imports live here."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# 'clean' parses reliably; 'fragmented' shreds equations into <=3-char lines and
# leans on window-chunk fallback. Measured over every chapter, the four maths
# books run 45-76% such lines against at most 33% for any other book, so the
# label is assigned by subject and the separation is not close.
Tier = Literal["clean", "fragmented"]

# 'parsed' chunks respect section boundaries; 'raw' are fixed windows over the
# same cleaned text. Both live in one table so the arms share a corpus.
ChunkSource = Literal["parsed", "raw"]


class BookSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str  # directory name under data/books/
    code: str  # ncert-cli catalog code, e.g. "kebo1"
    klass: int
    subject: str
    tier: Tier
    chapters: int  # expected count, checked as a build invariant


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    book: str
    chapter: int
    number: str | None  # "2.1"; None when the chapter carries no numbering
    text: str
    page: int  # page the section starts on


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    book: str
    chapter: int
    section: str | None
    page: int
    text: str
    source: ChunkSource


class RetrievalHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: int
    book: str
    chapter: int
    section: str | None
    page: int
    text: str
    score: float


class ParsedChapter(BaseModel):
    model_config = ConfigDict(frozen=True)

    book: str
    chapter: int
    sections: list[Section]
    pages: list[str]
    exercises: list[str]
