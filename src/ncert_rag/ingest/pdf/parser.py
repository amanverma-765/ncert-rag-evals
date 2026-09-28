"""Deep chapter parser: turns chapter PDFs into structured sections and exercises.

Consolidates font-gate induction, chapter number resolution, exercise cutoffs,
furniture removal, and section splitting into a single deep module.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ncert_rag.core.models import ParsedChapter, Section
from ncert_rag.ingest.pdf.extract import Line, page_texts
from ncert_rag.ingest.text.clean import clean_text, is_page_number

# "2.1" / "2.1.3", optionally followed by a title on the same line.
SECTION = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,2})(?![\d.])[ \t]*(.*)$")

_MIN_MARKS = 2  # chapters with fewer marks than this are treated as unnumbered
_MIN_KEEP = 0.3  # cuts that would drop >70% of a chapter are rejected as false matches

# Exercise question extraction patterns
_ORDINAL = re.compile(r"^(\d{1,2})\.\s*(.*)$")
_BY_CHAPTER = re.compile(r"^(\d{1,2})\.(\d{1,2})\s*(.*)$")
_MIN_RUN = 4
_MIN_WORDS = 6
_MAX_CHARS = 400
_ASKS = re.compile(
    r"(?i)^(what|why|how|when|where|which|who|whose|explain|define|describe|name|"
    r"write|give|state|list|differentiate|distinguish|discuss|compare|account|"
    r"identify|mention|justify|suggest|outline|illustrate|comment|answer|choose)\b"
)
_STAMP = re.compile(r"Reprint\s*\d{4}.*$")
_MIN_ALPHA = 0.7


@dataclass(frozen=True, slots=True)
class HeadingProfile:
    body_size: float
    min_size: float
    require_bold: bool


@dataclass(frozen=True, slots=True)
class Mark:
    """A detected section heading and where it sits in the line stream."""

    number: str
    title: str
    page: int
    index: int

    @property
    def prefix(self) -> int:
        return int(self.number.split(".")[0])

    @property
    def key(self) -> tuple[int, ...]:
        return tuple(int(p) for p in self.number.split("."))


# --- Font Gate Induction ------------------------------------------------------


def body_size(lines: list[Line]) -> float:
    """The font size most characters are set in."""
    weight: Counter[float] = Counter()
    for line in lines:
        weight[line.size] += len(line.text)
    return weight.most_common(1)[0][0] if weight else 10.5


def find_marks(lines: list[Line], profile: HeadingProfile) -> list[Mark]:
    """Every heading candidate the profile admits, in reading order."""
    marks: list[Mark] = []
    for i, line in enumerate(lines):
        if line.size < profile.min_size or (profile.require_bold and not line.bold):
            continue
        m = SECTION.match(line.text)
        if not m:
            continue
        number, title = m.group(1), m.group(2).strip()
        if not title:
            title = _title_after(lines, i, line.size)
        marks.append(Mark(number=number, title=title, page=line.page, index=i))
    return _dedupe(marks)


def _title_after(lines: list[Line], i: int, size: float) -> str:
    for nxt in lines[i + 1 : i + 3]:
        if abs(nxt.size - size) < 0.6 and not SECTION.match(nxt.text):
            return nxt.text
    return ""


def _dedupe(marks: list[Mark]) -> list[Mark]:
    out: list[Mark] = []
    for mark in marks:
        if out and out[-1].number == mark.number and mark.index - out[-1].index < 6:
            if len(mark.title) > len(out[-1].title):
                out[-1] = mark
            continue
        out.append(mark)
    return out


def _coherence(marks: list[Mark]) -> int:
    if not marks:
        return 0
    prefix = Counter(m.prefix for m in marks).most_common(1)[0][0]
    kept = [m for m in marks if m.prefix == prefix]

    run, last = 0, ()
    for mark in kept:
        if mark.key >= last:
            run += 1
            last = mark.key
    return run


def induce_profile(chapters: list[list[Line]]) -> HeadingProfile:
    """Pick the font gate that best explains this book's section numbering."""
    sizes = [body_size(lines) for lines in chapters if lines]
    body = Counter(sizes).most_common(1)[0][0] if sizes else 10.5

    gates = [
        HeadingProfile(body, body + 0.9, True),
        HeadingProfile(body, body + 0.9, False),
        HeadingProfile(body, body, True),
        HeadingProfile(body, body, False),
    ]
    return max(
        gates,
        key=lambda g: sum(_coherence(find_marks(lines, g)) for lines in chapters),
    )


# --- Chapter Number Resolution ------------------------------------------------


def chapter_number(marks: list[Mark], fallback: int) -> int:
    """Majority vote over the section marks' leading component."""
    if not marks:
        return fallback
    number, votes = Counter(m.prefix for m in marks).most_common(1)[0]
    return number if votes >= max(2, len(marks) // 3) else fallback


def book_offset(numbers: list[int]) -> int | None:
    """Constant gap between printed number and file position, if consistent."""
    if not numbers:
        return None
    gaps = Counter(number - i for i, number in enumerate(numbers, start=1))
    gap, votes = gaps.most_common(1)[0]
    return gap if votes >= len(numbers) - 1 else None


# --- Exercise Page Cutoffs & Extraction ---------------------------------------


def _is_question(text: str) -> bool:
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if alpha / len(text) < _MIN_ALPHA:
        return False
    return text.endswith("?") or bool(_ASKS.match(text))


def _starts(lines: list[Line], chapter: int) -> list[tuple[int, int, str]]:
    found = []
    for i, line in enumerate(lines):
        by_chapter = _BY_CHAPTER.match(line.text)
        if by_chapter:
            if int(by_chapter.group(1)) == chapter:
                found.append((i, int(by_chapter.group(2)), by_chapter.group(3).strip()))
            continue
        ordinal = _ORDINAL.match(line.text)
        if ordinal:
            found.append((i, int(ordinal.group(1)), ordinal.group(2).strip()))
    return found


def _runs(starts: list[tuple[int, int, str]]) -> list[list[tuple[int, int, str]]]:
    runs: list[list[tuple[int, int, str]]] = []
    for start in starts:
        if runs and start[1] == runs[-1][-1][1] + 1:
            runs[-1].append(start)
        else:
            runs.append([start])
    return [run for run in runs if len(run) >= _MIN_RUN]


def _questions(lines: list[Line], run: list[tuple[int, int, str]]) -> list[str]:
    bounds = [i for i, _n, _t in run[1:]] + [len(lines)]
    out = []
    for (index, _number, inline), end in zip(run, bounds, strict=True):
        body = " ".join([inline] + [line.text for line in lines[index + 1 : end]])
        body = _STAMP.sub("", re.sub(r"\s+", " ", body)).strip()
        if (
            len(body.split()) >= _MIN_WORDS
            and len(body) <= _MAX_CHARS
            and _is_question(body)
        ):
            out.append(body)
    return out


def find_exercises(lines: list[Line], chapter: int) -> tuple[list[str], int | None]:
    """Find this chapter's exercise questions, and the page they start on."""
    best: list[str] = []
    page: int | None = None
    for run in _runs(_starts(lines, chapter)):
        found = _questions(lines, run)
        if len(found) >= len(best):
            best, page = found, lines[run[0][0]].page
    return best, (page if best else None)


def page_cut(lines: list[Line], exercise_page: int | None) -> int | None:
    """First page to drop, or None to keep the whole chapter."""
    if exercise_page is None or exercise_page <= 1 or not lines:
        return None
    total = max(line.page for line in lines)
    return exercise_page if (exercise_page - 1) / total >= _MIN_KEEP else None


def before(lines: list[Line], cut: int | None) -> list[Line]:
    return lines if cut is None else [line for line in lines if line.page < cut]


def pages_before(pages: list[str], cut: int | None) -> list[str]:
    return pages if cut is None else pages[: cut - 1]


# --- Section Splitting & Furniture Removal ------------------------------------


def _drop_furniture(lines: list[Line], threshold: float = 0.4) -> list[Line]:
    pages = len({line.page for line in lines})
    if pages < 3:
        return lines

    counts = Counter(line.text for line in lines if len(line.text) <= 60)
    limit = max(2, threshold * pages)

    per_page: dict[int, list[int]] = {}
    for index, line in enumerate(lines):
        per_page.setdefault(line.page, []).append(index)
    position = {
        index: (place, len(group))
        for group in per_page.values()
        for place, index in enumerate(group)
    }

    kept = []
    for index, line in enumerate(lines):
        if SECTION.match(line.text):
            kept.append(line)
            continue
        place, total = position[index]
        if counts[line.text] >= limit or is_page_number(line.text, place, total):
            continue
        kept.append(line)
    return kept


def _advancing(marks: list[Mark], chapter: int) -> list[Mark]:
    kept: list[Mark] = []
    last: tuple[int, ...] = ()
    for mark in marks:
        if mark.prefix != chapter or mark.key <= last:
            continue
        kept.append(mark)
        last = mark.key
    return kept


def _text(lines: list[Line], start: int, end: int) -> str:
    return clean_text("\n".join(line.text for line in lines[start:end])).strip()


def split_sections(
    lines: list[Line], profile: HeadingProfile, book: str, chapter: int
) -> list[Section]:
    """Cut chapter lines into Section records."""
    lines = _drop_furniture(lines)
    kept = _advancing(find_marks(lines, profile), chapter)

    if len(kept) < _MIN_MARKS:
        body = _text(lines, 0, len(lines))
        page = lines[0].page if lines else 1
        return (
            [Section(book=book, chapter=chapter, number=None, text=body, page=page)]
            if body
            else []
        )

    sections: list[Section] = []
    opening = _text(lines, 0, kept[0].index)
    if len(opening) > 200:
        sections.append(
            Section(
                book=book,
                chapter=chapter,
                number=None,
                text=opening,
                page=lines[0].page,
            )
        )

    ends = [mark.index for mark in kept[1:]] + [len(lines)]
    for mark, end in zip(kept, ends, strict=True):
        body = _text(lines, mark.index, end)
        if body:
            sections.append(
                Section(
                    book=book,
                    chapter=chapter,
                    number=mark.number,
                    text=body,
                    page=mark.page,
                )
            )

    return sections


# --- High-Level Deep Parser Seam ----------------------------------------------


def parse_chapter(
    path: Path,
    lines: list[Line],
    profile: HeadingProfile,
    book: str,
    position: int,
) -> ParsedChapter:
    """Parse one chapter PDF completely into sections, raw pages, and exercises."""
    marks = find_marks(lines, profile)
    number = chapter_number(marks, position)

    exercises, exercise_page = find_exercises(lines, number)
    cut = page_cut(lines, exercise_page)

    sections = split_sections(before(lines, cut), profile, book, number)
    pages = pages_before(page_texts(path), cut)

    return ParsedChapter(
        book=book,
        chapter=number,
        sections=sections,
        pages=pages,
        exercises=exercises,
    )
