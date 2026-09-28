"""Pull text out of a chapter PDF, by page and by line.

`page_texts` feeds the raw chunks; `page_lines` carries the font metrics the
heading detection needs.
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ncert_rag.ingest.text.clean import clean_text, strip_running_heads

_BOLD_MARKERS = ("Demi", "Bold", "Black", "Heavy")

# Horizontal gap, in points, above which two spans on the same line are
# separate words. PyMuPDF's plain get_text() inserts a space at such gaps;
# joining spans directly welds words together ("En becomes" -> "Enbecomes"),
# which matters most in chemistry, where every subscript starts a new span.
# 1.5 reproduces the plain extractor on 98.3% of multi-span lines in a
# three-book sample, against 76.3% for a bare join, and is a clear optimum:
# 1.0 scores 95.8% and 2.0 scores 97.5%.
_SPAN_GAP = 1.5


def _join_spans(spans: list[dict]) -> str:
    """Span text in reading order, spaced the way the plain extractor spaces it.

    Subscripts abut their base (H, 2, O) and must not gain a space, so the gap
    is what decides, not the span boundary.
    """
    out = [spans[0]["text"]]
    for previous, span in zip(spans, spans[1:], strict=False):
        gap = span["bbox"][0] - previous["bbox"][2]
        if (
            gap > _SPAN_GAP
            and not out[-1].endswith(" ")
            and not span["text"].startswith(" ")
        ):
            out.append(" ")
        out.append(span["text"])
    return "".join(out)


@dataclass(frozen=True, slots=True)
class Line:
    text: str
    size: float
    bold: bool
    page: int


def page_texts(path: Path) -> list[str]:
    """Cleaned text per page, furniture removed."""
    with pymupdf.open(path) as doc:
        pages = [clean_text(page.get_text()) for page in doc]
    return strip_running_heads(pages)


def page_lines(path: Path) -> list[Line]:
    """Every non-empty line with the font metrics needed for heading detection.

    Size is the largest span on the line and bold is true if any span is, so a
    heading whose number and title differ slightly still reads as one unit.
    """
    out: list[Line] = []
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = clean_text(_join_spans(spans))
                    if not text:
                        continue
                    out.append(
                        Line(
                            text=text,
                            size=max(round(s["size"], 1) for s in spans),
                            bold=any(
                                m in s["font"] for s in spans for m in _BOLD_MARKERS
                            ),
                            page=pno,
                        )
                    )
    return out
