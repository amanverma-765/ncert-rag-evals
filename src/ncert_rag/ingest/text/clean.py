"""Text normalization applied identically to both chunk sets.

Pure functions on already-extracted text. Running both sides through the same
cleaning keeps chunking the only difference between them.
"""

import re
from collections import Counter

# PyMuPDF emits real ligature codepoints. NFKC would fix them but also flatten
# x² to x2, which the maths books cannot afford.
_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})

# "I N T R O D U C T I O N": headings set with letter tracking. Four letters
# minimum so ordinary prose ("I a m") is never a candidate.
_TRACKED = re.compile(r"\b(?:[A-Z] ){3,}[A-Z]\b")

# a word broken across a line: "photo-\nsynthesis". Only when the next line
# starts lowercase, so "well-\nKnown" style compounds survive.
_HYPHEN_BREAK = re.compile(r"(\w)-\n([a-z])")

# NCERT opens sections with a drop cap, which extracts as a lone capital on its
# own line ahead of the rest of the word
_DROP_CAP = re.compile(r"^([A-Z])\n(?=[a-z])", re.MULTILINE)

_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

# How many lines at each end of a page count as furniture. A page number sits
# at the top or bottom; a lone number in the middle of a page is content --
# usually a cell of a chemistry table, or a coefficient in a balanced equation.
# Matching every standalone number cost 1,937 content lines to remove 242 real
# page numbers, so position is what decides.
PAGE_EDGE = 2


def is_page_number(text: str, position: int, total: int) -> bool:
    """A standalone number near the top or bottom of its page."""
    if not _PAGE_NUMBER.match(text):
        return False
    return position < PAGE_EDGE or position >= total - PAGE_EDGE


def clean_text(text: str) -> str:
    """Normalize one page's extracted text."""
    text = text.translate(_LIGATURES).replace("\xa0", " ")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _DROP_CAP.sub(r"\1", text)
    text = _TRACKED.sub(lambda m: m.group().replace(" ", ""), text)
    # collapse runs of spaces/tabs but keep newlines: section splitting needs them
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_running_heads(pages: list[str], threshold: float = 0.4) -> list[str]:
    """Drop the running header/footer lines that repeat across a chapter.

    A short line appearing on 40%+ of pages is furniture (book title, chapter
    name, page numbers), not content.
    """
    if len(pages) < 3:
        return pages

    counts = Counter(
        line.strip()
        for page in pages
        for line in page.splitlines()
        if 0 < len(line.strip()) <= 60
    )
    furniture = {
        line for line, n in counts.items() if n >= max(2, threshold * len(pages))
    }

    kept = []
    for page in pages:
        lines = [line for line in page.splitlines() if line.strip()]
        kept.append(
            "\n".join(
                line
                for position, line in enumerate(lines)
                if line.strip() not in furniture
                and not is_page_number(line, position, len(lines))
            ).strip()
        )
    return kept
