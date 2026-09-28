"""Tests for the deep chapter parser module."""

from pathlib import Path

from ncert_rag.core.models import ParsedChapter
from ncert_rag.ingest.pdf.extract import Line
from ncert_rag.ingest.pdf.parser import (
    HeadingProfile,
    induce_profile,
    parse_chapter,
)


def test_induce_profile_selects_best_font_gate():
    # Chapter with bold headings at size 12.0 and body at 10.5
    ch1 = [
        Line("1.1 Overview", 12.0, True, 1),
        Line("Some body text here.", 10.5, False, 1),
        Line("1.2 Details", 12.0, True, 2),
        Line("More body text.", 10.5, False, 2),
    ]
    profile = induce_profile([[Line("Some intro", 10.5, False, 1)] + ch1])
    assert isinstance(profile, HeadingProfile)
    assert profile.body_size == 10.5


def test_parse_chapter_returns_structured_parsed_chapter(tmp_path: Path, monkeypatch):
    # Mock page_texts so we do not need a real PDF file on disk
    mock_pages = [
        "1.1 First Section\nContent on page 1\n",
        "1.2 Second Section\nContent on page 2\n",
        "1. What is X?\n2. Why is Y?\n3. How does Z?\n4. Explain W?\n",
    ]
    monkeypatch.setattr(
        "ncert_rag.ingest.pdf.parser.page_texts", lambda _path: mock_pages
    )

    lines = [
        Line("1.1 First Section", 12.0, True, 1),
        Line("Content on page 1", 10.5, False, 1),
        Line("1.2 Second Section", 12.0, True, 2),
        Line("Content on page 2", 10.5, False, 2),
        Line("1. What is the role of green leaves in plants?", 10.5, False, 3),
        Line("2. Why do living cells require energy from food?", 10.5, False, 3),
        Line("3. How does water move through xylem vessels?", 10.5, False, 3),
        Line("4. Explain how gas exchange occurs in daytime?", 10.5, False, 3),
    ]
    profile = HeadingProfile(body_size=10.5, min_size=11.0, require_bold=True)
    fake_pdf = tmp_path / "chapter_01.pdf"

    parsed = parse_chapter(
        path=fake_pdf,
        lines=lines,
        profile=profile,
        book="class_10_science",
        position=1,
    )

    assert isinstance(parsed, ParsedChapter)
    assert parsed.book == "class_10_science"
    assert parsed.chapter == 1
    assert len(parsed.sections) >= 1
    assert parsed.sections[0].number == "1.1"
    assert len(parsed.exercises) == 4
    assert parsed.exercises[0] == "What is the role of green leaves in plants?"
    # Page 3 is exercise page, so cut dropped it from raw pages
    assert len(parsed.pages) == 2
