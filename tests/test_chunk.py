from ncert_rag.core.models import Section
from ncert_rag.ingest.text.chunk import (
    _ENCODING,
    OVERLAP,
    TOKENS,
    _windows,
    from_pages,
    from_sections,
)


def section(number: str, text: str) -> Section:
    return Section(book="book", chapter=1, number=number, text=text, page=3)


def long_text(tokens: int) -> str:
    return " ".join(f"word{i}" for i in range(tokens))


# --- the window generator, tested on plain integers ---------------------------
# A decode/encode roundtrip is not token-identical when a window splits a word,
# so the overlap invariant is asserted here rather than on decoded chunk text.


def test_windows_overlap_by_exactly_overlap_tokens():
    got = list(_windows(list(range(TOKENS * 2))))
    assert len(got) > 1
    first, second = got[0][1], got[1][1]
    assert first[-OVERLAP:] == second[:OVERLAP]


def test_windows_cover_every_token():
    tokens = list(range(TOKENS * 3 + 7))
    seen = {t for _start, window in _windows(tokens) for t in window}
    assert seen == set(tokens)


def test_windows_report_where_each_one_starts():
    starts = [start for start, _window in _windows(list(range(TOKENS * 2)))]
    assert starts[0] == 0
    assert starts[1] == TOKENS - OVERLAP


def test_windows_on_empty_input_yield_nothing():
    assert list(_windows([])) == []


# --- section-aware chunks -----------------------------------------------------


def test_short_section_stays_one_chunk():
    chunks = from_sections([section("1.1", "a short passage of text")])
    assert len(chunks) == 1
    assert chunks[0].section == "1.1"
    assert chunks[0].source == "parsed"


def test_windows_never_span_two_sections():
    chunks = from_sections([section("1.1", "first"), section("1.2", "second")])
    assert [c.section for c in chunks] == ["1.1", "1.2"]
    assert "second" not in chunks[0].text


def test_long_section_splits_and_every_piece_keeps_its_section():
    chunks = from_sections([section("1.1", long_text(TOKENS * 2))])
    assert len(chunks) > 1
    assert all(c.section == "1.1" for c in chunks)


def test_no_chunk_exceeds_the_token_window():
    for chunk in from_sections([section("1.1", long_text(TOKENS * 2))]):
        # re-encoding a decoded window can shift a token or two at the split
        assert len(_ENCODING.encode(chunk.text)) <= TOKENS + 2


def test_empty_section_produces_no_chunk():
    chunks = from_sections([section("1.1", "real text"), section("1.2", "")])
    assert len(chunks) == 1
    assert all(c.text.strip() for c in chunks)


# --- raw windows --------------------------------------------------------------


def test_raw_chunks_carry_the_page_the_window_starts_on():
    chunks = from_pages("book", 7, ["first page", "second page"])
    assert chunks[0].page == 1
    assert all(c.source == "raw" and c.section is None for c in chunks)
    assert all(c.chapter == 7 for c in chunks)


def test_raw_chunking_keeps_text_from_every_page():
    joined = " ".join(c.text for c in from_pages("b", 1, [long_text(TOKENS), "tail"]))
    assert "word0" in joined
    assert "tail" in joined


def test_raw_chunking_does_not_run_two_pages_together():
    [chunk] = from_pages("b", 1, ["alpha", "beta"])
    assert "alphabeta" not in chunk.text
