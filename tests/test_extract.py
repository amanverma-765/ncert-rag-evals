"""Turning PDF spans into line text.

PyMuPDF splits a line at every font change, so a subscript starts a new span.
Joining spans directly welds words together; the plain extractor inserts a
space where they do not abut, and this reproduces that.
"""

from ncert_rag.ingest.pdf.extract import _SPAN_GAP, _join_spans


def span(text: str, x0: float, x1: float) -> dict:
    return {"text": text, "bbox": (x0, 0.0, x1, 10.0)}


def test_spans_separated_by_a_gap_get_a_space():
    # "En becomes" arrives as ['E']['n']['becomes ...'] with a gap before the word
    spans = [span("E", 10, 16), span("n", 16, 20), span("becomes", 30, 60)]
    assert _join_spans(spans) == "En becomes"


def test_abutting_spans_stay_welded():
    # H, subscript 2, O must not become "H 2 O"
    spans = [span("H", 10, 16), span("2", 16, 20), span("O", 20, 26)]
    assert _join_spans(spans) == "H2O"


def test_a_gap_below_the_threshold_is_not_a_word_break():
    spans = [span("kg", 10, 20), span("m", 20 + _SPAN_GAP / 2, 26)]
    assert _join_spans(spans) == "kgm"


def test_an_existing_space_is_not_doubled():
    spans = [span("decreases), ", 10, 40), span("E", 60, 66)]
    assert _join_spans(spans) == "decreases), E"


def test_a_leading_space_on_the_next_span_is_not_doubled():
    assert _join_spans([span("one", 10, 20), span(" two", 40, 60)]) == "one two"


def test_a_single_span_is_returned_unchanged():
    assert _join_spans([span("solitary", 10, 40)]) == "solitary"
