from ncert_rag.ingest.pdf.extract import Line
from ncert_rag.ingest.pdf.parser import before, page_cut, pages_before


def lines(pages: int) -> list[Line]:
    return [Line(f"text {p}", 10.5, False, p) for p in range(1, pages + 1)]


def test_no_exercises_keeps_the_whole_chapter():
    assert page_cut(lines(10), None) is None


def test_cut_is_the_page_the_exercises_start_on():
    assert page_cut(lines(10), 9) == 9


def test_a_cut_that_would_gut_the_chapter_is_refused():
    # "exercises" on page 2 of 10 is a misdetection, not an exercise list
    assert page_cut(lines(10), 2) is None


def test_exercises_on_page_one_are_refused():
    assert page_cut(lines(10), 1) is None


def test_before_drops_the_cut_page_and_everything_after():
    assert [line.page for line in before(lines(5), 4)] == [1, 2, 3]


def test_before_is_a_no_op_without_a_cut():
    assert len(before(lines(5), None)) == 5


def test_pages_before_cuts_the_raw_corpus_at_the_same_place():
    pages = ["p1", "p2", "p3", "p4", "p5"]
    assert pages_before(pages, 4) == ["p1", "p2", "p3"]
    assert pages_before(pages, None) == pages
