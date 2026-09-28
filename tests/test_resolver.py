from ncert_rag.ingest.pdf.parser import Mark, book_offset, chapter_number


def marks(*numbers: str) -> list[Mark]:
    return [Mark(n, "", 1, i) for i, n in enumerate(numbers)]


def test_chapter_number_follows_the_majority_of_section_marks():
    # a Part II file named chapter_01 whose sections read 6.x is chapter 6
    assert chapter_number(marks("6.1", "6.2", "6.2.1", "6.3"), fallback=1) == 6


def test_a_single_stray_mark_does_not_decide_the_chapter():
    assert chapter_number(marks("9.1"), fallback=4) == 4


def test_chapter_number_falls_back_when_nothing_was_found():
    assert chapter_number([], fallback=3) == 3


def test_noise_loses_to_the_dominant_prefix():
    assert chapter_number(marks("7.1", "7.2", "7.3", "2.5"), fallback=1) == 7


def test_offset_is_the_constant_gap_from_file_position():
    # files 1..5 printing chapters 6..10
    assert book_offset([6, 7, 8, 9, 10]) == 5


def test_offset_is_zero_for_a_single_volume_book():
    assert book_offset([1, 2, 3, 4]) == 0


def test_one_odd_chapter_is_tolerated():
    assert book_offset([1, 2, 3, 9]) == 0


def test_inconsistent_numbering_reports_no_offset():
    assert book_offset([4, 1, 9, 2]) is None


def test_no_chapters_reports_no_offset():
    assert book_offset([]) is None
