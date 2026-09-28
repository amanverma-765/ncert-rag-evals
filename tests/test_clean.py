"""Text normalization, and the rules that only fire on multi-line blocks.

Three of these once did nothing at all: `page_lines` cleans one line at a time,
and a single line contains no newline for them to match against.
"""

from ncert_rag.ingest.text.clean import (
    PAGE_EDGE,
    clean_text,
    is_page_number,
    strip_running_heads,
)


def test_word_broken_across_lines_is_rejoined():
    assert clean_text("photo-\nsynthesis happens") == "photosynthesis happens"


def test_a_compound_before_a_capital_survives():
    # "well-Known" is a real hyphen, not a line break in the middle of a word
    assert "well-\nKnown" in clean_text("well-\nKnown result")


def test_drop_cap_is_reattached():
    assert clean_text("C\nonsider the following") == "Consider the following"


def test_ligatures_become_letters_but_superscripts_survive():
    assert clean_text("ﬁnal oﬀer") == "final offer"
    assert clean_text("x² + y²") == "x² + y²"  # NFKC would flatten these


def test_letter_tracked_heading_is_collapsed():
    assert clean_text("I N T R O D U C T I O N") == "INTRODUCTION"


def test_ordinary_spaced_capitals_are_left_alone():
    assert clean_text("I a m") == "I a m"


def test_clean_text_is_idempotent():
    # sections.py runs it a second time on joined lines; that must be safe
    once = clean_text("photo-\nsynthesis  ﬁne\n\n\n\ntext")
    assert clean_text(once) == once


# --- page numbers are furniture only at a page edge -------------------------


def test_number_at_the_top_or_bottom_of_a_page_is_furniture():
    assert is_page_number("75", 0, 40)
    assert is_page_number("75", 39, 40)


def test_number_in_the_middle_of_a_page_is_content():
    # a coefficient in a balanced equation, or a cell of a chemistry table
    assert not is_page_number("2", 20, 40)


def test_non_numeric_text_is_never_a_page_number():
    assert not is_page_number("Chapter 5", 0, 40)


def test_page_edge_is_symmetric():
    assert is_page_number("9", PAGE_EDGE - 1, 30)
    assert not is_page_number("9", PAGE_EDGE, 30)


def test_running_heads_are_dropped_but_content_is_not():
    pages = [f"Science\nreal content {n}\n{n}" for n in range(1, 6)]
    out = strip_running_heads(pages)
    assert all("Science" not in page for page in out)
    assert all(f"real content {n}" in out[n - 1] for n in range(1, 6))


def test_a_mid_page_number_survives_running_head_removal():
    pages = ["Science\nH\n7\nO is water\n1"] + [
        f"Science\nplain text\n{n}" for n in range(2, 6)
    ]
    out = strip_running_heads(pages)
    assert "7" in out[0]  # mid-page, so content
    assert not out[0].endswith("1")  # page edge, so furniture


def test_a_number_repeating_on_most_pages_is_furniture_first():
    # The running-head rule runs ahead of the page-number rule, so a subscript
    # that happens to sit alone on a line on most pages is dropped as repeated
    # furniture regardless of position. Both chunk sets share this rule, so it
    # does not skew parsed against raw -- it just costs both of them the line.
    pages = ["H\n2\nO" for _ in range(5)]
    assert all("2" not in page for page in strip_running_heads(pages))
