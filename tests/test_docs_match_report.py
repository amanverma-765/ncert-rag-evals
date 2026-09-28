"""README.md and EVALUATION.md copy REPORT.md's tables; this fails when a copy drifts.

Both documents restate generated numbers so a reader never has to open the
report to see them. Copies go stale silently -- that is how a fragmented-tier
paragraph ended up quoting figures from a corpus two rebuilds old -- so every
copied row is compared against the generated one here.

Only tables are guarded. Prose figures ("+6.0 points", "three questions") are
written by hand and reviewed by hand.
"""

from pathlib import Path

import pytest

from evals.run import REPORT_PATH
from ncert_rag.retrieve import ARMS

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
EVALUATION = ROOT / "evals" / "EVALUATION.md"

R_AT_5 = 1  # cells are [R@1, R@5, R@10, MRR, n]
N = 4


def _sections(path: Path) -> dict[str, str]:
    """Markdown split into {heading: body}, so same-named tables stay apart."""
    out: dict[str, str] = {}
    heading, body = "", []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            out[heading] = "\n".join(body)
            heading, body = line.lstrip("#").strip(), []
        else:
            body.append(line)
    out[heading] = "\n".join(body)
    return out


def _rows(body: str) -> dict[str, list[str]]:
    """{arm: cells} for every table row whose first cell names an arm."""
    rows = {}
    for line in body.splitlines():
        cells = [c.strip().strip("`*") for c in line.strip().strip("|").split("|")]
        if len(cells) > 1:
            arm = cells[0]
            if arm == "vector_parsed":
                arm = "vector"
            if arm in ARMS:
                rows[arm] = [c.strip().strip("*") for c in cells[1:]]
    return rows


def _report(heading: str) -> dict[str, list[str]]:
    rows = _rows(_sections(REPORT_PATH)[heading])
    # A renamed heading would hand every assertion an empty dict and pass.
    assert len(rows) == len(ARMS), f"REPORT.md '{heading}' lost rows: {sorted(rows)}"
    return rows


def _tier_summary(body: str) -> dict[str, list[str]]:
    summary = {}
    for line in body.splitlines():
        cells = [c.strip().strip("`*") for c in line.strip().strip("|").split("|")]
        if cells and cells[0] in ("clean", "fragmented"):
            summary[cells[0]] = [c.strip().strip("*") for c in cells[1:]]
    return summary


def test_evaluation_accuracy_table_matches_report() -> None:
    expected = _report("All questions")
    assert _rows(_sections(EVALUATION)["4. Accuracy Results"]) == expected


@pytest.mark.parametrize("tier", ["clean", "fragmented"])
def test_evaluation_tier_summary_matches_report(tier: str) -> None:
    report = _report(f"{tier.capitalize()} tier")
    expected = [report[arm][R_AT_5] for arm in ("bm25", "hybrid", "expansion_hybrid")]
    expected.append(report["bm25"][N])

    summary = _tier_summary(_sections(EVALUATION)["Performance by Tier (R@5)"])
    assert summary[tier] == expected


def test_readme_full_comparison_matches_report() -> None:
    expected = _report("All questions")
    assert _rows(_sections(README)["Full Evaluation Results"]) == expected


@pytest.mark.parametrize("tier", ["clean", "fragmented"])
def test_readme_tier_summary_matches_report(tier: str) -> None:
    """The README's tier strip is R@5 for three arms, pulled from two tables."""
    report = _report(f"{tier.capitalize()} tier")
    expected = [report[arm][R_AT_5] for arm in ("bm25", "hybrid", "expansion_hybrid")]
    expected.append(report["bm25"][N])

    summary = _tier_summary(_sections(README)["The Two Text Tiers"])
    assert summary[tier] == expected
