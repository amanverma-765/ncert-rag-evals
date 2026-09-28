"""Build the query set: exercise questions, reworded the way students ask.

Exercise questions are free ground truth: one printed at the end of chapter 5
is answered by chapter 5. But nobody types them as printed, and an arm measured
on textbook wording looks far better than it will ever be in a student's hands:
BM25 scored 89.8% R@5 on the printed wording and 75.2% once the vocabulary was
stripped out. So the query set holds the reworded version only.
"""

import json
import random
import re
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ncert_rag.core.registry import BY_SLUG
from ncert_rag.services import llm
from ncert_rag.store import db

QUESTIONS_PATH = Path(__file__).parent / "questions.json"

PER_CHAPTER = 2  # keeps one long chapter from dominating the query set
SEED = 20260825
WORKERS = 3  # the proxy drops connections when pushed harder than this
MAX_CHARS = 300  # a student types a question, not an essay

# The paraphraser must not be a model any retrieval arm uses. When it was, each
# rewriter scored best on the questions its own family had written, so the query
# set was quietly grading the arms on whose vocabulary it happened to borrow.
PARAPHRASER = "ag/claude-sonnet-4-6"

_INSTRUCTIONS = (
    "Rewrite the textbook question the way a curious student would type it into "
    "a search box. Use everyday words: strip the technical terms and name the "
    "idea in plain language instead. It must stay a single question ending in a "
    "question mark, answerable from the same passage, under 30 words. Never "
    "answer it, never show working, never use code or headings. Even if the "
    "original says 'write a program' or 'solve', ask for it in plain words "
    "instead. Reply with the question only."
)

_MARKUP = re.compile(r"\*+")


def _candidates(conn: sqlite3.Connection) -> list[tuple[str, int, str]]:
    """Exercise questions with an unambiguous chapter label.

    Computer Science and Informatics Practices share whole chapters, so some
    questions appear verbatim under two books. Those cannot grade a retriever --
    either chapter is a defensible answer, so they are dropped rather than
    counted wrong.
    """
    rows = conn.execute(
        "SELECT book, chapter, question FROM exercises ORDER BY book, chapter, id"
    ).fetchall()

    owners = Counter()
    for row in rows:
        owners[row["question"]] += 1
    duplicated = {q for q, n in owners.items() if n > 1}

    return [
        (row["book"], row["chapter"], row["question"])
        for row in rows
        if row["question"] not in duplicated
    ]


def _sample(candidates: list[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    by_chapter: dict[tuple[str, int], list[tuple[str, int, str]]] = {}
    for item in candidates:
        by_chapter.setdefault((item[0], item[1]), []).append(item)

    rng = random.Random(SEED)
    picked = []
    for key in sorted(by_chapter):
        group = by_chapter[key]
        picked.extend(rng.sample(group, min(PER_CHAPTER, len(group))))
    return picked


def _usable(text: str) -> bool:
    """Reject anything that is not a plain question.

    Asked to reword "write a program that applies binary search", the model
    tends to write the program. Such answers carry textbook vocabulary and
    retrieve far too easily, inflating every arm.
    """
    return (
        text.endswith("?")
        and len(text) <= MAX_CHARS
        and "```" not in text
        and "\n" not in text
    )


def _paraphrase(questions: list[str]) -> list[str | None]:
    """None where the model failed or wrote something that is not a question."""
    agent = llm.agent(_INSTRUCTIONS, model=PARAPHRASER)

    def one(question: str) -> str | None:
        try:
            text = _MARKUP.sub("", llm.ask(agent, question)).strip()
        except Exception:
            return None
        return text if _usable(text) else None

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(one, questions))


def generate(regenerate: bool = False) -> None:
    """Build the query set, reusing existing wordings unless told otherwise.

    The model is not seeded, so rewording a question twice gives different
    text and two runs stop being comparable. Existing wordings are kept by
    default so a prompt tweak cannot silently invalidate the last numbers.
    """
    conn = db.connect()
    picked = _sample(_candidates(conn))
    books = len({book for book, _chapter, _q in picked})
    print(f"{len(picked)} exercise questions across {books} books")

    existing = load() if QUESTIONS_PATH.exists() else []
    kept = {} if regenerate else {q["id"]: q["question"] for q in existing}
    if kept:
        print(f"Reusing {len(kept)} existing wordings; pass --regenerate to redraw")

    todo = [
        (i, question)
        for i, (_b, _c, question) in enumerate(picked)
        if f"q{i:04d}" not in kept
    ]
    if todo:
        print(f"Rewording {len(todo)} with {PARAPHRASER}...")
    fresh = dict(
        zip(
            (i for i, _q in todo),
            _paraphrase([q for _i, q in todo]),
            strict=True,
        )
    )
    reworded = [kept.get(f"q{i:04d}") or fresh.get(i) for i in range(len(picked))]

    records = [
        {
            "id": f"q{i:04d}",
            "question": text,
            "book": book,
            "chapter": chapter,
            "tier": BY_SLUG[book].tier,
        }
        for i, ((book, chapter, _original), text) in enumerate(
            zip(picked, reworded, strict=True)
        )
        if text
    ]

    dropped = len(picked) - len(records)
    if dropped:
        print(f"{dropped} dropped: the model failed or did not return a question")

    QUESTIONS_PATH.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} questions to {QUESTIONS_PATH}")


def load() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text())


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="redraw every wording; results stop being comparable with the last run",
    )
    generate(regenerate=parser.parse_args().regenerate)
