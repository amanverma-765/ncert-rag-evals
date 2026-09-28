"""What each arm costs to serve, alongside what it retrieves.

Expansion pays a model call per query; the vector arms pay once at index time.
Recall alone cannot price that, so latency is measured here too.

Queries run one at a time: a thread pool would measure pool throughput rather
than the latency one student waits through.
"""

import argparse
import random
import statistics
import time
from dataclasses import dataclass

from evals.metrics import first_hit
from evals.questions import load
from evals.run import REPORT_PATH
from ncert_rag.retrieve import ARMS, expansion
from ncert_rag.store import db

SEED = 20260825
K = 10


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    recall: float
    median: float


def _measure(retriever, questions: list[dict]) -> tuple[list[float], float]:
    """Time each query and score it. Returns (latencies, R@5)."""
    times, ranks = [], []
    for question in questions:
        start = time.perf_counter()
        hits = retriever.retrieve(question["question"], K)
        times.append((time.perf_counter() - start) * 1000)
        ranks.append(
            first_hit(
                [(hit.book, hit.chapter) for hit in hits],
                (question["book"], question["chapter"]),
            )
        )
    found = sum(1 for rank in ranks if rank is not None and rank <= 5)
    return times, found / len(ranks)


def main(n: int) -> None:
    conn = db.connect()
    questions = load()
    rng = random.Random(SEED)
    sample = rng.sample(questions, min(n, len(questions)))

    print(f"Timing {len(sample)} queries per arm, serially...")
    rows = []
    for name, make in ARMS.items():
        # every arm starts cold, or the second expansion arm measured would
        # report the first one's cached rewrites as free
        expansion.clear_cache()
        latencies, recall = _measure(make(conn), sample)
        median = statistics.median(latencies)
        rows.append(Row(name=name, recall=recall, median=median))
        print(f"  {name}: R@5 {recall:.1%}, median {median:.1f} ms")

    _append_report(rows, len(sample))


def _append_report(rows: list[Row], n: int) -> None:
    content = REPORT_PATH.read_text()
    if "\n## Serving cost\n" in content:
        content = content.split("\n## Serving cost\n")[0]
    elif content.startswith("## Serving cost\n"):
        content = ""

    lines = [
        "",
        "## Serving cost",
        "",
        f"Accuracy and latency from the same {n} queries, run one at a time on "
        "CPU so the timings are what one student waits through rather than pool "
        "throughput.",
        "",
        "| arm | R@5 | median |",
        "|---|---|---|",
    ]
    lines += [
        f"| {row.name} | {row.recall:.1%} | {row.median:.1f} ms |" for row in rows
    ]
    lines += [
        "",
        "Latency is all this measures. What an arm costs in tokens -- the "
        "context it hands the answering model, and the rewrite on top for the "
        "expansion arms -- is no longer measured at all, so the price of a "
        "query cannot be read off this table. See EVALUATION.md §6.",
    ]

    REPORT_PATH.write_text(content.rstrip() + "\n" + "\n".join(lines) + "\n")
    print(f"\nAppended serving cost to {REPORT_PATH}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=80, help="queries to time")
    main(parser.parse_args().n)
