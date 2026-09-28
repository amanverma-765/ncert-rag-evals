# ncert-rag

Retrieval over NCERT science and mathematics textbooks (classes 10-12), built to answer one question with measurements rather than assumptions: **what actually serves student queries, and what does it cost?**

The corpus is small, the vocabulary is technical, and every chapter arrives as its own PDF with its structure intact. Because these facts strongly favour plain full-text search, this repo ships eight retrieval techniques over one shared corpus and measures them against 282 real exercise questions to see if embeddings and LLMs are actually necessary.

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
uv sync                              # create .venv, install dependencies
uv run ncert-rag build               # download, parse, and index 17 books (~10 mins)
uv run ncert-rag search "why do we feel tired after running fast" --arm hybrid

```

*Note: `build` is idempotent; books with unchanged PDF hashes are skipped.*

## The Bottom Line

|  | Technique | R@5 | Median Latency |
| --- | --- | --- | --- |
| **Best Accuracy** | `expansion_hybrid` | 92.9% | ~4.7 s |
| **Recommended** | `hybrid` | 86.9% | 8.9 ms |
| **Baseline** | `bm25` | 75.5% | 3.5 ms |

**The takeaway:** Retrieval with no model at query time (`hybrid`) reaches ~87% accuracy in milliseconds. Putting an LLM in front of it (`expansion_hybrid`) adds roughly 6 points of accuracy, but costs several hundred times the latency.

## The Eight Techniques

| Arm | Description |
| --- | --- |
| `bm25` | Keyword search (SQLite FTS5) over section-aware chunks. |
| `vector` | Dense search over embeddings of the same chunks. |
| `vector_raw` | Dense search over fixed windows cut with **no** parsing. |
| `hybrid` | `bm25` + `vector`, fused by reciprocal rank (RRF). |
| `expansion_bm25` | LLM rewrites the query into textbook vocabulary, then `bm25`. |
| `expansion_vector` | Same LLM rewrite → `vector`. |
| `expansion_raw` | Same LLM rewrite → `vector_raw`. |
| `expansion_hybrid` | Same LLM rewrite → `hybrid`. |

*Note: `vector` vs `vector_raw` isolates the value of the structural parser. The `expansion_*` family isolates the value of an LLM query rewrite.*

## Full Evaluation Results

Scored over 282 questions. Tests fail if the code and these numbers ever disagree (see [`evals/REPORT.md`](https://www.google.com/search?q=evals/REPORT.md)).

| Arm | R@1 | R@5 | R@10 | MRR | n |
| --- | --- | --- | --- | --- | --- |
| **bm25** | 53.9% | 75.5% | 81.9% | 0.635 | 282 |
| **vector** | 59.6% | 83.3% | 90.8% | 0.698 | 282 |
| **vector_raw** | 58.9% | 84.4% | 89.7% | 0.693 | 282 |
| **hybrid** | 58.5% | 86.9% | 92.9% | 0.704 | 282 |
| **expansion_bm25** | 71.6% | 90.8% | 94.3% | 0.802 | 282 |
| **expansion_vector** | 71.3% | 90.8% | 94.3% | 0.800 | 282 |
| **expansion_raw** | 69.9% | 92.2% | 95.0% | 0.789 | 282 |
| **expansion_hybrid** | 72.7% | 92.9% | 95.0% | 0.815 | 282 |

### The Two Text Tiers

Not all books extract equally well from PDFs. Results are split to prevent the hardest parsing challenges from hiding inside the average:

* **`clean` (13 non-math books):** Extracts as readable prose. Chunks read like the page they came from.
* **`fragmented` (4 math books):** Equations (superscripts, fractions, symbols) shred into one- and two-character fragments during PDF extraction.

**R@5 for the headline arms by tier:**

| Tier | bm25 | hybrid | expansion_hybrid | n |
| --- | --- | --- | --- | --- |
| `clean` | 74.9% | 87.0% | 92.9% | 239 |
| `fragmented` | 79.1% | 86.0% | 93.0% | 43 |

*Note: Read the fragmented row with counts, not percentages. At n=43, a single question moves the number by 2.3 points.*

### Which Differences are Real?

Because every technique answers the same 282 questions, claims are tested using **paired tests**—counting only the questions two techniques disagree on.

| Claim | Gap at R@5 | p-value | 95% CI | Verdict |
| --- | --- | --- | --- | --- |
| Fusion beats BM25 alone | +11.3 | <0.0001 | [+7.3, +15.4] | **Established** |
| Embeddings beat BM25 | +7.8 | 0.0077 | [+2.4, +13.2] | **Established** |
| LLM rewrite beats its base | +6.0 | 0.0021 | [+2.5, +9.6] | **Established** |
| Fusion beats dense alone | +3.5 | 0.12 | [-0.5, +7.6] | Undecided |
| Parser improves recall | **-1.1** | 0.70 | [-4.7, +2.5] | Undecided |

## Architecture & Layout

Dependencies flow one way: `core` ← `ingest` → `store` ← `retrieve` ← `evals`.

* **`core/`**: Pydantic models and the book registry (no internal imports).
* **`ingest/`**: Pipeline to download, clean, parse, and chunk PDFs. Subdivided into `ingest/pdf/` (layout extraction & structural parsing) and `ingest/text/` (normalization & token chunking).
* **`store/`**: SQLite schema (FTS5 index) and Chroma collections.
* **`retrieve/`**: The eight arms behind one `Retriever` protocol.
* **`services/`**: Chat and embedding model integrations.
* **`evals/`**: Query-set generation, metrics, runner, and cost benchmarking.

### The Storage Model

* **Text & Index:** `data/corpus.db` (SQLite)
* **Embeddings:** `data/chroma` (7,984 embeddings, one collection per chunk set)
* *Warning:* During a rebuild, a book's vectors are deleted **before** its chunk rows to prevent SQLite from reassigning freed IDs to new books and serving mismatched chapters.

## Reproducing the Numbers

The 282 ground-truth queries are pre-generated and stored in `evals/questions.json`, so you do **not** need to regenerate them every run.

The four `expansion_*` arms require an LLM. Ensure your local 9router proxy is running (`http://localhost:20128/v1`) and `NINEROUTER_API_KEY` is set in `.env`.

```bash
uv run python -m evals.run         # score every arm over questions.json -> evals/REPORT.md
uv run python -m evals.cost        # latency benchmark -> appended to REPORT.md
uv run pytest                      # run tests
uv run ruff check                  # lint code

```

To run offline without LLMs (evaluating model-free arms directly):

```bash
uv run python -m evals.run --arms bm25,vector,vector_raw,hybrid --force

```

*(Optional: only if you change the book catalog or want to redraw query wording with an LLM, run `uv run python -m evals.questions`)*

## ⚠️ Crucial Warning: The Exercise Data Leak

**Do not index the exercise pages.** NCERT prints the exercise questions inside the chapter. Because our evaluation queries come from those exact exercises, leaving them in the index rewards the retriever for finding the *question itself* rather than the *explanation*.

If left in, this leak inflates BM25's R@1 into the 90s and completely inverts the ranking of every arm. `ingest/pdf/parser.py` exists solely to strip these pages out. Do not remove it.