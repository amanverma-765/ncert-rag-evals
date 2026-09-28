# NCERT Retrieval Evaluation

Eight retrieval techniques measured across 17 NCERT science and mathematics textbooks to answer a single practical question: **what actually serves student queries, and what does it cost?**

This report analyzes the accuracy, latency, and statistical reality of different retrieval methods. The underlying data is generated dynamically via the evaluation pipeline (`uv run python -m evals.run`).

## 1. The Bottom Line

|  | Technique | R@5 | Median Latency |
| --- | --- | --- | --- |
| **Best Accuracy** | `expansion_hybrid` | **92.9%** | ~4.7 s |
| **Best Value** | `hybrid` | 86.9% | **8.9 ms** |
| Baseline | `bm25` | 75.5% | 3.5 ms |

Retrieval without a model at query time (`hybrid`) reaches ~87% accuracy. Adding an LLM rewrite step (`expansion`) adds roughly 6 points of accuracy, but costs several hundred times the latency to achieve.

---

## 2. Methodology Overview

* **Corpus:** 17 books (classes 10-12, English editions) spanning 169 chapters. Physics is currently excluded pending formula reconstruction.
* **Queries (n=282):** Derived from end-of-chapter exercise questions to provide free ground truth (a question printed in Chapter 5 is answered by Chapter 5). The wording is rewritten by an LLM (`ag/claude-sonnet-4-6`) to simulate natural student phrasing rather than perfect textbook vocabulary.
* **Metric:** **Recall@5 (R@5)**. Did any of the top 5 retrieved passages come from the correct chapter? (Top 5 is the standard context window passed to an answering model).

### Text Tiers

Books extract differently from PDFs, so results are categorized to prevent difficult edge cases from hiding in the average:

* **`clean` (n=239):** 13 non-math books. Text extracts as readable prose.
* **`fragmented` (n=43):** 4 math books. Equations shred into a scatter of one- and two-character fragments during PDF extraction.

---

## 3. The Eight Techniques

| Arm | Description |
| --- | --- |
| `bm25` | Keyword search (SQLite FTS5) over section-aware chunks. |
| `vector` | Dense search (cosine) over embeddings of the same section-aware chunks. |
| `vector_raw` | Dense search over fixed windows cut with **no** structural parsing. |
| `hybrid` | `bm25` + `vector` fused by reciprocal rank (RRF). |
| `expansion_*` | Four additional arms where an LLM (`ag/gemini-3.7-flash-medium`) rewrites the query into textbook vocabulary *before* searching via the base method. |

---

## 4. Accuracy Results

*Results reflect the full 282-question dataset. All model-free arms are reproducible; expansion arms may drift by ~2 points run-to-run due to unseeded LLM calls.*

| Arm | R@1 | R@5 | R@10 | MRR | n |
| --- | --- | --- | --- | --- | --- |
| **bm25** | 53.9% | 75.5% | 81.9% | 0.635 | 282 |
| **vector** | 59.6% | 83.3% | 90.8% | 0.698 | 282 |
| **vector_raw** | 58.9% | 84.4% | 89.7% | 0.693 | 282 |
| **hybrid** | 58.5% | 86.9% | 92.9% | 0.704 | 282 |
| **expansion_bm25** | 71.6% | 90.8% | 94.3% | 0.802 | 282 |
| **expansion_vector** | 71.3% | 90.8% | 94.3% | 0.800 | 282 |
| **expansion_raw** | 69.9% | 92.2% | 95.0% | 0.789 | 282 |
| **expansion_hybrid** | 72.7% | **92.9%** | 95.0% | 0.815 | 282 |

### Performance by Tier (R@5)

*Warning: At n=43 in the fragmented tier, a single question shifts the score by 2.3 points. Read this tier for directional signals, not precise percentages.*

| Tier | bm25 | hybrid | expansion_hybrid | n |
| --- | --- | --- | --- | --- |
| `clean` | 74.9% | 87.0% | 92.9% | 239 |
| `fragmented` | 79.1% | 86.0% | 93.0% | 43 |

---

## 5. Which Differences are Real (Paired Analysis)

Because every arm answers the same 282 questions, we compare them using **McNemar's test** (a paired test). This counts only the questions where two techniques *disagree* (shown as `wins/losses`). A high p-value combined with a wide interval means the dataset is too small to separate the arms, not that they perform identically.

| Comparison | R@5 Shift | Gap | Disagree | p-value | 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `hybrid` over `bm25` | 75.5% → 86.9% | +11.3 | 35/3 | <0.0001 | [+7.3, +15.4] | **Established** |
| `vector` over `bm25` | 75.5% → 83.3% | +7.8 | 42/20 | 0.0077 | [+2.4, +13.2] | **Established** |
| `expansion_hybrid` over `hybrid` | 86.9% → 92.9% | +6.0 | 22/5 | 0.0021 | [+2.5, +9.6] | **Established** |
| `hybrid` over `vector` | 83.3% → 86.9% | +3.5 | 22/12 | 0.1227 | [-0.5, +7.6] | Undecided |
| `vector` over `vector_raw` | 84.4% → 83.3% | -1.1 | 12/15 | 0.7003 | [-4.7, +2.5] | Undecided |

* **Embeddings earn their keep:** `vector` decisively beats `bm25` (+7.8 points, winning 42 tie-breakers to 20). BM25 fails on student queries because it requires exact keyword matches. When a student asks about feeling "tired after running fast", BM25 retrieves Computer Science chapters about "running loops fast", whereas dense vectors correctly map it to Biology's "anaerobic respiration".
* **Fusion rescues BM25:** Fusing dense retrieval *into* BM25 fixes BM25's blind spots (winning 35 tie-breakers to 3). However, this evaluation cannot confirm that fusion clearly improves dense retrieval by itself (+3.5 points, p=0.12).
* **Nothing justifies the parser (yet):** `vector` (parsed chunks) does not outperform `vector_raw` (blind windows). Splitting 12 wins to 15 losses with a wide confidence interval means this dataset cannot tell them apart. The parser might save token costs by preventing padded windows, but from a pure recall perspective, strict section boundaries do not improve accuracy here.
* **The rewrite is powerful, but expensive:** Appending textbook terms to a query via LLM (`expansion_hybrid`) yields the highest overall accuracy (+6.0 points). It is especially transformative for the mangled text in the `fragmented` (math) tier, though the math sample size is too small to declare victory definitively.

---

## 6. Serving Cost & Latency

*Measured serially over an 80-question sample. This measures **latency to serve**, not token cost.*

| Arm | R@5 | Median Latency |
| --- | --- | --- |
| **bm25** | 80.0% | 3.5 ms |
| **vector** | 85.0% | 4.9 ms |
| **vector_raw** | 83.8% | 4.8 ms |
| **hybrid** | 88.8% | **8.9 ms** |
| **expansion_bm25** | 90.0% | 4563.4 ms |
| **expansion_vector** | 93.8% | 4625.3 ms |
| **expansion_raw** | 93.8% | 4585.2 ms |
| **expansion_hybrid** | 96.2% | **4708.8 ms** |

The architectural gap is stark: model-free arms resolve in **single-digit milliseconds**, while placing an LLM call on the critical path pushes latency to **four to five seconds**.

---

## 7. Recommendations

1. **Ship `hybrid` as the default.** It delivers ~87% accuracy in under 10 milliseconds with no model dependency at query time. It is the best balance of cost and performance.
2. **Add `expansion_hybrid` behind a cache.** If you need to break the 90% accuracy ceiling, use the rewrite model. Because student questions repeat heavily, a cached rewrite path bypasses the 5-second latency penalty for common queries.
3. **Experiment with routing for math.** The expansion rewrite shows massive promise for repairing the broken equations in mathematics queries. Consider routing math queries exclusively through the expansion arm, but validate this on a larger sample first.

---

## 8. Crucial Implementation Traps

If you are modifying this pipeline, beware of these methodological pitfalls that will silently invalidate your results:

* **The Data Leak:** NCERT prints exercise questions *inside* the chapters. If you index these pages, the retriever just searches for the question's literal text instead of the educational explanation. This falsely spikes BM25 accuracy into the 90s. The pipeline specifically strips these pages (`ingest/pdf/parser.py`) to prevent this.
* **The Self-Grading Rewriter:** If the LLM used to generate the test queries is the same model family used in the `expansion` arms, that arm gets an unfair ~11-point advantage because it perfectly predicts its own vocabulary.
* **The Overzealous Rewriter:** When asking an LLM to rewrite a query for expansion, only ask for *search terms*. Asking the LLM to draft a hypothetical "answer passage" yields better recall, but pollutes the search with fabricated textbook prose that doesn't actually exist in the corpus.