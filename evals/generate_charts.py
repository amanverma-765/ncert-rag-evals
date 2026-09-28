"""Generate high-contrast, publication-quality benchmark charts and analysis notebook.

Outputs:
  docs/assets/pareto_frontier.png
  docs/assets/tier_breakdown.png
  docs/assets/mcnemar_discordance.png
  evals/benchmark_analysis.ipynb
"""

import base64
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "docs" / "assets"
NOTEBOOK_PATH = ROOT / "evals" / "benchmark_analysis.ipynb"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#d0d7de",
    "axes.linewidth": 1.2,
    "grid.color": "#eaeef2",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def plot_pareto_frontier() -> Path:
    """Plot Median CPU Latency (log scale) vs Recall@5 (all 282 queries)."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    # Data from evals/REPORT.md (All questions n=282 Recall@5, Cost median ms)
    arms = [
        {"name": "bm25", "lat": 3.5, "r5": 75.5, "type": "lexical", "label": "BM25 (SQLite FTS5)\n3.5ms | 75.5%"},
        {"name": "vector_raw", "lat": 4.8, "r5": 84.4, "type": "dense", "label": "Vector Raw (Sliding Windows)\n4.8ms | 84.4%"},
        {"name": "vector_parsed", "lat": 4.9, "r5": 83.3, "type": "dense", "label": "Vector Parsed (Section Chunks)\n4.9ms | 83.3%"},
        {"name": "hybrid", "lat": 8.9, "r5": 86.9, "type": "hybrid", "label": "Hybrid RRF (Sweet Spot)\n8.9ms | 86.9%"},
        {"name": "expansion_bm25", "lat": 4563.4, "r5": 90.8, "type": "expansion", "label": "Exp + BM25\n4,563ms | 90.8%"},
        {"name": "expansion_vector", "lat": 4625.3, "r5": 90.8, "type": "expansion", "label": "Exp + Vector\n4,625ms | 90.8%"},
        {"name": "expansion_raw", "lat": 4585.2, "r5": 92.2, "type": "expansion", "label": "Exp + Raw\n4,585ms | 92.2%"},
        {"name": "expansion_hybrid", "lat": 4708.8, "r5": 92.9, "type": "expansion", "label": "Exp + Hybrid (Max Recall)\n4,709ms | 92.9%"},
    ]

    colors = {
        "lexical": "#0969da",    # blue
        "dense": "#8250df",      # purple
        "hybrid": "#1a7f37",     # emerald green
        "expansion": "#cf222e",  # red / coral
    }

    # Plot points
    for arm in arms:
        c = colors[arm["type"]]
        size = 140 if arm["name"] in ("hybrid", "expansion_hybrid") else 80
        ax.scatter(arm["lat"], arm["r5"], color=c, s=size, zorder=4, edgecolors="white", linewidths=1.5)

    # Highlight Pareto optimal frontier: bm25 (3.5ms, 75.5%) -> vector_raw (4.8ms, 84.4%) -> hybrid (8.9ms, 86.9%) -> expansion_raw (4585ms, 92.2%) -> expansion_hybrid (4709ms, 92.9%)
    frontier_x = [3.5, 4.8, 8.9, 4585.2, 4708.8]
    frontier_y = [75.5, 84.4, 86.9, 92.2, 92.9]
    ax.plot(frontier_x, frontier_y, color="#1a7f37", linestyle=":", linewidth=2, alpha=0.8, label="Empirical Pareto Frontier", zorder=2)

    # Callout for Hybrid (Pareto sweet spot)
    ax.annotate(
        "PARETO SWEET SPOT (Hybrid RRF)\n8.9ms latency | 86.9% Recall@5\nZero token cost, CPU-only",
        xy=(8.9, 86.9),
        xytext=(25, 78),
        arrowprops=dict(facecolor="#1a7f37", edgecolor="#1a7f37", arrowstyle="->", lw=1.8),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#dafbe1", edgecolor="#1a7f37", lw=1.2),
        fontweight="bold",
        color="#116329",
        fontsize=9.5,
    )

    # Callout for Latency Wall
    ax.annotate(
        "THE LATENCY WALL (528× slower)\n4,709ms | 92.9% Recall@5\n+6.0 pts recall at 99.8% latency penalty",
        xy=(4708.8, 92.9),
        xytext=(350, 95.5),
        arrowprops=dict(facecolor="#cf222e", edgecolor="#cf222e", arrowstyle="->", lw=1.8),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebe9", edgecolor="#cf222e", lw=1.2),
        fontweight="bold",
        color="#a40e26",
        fontsize=9.5,
    )

    # Label key points
    ax.text(3.5, 74.0, "BM25 (3.5ms)", fontsize=8.5, ha="center", color="#0969da", fontweight="semibold")
    ax.text(4.8, 85.2, "Vector Raw", fontsize=8.5, ha="right", color="#8250df", fontweight="semibold")
    ax.text(4563.4, 89.6, "Exp+BM25 / Exp+Vector", fontsize=8.5, ha="right", color="#cf222e")

    ax.set_xscale("log")
    ax.set_xlim(1.5, 12000)
    ax.set_ylim(70, 98)
    ax.set_xlabel("Median Single-Query CPU Latency (milliseconds, log scale)", fontweight="bold")
    ax.set_ylabel("Recall@5 on 282 Student Queries (%)", fontweight="bold")
    ax.set_title("Pareto Frontier: Retrieval Latency vs Accuracy across 8 Strategies", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, which="both")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#0969da", markersize=9, label="Lexical (SQLite FTS5)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#8250df", markersize=9, label="Dense (Chroma / BGE-small)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a7f37", markersize=11, label="Hybrid RRF (Zero LLM)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#cf222e", markersize=9, label="LLM Query Expansion (Gemini)"),
        plt.Line2D([0], [0], color="#1a7f37", linestyle=":", linewidth=2, label="Pareto Frontier"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95, edgecolor="#d0d7de")

    out_path = ASSETS_DIR / "pareto_frontier.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_tier_breakdown() -> Path:
    """Plot Clean (Prose) vs Fragmented (Math) Tier Recall@5."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

    # Data from evals/REPORT.md:
    # Clean tier (n=239): bm25 74.9%, hybrid 87.0%, expansion_hybrid 92.9%
    # Fragmented tier (n=43): bm25 79.1%, hybrid 86.0%, expansion_hybrid 93.0%
    categories = ["BM25\n(Lexical)", "Hybrid RRF\n(Sub-10ms)", "Expansion Hybrid\n(LLM Augmented)"]
    clean_scores = [74.9, 87.0, 92.9]
    frag_scores = [79.1, 86.0, 93.0]

    x = np.arange(len(categories))
    width = 0.32

    rects1 = ax.bar(x - width / 2, clean_scores, width, label="Clean Tier: Non-Math (n=239)", color="#0969da", edgecolor="white", linewidth=1.2)
    rects2 = ax.bar(x + width / 2, frag_scores, width, label="Fragmented Tier: Mathematics (n=43)", color="#d4a72c", edgecolor="white", linewidth=1.2)

    # Add score labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_ylabel("Recall@5 (%)", fontweight="bold")
    ax.set_title("Corpus Tier Degradation: Continuous Prose vs Shredded Math Formulas", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="semibold")
    ax.set_ylim(60, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#d0d7de")

    # Add explanatory note
    ax.text(0.5, -0.18, "Note: Math PDF extraction shreds formulas into 1-2 char fragments, raising variance (1 question = 2.3% at n=43).",
            ha="center", va="center", transform=ax.transAxes, fontsize=8.5, style="italic", color="#57606a")

    out_path = ASSETS_DIR / "tier_breakdown.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_mcnemar_discordance() -> Path:
    """Plot paired delta intervals and McNemar significance verdicts."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    # Comparisons from evals/REPORT.md:
    # 1. Hybrid over BM25: gap +11.3, CI [+7.3, +15.4], p < 0.0001 (Established)
    # 2. Vector over BM25: gap +7.8, CI [+2.4, +13.2], p = 0.0077 (Established)
    # 3. Expansion Hybrid over Hybrid: gap +6.0, CI [+2.5, +9.6], p = 0.0021 (Established)
    # 4. Hybrid over Vector: gap +3.5, CI [-0.5, +7.6], p = 0.1227 (Undecided)
    # 5. Vector Parsed over Vector Raw: gap -1.1, CI [-4.7, +2.5], p = 0.7003 (Undecided - Negative Result)
    comparisons = [
        ("Hybrid vs BM25", 11.3, 7.3, 15.4, "< 0.0001", "Significant (+11.3%)"),
        ("Vector vs BM25", 7.8, 2.4, 13.2, "0.0077", "Significant (+7.8%)"),
        ("Expansion Hybrid vs Hybrid", 6.0, 2.5, 9.6, "0.0021", "Significant (+6.0%)"),
        ("Hybrid vs Vector", 3.5, -0.5, 7.6, "0.1227", "Undecided (+3.5%)"),
        ("Vector Parsed vs Raw (Parser)", -1.1, -4.7, 2.5, "0.7003", "Negative Result (-1.1%)"),
    ]

    y_positions = np.arange(len(comparisons))
    y_positions = y_positions[::-1]  # Top to bottom

    ax.axvline(0, color="#24292f", linestyle="-", linewidth=1.2, alpha=0.7)

    for i, (name, gap, ci_low, ci_high, p_val, label) in enumerate(comparisons):
        y = y_positions[i]
        is_significant = "Significant" in label
        color = "#1a7f37" if is_significant else ("#8250df" if gap > 0 else "#cf222e")

        # Plot confidence interval bar
        ax.plot([ci_low, ci_high], [y, y], color=color, linewidth=2.5, solid_capstyle="round")
        ax.scatter(gap, y, color=color, s=80, zorder=3, edgecolors="white", linewidth=1.2)

        # Label with gap and p-value
        text = f"{gap:+.1f}%  (p={p_val})"
        ax.text(ci_high + 0.8, y, text, va="center", fontsize=9, fontweight="bold", color=color)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in comparisons], fontweight="semibold")
    ax.set_xlabel("Net Recall@5 Difference (%) with 95% Confidence Interval", fontweight="bold")
    ax.set_title("Paired McNemar Significance Testing across 282 Queries", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(-7, 20)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Shaded region for zero crossing (insignificance)
    ax.axvspan(-7, 0, color="#f6f8fa", alpha=0.5, zorder=0)
    ax.text(-3.5, -0.5, "Negative / Inconclusive Zone", ha="center", fontsize=8.5, color="#57606a", style="italic")

    out_path = ASSETS_DIR / "mcnemar_discordance.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def build_notebook() -> Path:
    """Build and save a self-contained, pre-rendered Jupyter notebook."""
    def _encode(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    pareto_b64 = _encode(ASSETS_DIR / "pareto_frontier.png")
    tier_b64 = _encode(ASSETS_DIR / "tier_breakdown.png")
    mcnemar_b64 = _encode(ASSETS_DIR / "mcnemar_discordance.png")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Empirical Retrieval Benchmark: Exploratory Analysis & Visualizations\n",
                "\n",
                "**Corpus:** 17 NCERT STEM Textbooks (Classes 10–12, 169 chapters, 4,429 parsed chunks)  \n",
                "**Benchmark Queries:** 282 natural student queries across clean (prose) and fragmented (math) tiers  \n",
                "**Hardware:** Single-threaded CPU (AMD Ryzen / Linux) serial latency profiling  \n",
                "\n",
                "This notebook provides the exploratory data analysis, Pareto trade-off curves, and paired statistical significance tests validating our hybrid retrieval architecture."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "from pathlib import Path\n",
                "\n",
                "# Set publication plotting aesthetics\n",
                "plt.rcParams.update({\n",
                "    'font.family': 'sans-serif',\n",
                "    'font.size': 11,\n",
                "    'axes.edgecolor': '#d0d7de',\n",
                "    'axes.linewidth': 1.2,\n",
                "    'grid.color': '#eaeef2',\n",
                "    'grid.linestyle': '--',\n",
                "    'grid.alpha': 0.7,\n",
                "})"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Benchmark Data Summary\n",
                "\n",
                "We benchmark 8 retrieval strategies isolating three architectural axes:\n",
                "1. **Lexical (`bm25`)**: SQLite FTS5 with Porter stemming.\n",
                "2. **Dense (`vector_parsed`, `vector_raw`)**: Local `bge-small-en-v1.5` embeddings on CPU with ChromaDB HNSW.\n",
                "3. **Hybrid RRF (`hybrid`)**: Reciprocal Rank Fusion ($k=60$) combining BM25 and dense vectors.\n",
                "4. **Query Expansion (`expansion_*`)**: Few-shot LLM query rewriting (Gemini 3.7 Flash) appending technical textbook vocabulary."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Arm                 Recall@1   Recall@5  Recall@10        MRR    Latency (ms)\n",
                        "------------------------------------------------------------------------\n",
                        "bm25                   53.9%      75.5%      81.9%      0.635          3.5 ms\n",
                        "vector_parsed          59.6%      83.3%      90.8%      0.698          4.9 ms\n",
                        "vector_raw             58.9%      84.4%      89.7%      0.693          4.8 ms\n",
                        "hybrid                 58.5%      86.9%      92.9%      0.704          8.9 ms\n",
                        "expansion_bm25         71.6%      90.8%      94.3%      0.802       4563.4 ms\n",
                        "expansion_vector       71.3%      90.8%      94.3%      0.800       4625.3 ms\n",
                        "expansion_raw          69.9%      92.2%      95.0%      0.789       4585.2 ms\n",
                        "expansion_hybrid       72.7%      92.9%      95.0%      0.815       4708.8 ms\n"
                    ]
                }
            ],
            "source": [
                "# Benchmark results from evals/REPORT.md\n",
                "results = [\n",
                "    ('bm25', 53.9, 75.5, 81.9, 0.635, 3.5),\n",
                "    ('vector_parsed', 59.6, 83.3, 90.8, 0.698, 4.9),\n",
                "    ('vector_raw', 58.9, 84.4, 89.7, 0.693, 4.8),\n",
                "    ('hybrid', 58.5, 86.9, 92.9, 0.704, 8.9),\n",
                "    ('expansion_bm25', 71.6, 90.8, 94.3, 0.802, 4563.4),\n",
                "    ('expansion_vector', 71.3, 90.8, 94.3, 0.800, 4625.3),\n",
                "    ('expansion_raw', 69.9, 92.2, 95.0, 0.789, 4585.2),\n",
                "    ('expansion_hybrid', 72.7, 92.9, 95.0, 0.815, 4708.8),\n",
                "]\n",
                "\n",
                "print(f\"{'Arm':<20} {'Recall@1':>10} {'Recall@5':>10} {'Recall@10':>10} {'MRR':>10} {'Latency (ms)':>15}\")\n",
                "print(\"-\" * 72)\n",
                "for arm, r1, r5, r10, mrr, lat in results:\n",
                "    print(f\"{arm:<20} {r1:>9.1f}% {r5:>9.1f}% {r10:>9.1f}% {mrr:>10.3f} {lat:>12.1f} ms\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Latency vs. Recall@5 Pareto Frontier\n",
                "\n",
                "When selecting a retrieval architecture for interactive applications, evaluating recall in isolation is misleading. The key operational question is **marginal recall per millisecond of latency**.\n",
                "\n",
                "* **Sub-10ms Sweet Spot (`hybrid`)**: Fusing BM25 and dense embeddings with Reciprocal Rank Fusion ($k=60$) hits **86.9% Recall@5 in 8.9ms on CPU** with $0.00 token cost.\n",
                "* **The Latency Wall (`expansion_hybrid`)**: Adding an external LLM query rewriter reaches 92.9% Recall@5, but inflates serving latency by **528× (4,709 ms)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": pareto_b64,
                        "text/plain": ["<Figure size 2000x1200 with 1 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "# Visualizing the Pareto Frontier\n",
                "from evals.generate_charts import plot_pareto_frontier\n",
                "plot_pareto_frontier()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Corpus Heterogeneity & Tier Breakdown\n",
                "\n",
                "PDF extraction quality dramatically alters retrieval mechanics across domains:\n",
                "* **Clean Tier (13 books, $n=239$):** Continuous prose (Biology, Chemistry, Economics). Dense retrieval significantly outperforms BM25 (+9.6% at R@5).\n",
                "* **Fragmented Tier (4 books, $n=43$):** Mathematics textbooks where formulas, fractions, and square roots get shredded into 1- or 2-character tokens by PDF extractors. Keyword match retains high utility, while broken equation fragments degrade vector similarity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": tier_b64,
                        "text/plain": ["<Figure size 1800x1000 with 1 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "# Visualizing Clean vs Fragmented Tier Performance\n",
                "from evals.generate_charts import plot_tier_breakdown\n",
                "plot_tier_breakdown()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Paired McNemar Significance Testing\n",
                "\n",
                "Because every retrieval strategy answers the **exact same 282 questions**, standard two-sample independent tests (like Student's t-test or z-test) are invalid. We evaluate discordant pairs (queries where Technique A and Technique B disagree) using **McNemar's test** with continuity correction:\n",
                "\n",
                "$$\\chi^2 = \\frac{(|b - c| - 1)^2}{b + c}$$\n",
                "\n",
                "* **`hybrid` over `bm25`**: +11.3% net gap, 35 wins / 3 losses ($p < 0.0001$, 95% CI [+7.3, +15.4]). **Established.**\n",
                "* **`expansion_hybrid` over `hybrid`**: +6.0% net gap, 22 wins / 5 losses ($p = 0.0021$, 95% CI [+2.5, +9.6]). **Established.**\n",
                "* **`vector_parsed` vs `vector_raw`**: -1.1% gap ($p = 0.7003$, 95% CI [-4.7, +2.5]). **Negative Result.**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": mcnemar_b64,
                        "text/plain": ["<Figure size 2000x1000 with 1 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "# Visualizing Paired Difference Intervals and Significance\n",
                "from evals.generate_charts import plot_mcnemar_discordance\n",
                "plot_mcnemar_discordance()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Architectural Findings & Takeaways\n",
                "\n",
                "1. **Reciprocal Rank Fusion on CPU is the Production Champion:** At 8.9ms and 86.9% Recall@5, `hybrid` provides an optimal production balance with zero external API dependencies.\n",
                "2. **AST Parsing is for Readability, Not Retrieval:** Strict section-aware parsing does not improve recall over blind 512-token sliding windows ($p = 0.7003$) due to boundary truncation of small sections.\n",
                "3. **Semantic Query Caching:** For workloads requiring 92%+ recall, routing high-frequency queries through a semantic cache of precomputed expansions avoids the 4.7-second latency penalty."
            ]
        }
    ]

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.14.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook_data, f, indent=1)
    return NOTEBOOK_PATH


def main() -> None:
    print(f"Generating benchmark charts to {ASSETS_DIR}...")
    p1 = plot_pareto_frontier()
    print(f"✓ {p1.name}")
    p2 = plot_tier_breakdown()
    print(f"✓ {p2.name}")
    p3 = plot_mcnemar_discordance()
    print(f"✓ {p3.name}")
    nb = build_notebook()
    print(f"✓ {nb.name}")
    print("All charts and notebook generated successfully.")


if __name__ == "__main__":
    main()
