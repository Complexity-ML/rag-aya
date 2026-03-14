"""
RAG-Aya :: Benchmark Visualizations

Generates research-quality figures from benchmark_results.json.

Figures produced:
  1. t-SNE — query + chunk embeddings in semantic space
  2. Per-query metric breakdown (bar chart)
  3. Category comparison: in-domain vs out-of-domain, English vs French
  4. Retrieval score vs keyword coverage scatter

Usage:
    python visualize.py                                    # uses benchmark_results.json
    python visualize.py --results benchmark_results.json --out figures/
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ── colour palette ─────────────────────────────────────────────────────────
C_ENG  = "#2196F3"   # blue
C_FRA  = "#FF9800"   # orange
C_IN   = "#4CAF50"   # green
C_OUT  = "#F44336"   # red
C_CLEAN = "#4CAF50"
C_DIRTY = "#F44336"
CMAP_SCORE = "RdYlGn"


def load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Figure 1: t-SNE query + chunk embeddings ───────────────────────────────

def figure_tsne(results: dict, index_path: str, embed_model: str, out_dir: str):
    """t-SNE of query embeddings + indexed chunk embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.manifold import TSNE
        import json as _json
    except ImportError:
        print("[visualize] sklearn or sentence-transformers not installed; skipping t-SNE")
        return

    queries = [r["query"] for r in results["results"]]
    labels  = [r["description"] for r in results["results"]]
    scores  = [r["score"] for r in results["results"]]
    langs   = [r["lang"] for r in results["results"]]
    domains = [r["in_domain"] for r in results["results"]]

    # Load chunk texts from index
    chunk_file = os.path.join(index_path, "chunks.json")
    chunk_texts, chunk_langs = [], []
    if os.path.exists(chunk_file):
        with open(chunk_file, encoding="utf-8") as f:
            chunks = _json.load(f)
        # Sample up to 80 chunks for clarity
        step = max(1, len(chunks) // 80)
        for c in chunks[::step]:
            chunk_texts.append(c["text"][:120])
            chunk_langs.append(c.get("language", "eng"))

    model = SentenceTransformer(embed_model)
    all_texts = queries + chunk_texts
    embeddings = model.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)

    n_queries = len(queries)
    n_total   = len(all_texts)

    # t-SNE — perplexity must be < n_samples
    perp = min(10, n_total - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor("#F9F9F9")
    fig.patch.set_facecolor("white")

    # --- Chunks (background dots) ---
    for i in range(n_queries, n_total):
        color = C_ENG if chunk_langs[i - n_queries] == "eng" else C_FRA
        ax.scatter(coords[i, 0], coords[i, 1], c=color, alpha=0.25,
                   s=18, zorder=1, edgecolors="none")

    # --- Queries (foreground, sized by score) ---
    cmap = plt.get_cmap(CMAP_SCORE)
    for i in range(n_queries):
        x, y = coords[i, 0], coords[i, 1]
        score = scores[i]
        color = cmap(score)
        marker = "o" if langs[i] == "eng" else "s"
        size   = 180 + 200 * score
        edge   = C_IN if domains[i] else C_OUT
        ax.scatter(x, y, c=[color], s=size, marker=marker,
                   edgecolors=edge, linewidths=2.2, zorder=3)
        # Short label (wrap at first parenthesis)
        short = labels[i].split("(")[0].strip()
        ax.annotate(
            short, (x, y),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

    # Colour bar for composite score
    sm = plt.cm.ScalarMappable(cmap=CMAP_SCORE, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Composite score", fontsize=9)

    # Legend
    legend_elements = [
        mpatches.Patch(color=C_ENG, alpha=0.6, label="English chunk"),
        mpatches.Patch(color=C_FRA, alpha=0.6, label="French chunk"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
                   markersize=9, markeredgecolor=C_IN, markeredgewidth=2, label="In-domain query"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#999",
                   markersize=9, markeredgecolor=C_OUT, markeredgewidth=2, label="Out-of-domain query"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
                   markersize=9, label="● = English query  ■ = French query"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.9)

    ax.set_title(
        "t-SNE: Query & Chunk Embeddings in Semantic Space\n"
        "(paraphrase-multilingual-MiniLM-L12-v2 · node size = composite score)",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.tick_params(labelsize=8)

    path = os.path.join(out_dir, "fig1_tsne.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")


# ── Figure 2: per-query metric breakdown ───────────────────────────────────

def figure_metrics(results: dict, out_dir: str):
    rows = results["results"]
    n = len(rows)
    descs  = [r["description"] for r in rows]
    retr   = [r["retrieval_score"] for r in rows]
    kw     = [r["kw_score"] for r in rows]
    lc     = [float(r["lang_correct"]) for r in rows]
    clean  = [float(r["is_clean"]) for r in rows]
    score  = [r["score"] for r in rows]
    langs  = [r["lang"] for r in rows]

    x = np.arange(n)
    w = 0.15

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_facecolor("#F9F9F9")

    bars = [
        ("Retrieval sim.", retr,  "#5C6BC0"),
        ("Keyword cov.",   kw,    "#26A69A"),
        ("Lang correct",   lc,    "#66BB6A"),
        ("Clean output",   clean, "#FFA726"),
        ("Composite ↑",    score, "#EF5350"),
    ]
    offsets = np.linspace(-(len(bars)-1)/2, (len(bars)-1)/2, len(bars)) * w

    for (label, vals, color), offset in zip(bars, offsets):
        ax.bar(x + offset, vals, width=w, label=label, color=color, alpha=0.85, zorder=2)

    # Language markers on x-axis
    xtick_labels = []
    for i, (d, lang) in enumerate(zip(descs, langs)):
        flag = "[EN]" if lang == "eng" else "[FR]"
        xtick_labels.append(f"{flag} {d}")

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score [0–1]", fontsize=9)
    ax.set_title(
        "Per-Query Metric Breakdown — RAG-Aya (tiny-aya Q4_K · WMT gold/finance corpus)",
        fontsize=11,
    )
    ax.axhline(0.8, color="#888", lw=0.8, ls="--", zorder=1)
    ax.legend(loc="upper right", fontsize=8, ncol=5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    path = os.path.join(out_dir, "fig2_metrics_per_query.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")


# ── Figure 3: category comparison ──────────────────────────────────────────

def figure_categories(results: dict, out_dir: str):
    rows = results["results"]

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    categories = {
        "In-domain":     [r for r in rows if r["in_domain"]],
        "Out-of-domain": [r for r in rows if not r["in_domain"]],
        "English":       [r for r in rows if r["lang"] == "eng"],
        "French":        [r for r in rows if r["lang"] == "fra"],
    }
    metric_keys = ["score", "retrieval_score", "kw_score", "lang_correct", "is_clean"]
    metric_labels = ["Composite", "Retrieval", "Keywords", "Lang OK", "Clean"]
    cat_colors = [C_IN, C_OUT, C_ENG, C_FRA]

    x = np.arange(len(metric_keys))
    w = 0.18
    n_cats = len(categories)
    offsets = np.linspace(-(n_cats-1)/2, (n_cats-1)/2, n_cats) * w

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#F9F9F9")

    for (cat, subset), offset, color in zip(categories.items(), offsets, cat_colors):
        vals = [avg([float(r[k]) for r in subset]) for k in metric_keys]
        bars = ax.bar(x + offset, vals, width=w, label=f"{cat} (n={len(subset)})",
                      color=color, alpha=0.82, zorder=2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Average score [0–1]", fontsize=9)
    ax.set_title(
        "Category Comparison — In-domain vs Out-of-domain · English vs French",
        fontsize=11,
    )
    ax.axhline(0.8, color="#888", lw=0.8, ls="--", zorder=1)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    path = os.path.join(out_dir, "fig3_category_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")


# ── Figure 4: retrieval vs keyword scatter ─────────────────────────────────

def figure_scatter(results: dict, out_dir: str):
    rows = results["results"]

    retr   = np.array([r["retrieval_score"] for r in rows])
    kw     = np.array([r["kw_score"] for r in rows])
    score  = np.array([r["score"] for r in rows])
    lat    = np.array([r["latency_s"] for r in rows])
    descs  = [r["description"] for r in rows]
    langs  = [r["lang"] for r in rows]
    domain = [r["in_domain"] for r in rows]

    cmap = plt.get_cmap(CMAP_SCORE)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#F9F9F9")

    for i in range(len(rows)):
        marker = "o" if langs[i] == "eng" else "s"
        edge   = "#2E7D32" if domain[i] else "#C62828"
        size   = 60 + lat[i] / 4
        ax.scatter(retr[i], kw[i], c=[cmap(score[i])], s=size, marker=marker,
                   edgecolors=edge, linewidths=1.8, zorder=3, alpha=0.9)
        short = descs[i].split("(")[0].strip()
        ax.annotate(short, (retr[i], kw[i]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color="#444",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.65, ec="none"))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP_SCORE, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Composite score", fontsize=9)

    # Reference lines
    ax.axhline(0.5, color="#aaa", lw=0.8, ls="--")
    ax.axvline(0.5, color="#aaa", lw=0.8, ls="--")

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
                   markeredgecolor="#2E7D32", markeredgewidth=2, markersize=9, label="In-domain"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#999",
                   markeredgecolor="#C62828", markeredgewidth=2, markersize=9, label="Out-of-domain"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
                   markersize=8, label="● English  ■ French"),
        mpatches.Patch(color="#ccc", label="Node size ∝ latency (s)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, framealpha=0.9)

    ax.set_xlabel("Retrieval similarity score", fontsize=10)
    ax.set_ylabel("Keyword coverage", fontsize=10)
    ax.set_title(
        "Retrieval Quality vs Answer Keyword Coverage\n(node size ∝ generation latency)",
        fontsize=11,
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.35, zorder=0)

    path = os.path.join(out_dir, "fig4_retrieval_vs_keywords.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")


# ── Figure 5: emoji profile ────────────────────────────────────────────────

def count_emojis(text: str):
    """Return list of individual emoji characters found in text."""
    # Unicode ranges covering most emoji blocks
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Misc symbols & pictographs
        (0x1F680, 0x1F6FF),  # Transport & map
        (0x1F700, 0x1F77F),  # Alchemical
        (0x1F900, 0x1F9FF),  # Supplemental symbols
        (0x1FA00, 0x1FA6F),  # Chess symbols etc.
        (0x1FA70, 0x1FAFF),  # Symbols extended-A
        (0x2600,  0x26FF),   # Misc symbols
        (0x2700,  0x27BF),   # Dingbats
        (0x231A,  0x231B),   # Watch / hourglass
        (0x23E9,  0x23F3),   # Other clock symbols
    ]
    found = []
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in emoji_ranges):
            found.append(ch)
    return found


def figure_emoji(results: dict, out_dir: str):
    """Bar chart: emoji count per query + emoji inventory per response."""
    rows = results["results"]
    descs  = [r["description"] for r in rows]
    langs  = [r["lang"] for r in rows]

    all_emojis = [count_emojis(r["response"]) for r in rows]
    counts     = [len(e) for e in all_emojis]

    # Unique emoji inventory across all responses
    from collections import Counter
    from matplotlib import font_manager
    total_counter = Counter()
    for emojis in all_emojis:
        total_counter.update(emojis)

    # Find an emoji-capable font (Segoe UI Emoji on Windows, fallbacks for other OS)
    emoji_font = None
    for fname in ["Segoe UI Emoji", "Apple Color Emoji", "Noto Emoji", "Symbola"]:
        try:
            prop = font_manager.FontProperties(family=fname)
            if font_manager.findfont(prop, fallback_to_default=False):
                emoji_font = prop
                break
        except Exception:
            pass

    fig = plt.figure(figsize=(14, 6), layout="constrained")
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Left: bar chart per query
    colors = [C_ENG if l == "eng" else C_FRA for l in langs]
    x = np.arange(len(rows))
    bars = ax1.bar(x, counts, color=colors, alpha=0.82, zorder=2)

    for bar, emojis in zip(bars, all_emojis):
        if emojis:
            sample = "".join(dict.fromkeys(emojis))[:6]  # up to 6 unique
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                sample,
                ha="center", va="bottom", fontsize=11,
                fontproperties=emoji_font if emoji_font else None,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{'[EN]' if l == 'eng' else '[FR]'} {d}" for d, l in zip(descs, langs)],
        rotation=30, ha="right", fontsize=8,
    )
    ax1.set_ylabel("Emoji count in response", fontsize=9)
    ax1.set_title("Emoji Generation per Query", fontsize=11)
    ax1.grid(axis="y", alpha=0.4, zorder=0)
    ax1.set_facecolor("#F9F9F9")

    legend_elements = [
        mpatches.Patch(color=C_ENG, alpha=0.82, label="English query"),
        mpatches.Patch(color=C_FRA, alpha=0.82, label="French query"),
    ]
    ax1.legend(handles=legend_elements, fontsize=9)

    # Right: top-N emoji inventory
    top = total_counter.most_common(20)
    if top:
        emojis_list, freq_list = zip(*top)
        y = np.arange(len(top))
        ax2.barh(y, freq_list, color="#7E57C2", alpha=0.82, zorder=2)
        ax2.set_yticks(y)
        ax2.set_yticklabels(emojis_list, fontsize=14,
                            fontproperties=emoji_font if emoji_font else None)
        ax2.set_xlabel("Occurrences", fontsize=9)
        ax2.set_title(f"Emoji Inventory\n({sum(counts)} total, {len(total_counter)} unique)", fontsize=11)
        ax2.grid(axis="x", alpha=0.4, zorder=0)
        ax2.set_facecolor("#F9F9F9")
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "No emojis found", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="#888")
        ax2.set_title("Emoji Inventory", fontsize=11)

    fig.suptitle(
        "Aya Tiny (Q4_K) — Emoji Expressiveness Profile",
        fontsize=12, fontweight="bold",
    )

    path = os.path.join(out_dir, "fig5_emoji_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG-Aya Benchmark Visualizations")
    parser.add_argument("--results", default="benchmark_results.json")
    parser.add_argument("--out", default="figures/")
    parser.add_argument("--index-path", default="index/")
    parser.add_argument("--embed-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE (requires sklearn)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    results = load_results(args.results)
    print(f"[visualize] Loaded {results['n_queries']} results from {args.results}")

    if not args.skip_tsne:
        figure_tsne(results, args.index_path, args.embed_model, args.out)

    figure_metrics(results, args.out)
    figure_categories(results, args.out)
    figure_scatter(results, args.out)
    figure_emoji(results, args.out)

    print(f"[visualize] All figures saved to {args.out}")


if __name__ == "__main__":
    main()
