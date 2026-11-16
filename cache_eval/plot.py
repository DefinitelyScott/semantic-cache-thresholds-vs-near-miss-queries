"""Draw the one figure in the README: similarity by kind of incoming query."""

from __future__ import annotations

from pathlib import Path

from cache_eval.experiment import ScoredTriple, summarize

KINDS = (
    ("paraphrase", "paraphrase\n(should be served)", "#2a7ab9"),
    ("near_miss", "near-miss\n(should not)", "#d1495b"),
    ("unrelated", "unrelated\n(should not)", "#7a7a7a"),
)


def plot_scores(measure: str, scored: list[ScoredTriple], path: Path) -> Path:
    """Save a strip plot of the three score groups with the easy-tuned threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for x, (attr, label, color) in enumerate(KINDS):
        scores = [getattr(s, attr) for s in scored]
        # Spread points sideways by their position in the list, not at random,
        # so the figure is identical on every run.
        offsets = [((i % 7) - 3) * 0.04 for i in range(len(scores))]
        ax.scatter([x + o for o in offsets], scores, s=22, color=color, alpha=0.85, edgecolors="none")

    summary = summarize(measure, scored)
    threshold = summary.threshold_tuned_on_unrelated
    if threshold is not None:
        ax.axhline(threshold, color="black", linewidth=1, linestyle="--")
        ax.text(2.45, threshold + 0.02, f"threshold {threshold:.2f}\n(tuned on unrelated)",
                ha="right", va="bottom", fontsize=8)

    ax.set_xticks(range(len(KINDS)), [label for _, label, _ in KINDS])
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel(f"similarity to cached query ({measure})")
    ax.set_title(
        f"The near-miss outscored the paraphrase for {summary.near_miss_above_paraphrase} "
        f"of {len(scored)} cached queries",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
