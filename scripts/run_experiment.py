"""Run the offline experiment and print every table quoted in the README.

Usage:
    python scripts/run_experiment.py            # print the report
    python scripts/run_experiment.py --save     # also write outputs/results.json (per-query scores)
    python scripts/run_experiment.py --figure   # also redraw docs/similarity_by_query_kind.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cache_eval.dataset import load_triples  # noqa: E402
from cache_eval.experiment import build_measures, format_report, score_triples  # noqa: E402

FIGURE_MEASURE = "tfidf_cosine"


def save_results(path: Path) -> None:
    """Write every per-query similarity score, for every measure, to JSON."""
    triples = load_triples()
    results = {}
    for name, similarity in build_measures(triples).items():
        results[name] = [
            {
                **asdict(s.triple),
                "unrelated_text": s.unrelated_text,
                "sim_paraphrase": round(s.paraphrase, 6),
                "sim_near_miss": round(s.near_miss, 6),
                "sim_unrelated": round(s.unrelated, 6),
            }
            for s in score_triples(triples, similarity)
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", action="store_true", help="write outputs/results.json")
    parser.add_argument("--figure", action="store_true", help="redraw docs/similarity_by_query_kind.png")
    args = parser.parse_args()

    triples = load_triples()
    print(format_report(triples), end="")

    if args.save:
        out = ROOT / "outputs" / "results.json"
        save_results(out)
        print(f"\nwrote {out.relative_to(ROOT)}")
    if args.figure:
        from cache_eval.plot import plot_scores

        similarity = build_measures(triples)[FIGURE_MEASURE]
        out = plot_scores(FIGURE_MEASURE, score_triples(triples, similarity),
                          ROOT / "docs" / "similarity_by_query_kind.png")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
