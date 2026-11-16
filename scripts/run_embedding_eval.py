"""OPTIONAL: rerun the analysis with OpenAI embeddings instead of word overlap.

Needs ``pip install openai==1.109.1`` and ``OPENAI_API_KEY``. It embeds the 90
distinct queries in one batched request (a fraction of a cent with
text-embedding-3-small) and prints the same report block as the offline run.

Usage:
    OPENAI_API_KEY=... python scripts/run_embedding_eval.py --model text-embedding-3-small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cache_eval.dataset import all_queries, load_triples  # noqa: E402
from cache_eval.embeddings import EmbeddingCosine, openai_embedder  # noqa: E402
from cache_eval.experiment import format_measure, score_triples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model name")
    args = parser.parse_args()

    try:
        embed = openai_embedder(args.model)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    triples = load_triples()
    similarity = EmbeddingCosine(embed)
    similarity.prefetch(all_queries(triples))
    print("\n".join(format_measure(f"openai:{args.model}", score_triples(triples, similarity))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
