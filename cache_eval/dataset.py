"""Load and validate the hand-written query triples.

Each triple is one cached query (the *anchor*) plus two incoming queries:

- a *paraphrase*, which asks for the same thing and should reuse the cached
  answer, and
- a *near-miss*, which looks similar but needs a different answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "query_triples.json"

NEAR_MISS_TYPES: tuple[str, ...] = ("number", "negation", "role_swap", "entity", "antonym")


@dataclass(frozen=True)
class Triple:
    """One cached query and its two test queries."""

    id: int
    type: str
    anchor: str
    paraphrase: str
    near_miss: str


class DatasetError(ValueError):
    """Raised when the data file is missing fields or internally inconsistent."""


def load_triples(path: Path = DEFAULT_DATA_PATH) -> list[Triple]:
    """Read the triples from ``path`` and check that they are well formed.

    Raises:
        DatasetError: if a field is missing or empty, an id repeats, a type is
            unknown, or a query appears twice in the file.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"data file is not valid JSON: {path}: {exc}") from exc

    triples = [_parse_triple(item) for item in raw.get("triples", [])]
    if not triples:
        raise DatasetError(f"no triples found in {path}")
    _check_consistency(triples)
    return triples


def _parse_triple(item: dict) -> Triple:
    fields = ("id", "type", "anchor", "paraphrase", "near_miss")
    missing = [name for name in fields if name not in item]
    if missing:
        raise DatasetError(f"triple {item.get('id', '?')} is missing {missing}")
    for name in ("anchor", "paraphrase", "near_miss"):
        if not str(item[name]).strip():
            raise DatasetError(f"triple {item['id']} has an empty {name!r}")
    if item["type"] not in NEAR_MISS_TYPES:
        raise DatasetError(f"triple {item['id']} has unknown type {item['type']!r}")
    return Triple(
        id=int(item["id"]),
        type=item["type"],
        anchor=item["anchor"],
        paraphrase=item["paraphrase"],
        near_miss=item["near_miss"],
    )


def _check_consistency(triples: list[Triple]) -> None:
    ids = [t.id for t in triples]
    if len(ids) != len(set(ids)):
        raise DatasetError("triple ids are not unique")
    queries = [q.lower() for t in triples for q in (t.anchor, t.paraphrase, t.near_miss)]
    if len(queries) != len(set(queries)):
        raise DatasetError("the same query text appears more than once")


def all_queries(triples: list[Triple]) -> list[str]:
    """Every query string in the dataset, in file order (used to fit IDF weights)."""
    return [q for t in triples for q in (t.anchor, t.paraphrase, t.near_miss)]
