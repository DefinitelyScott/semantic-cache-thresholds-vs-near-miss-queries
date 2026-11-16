import json
from pathlib import Path

import pytest

from cache_eval.dataset import NEAR_MISS_TYPES, DatasetError, all_queries, load_triples


def write(tmp_path: Path, triples: list[dict]) -> Path:
    path = tmp_path / "triples.json"
    path.write_text(json.dumps({"triples": triples}), encoding="utf-8")
    return path


def triple(i: int, **overrides) -> dict:
    base = {"id": i, "type": "number", "anchor": f"anchor {i}",
            "paraphrase": f"paraphrase {i}", "near_miss": f"near miss {i}"}
    return {**base, **overrides}


def test_shipped_dataset_is_balanced():
    triples = load_triples()
    assert len(triples) == 30
    for kind in NEAR_MISS_TYPES:
        assert sum(t.type == kind for t in triples) == 6


def test_all_queries_lists_three_per_triple_in_order():
    triples = load_triples()
    queries = all_queries(triples)
    assert len(queries) == 90
    assert queries[:3] == [triples[0].anchor, triples[0].paraphrase, triples[0].near_miss]


def test_missing_field_is_rejected(tmp_path):
    bad = triple(1)
    del bad["near_miss"]
    with pytest.raises(DatasetError, match="missing"):
        load_triples(write(tmp_path, [bad]))


def test_unknown_type_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="unknown type"):
        load_triples(write(tmp_path, [triple(1, type="typo")]))


def test_empty_text_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="empty"):
        load_triples(write(tmp_path, [triple(1, paraphrase="   ")]))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(DatasetError, match="ids"):
        load_triples(write(tmp_path, [triple(1), triple(1, anchor="other")]))


def test_duplicate_query_text_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="more than once"):
        load_triples(write(tmp_path, [triple(1), triple(2, anchor="ANCHOR 1")]))


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_triples(tmp_path / "nope.json")


def test_invalid_json_gives_a_clear_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_triples(path)
