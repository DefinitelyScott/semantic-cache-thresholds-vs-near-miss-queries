"""End-to-end: the scripts run from a clean checkout and produce output."""

import subprocess
import sys
from pathlib import Path

from cache_eval.dataset import load_triples
from cache_eval.experiment import build_measures, score_triples
from cache_eval.plot import plot_scores

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, env=env)


def test_run_experiment_prints_every_measure():
    result = run("scripts/run_experiment.py")
    assert result.returncode == 0, result.stderr
    for name in ("tfidf_cosine", "tfidf_cosine_bigrams", "jaccard"):
        assert f"=== {name} ===" in result.stdout


def test_embedding_script_fails_cleanly_without_a_key():
    result = run("scripts/run_embedding_eval.py", env={"PATH": "", "OPENAI_API_KEY": ""})
    assert result.returncode == 1
    assert "OPENAI_API_KEY" in result.stderr


def test_plot_writes_a_png(tmp_path):
    triples = load_triples()
    scored = score_triples(triples, build_measures(triples)["tfidf_cosine"])
    out = plot_scores("tfidf_cosine", scored, tmp_path / "fig.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
