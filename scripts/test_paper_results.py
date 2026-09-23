"""Offline checks for the paper results generator."""

import tempfile
from pathlib import Path

import paper_results as pr

assert pr.tex("Tom & Jerry_50% #1 $") == r"Tom \& Jerry\_50\% \#1 \$"
assert (
    pr.plain("**Target:** *Heat* (1995)\n- Genre: Drama")
    == "Target: Heat (1995) Genre: Drama"
)
close = [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]]
assert pr.neighbour_agreement(close, [{"a"}, {"a"}, {"b"}, {"b"}], k=1) == 1.0
assert pr.neighbour_agreement(close, [{"a"}, {"b"}, {"a"}, {"b"}], k=1) == 0.0
keys = pr.movie_keys(
    {"director": "unknown", "actors": "A|B|C|D", "genres": "Drama|Comedy", "Genres": ""}
)
assert keys == {"director": set(), "actor": {"A", "B", "C"}, "genre set": {"Comedy|Drama"}}
with tempfile.TemporaryDirectory() as folder:
    pr.OUT = Path(folder)
    pr.table("t", "Cap", ["A", "B"], [["x", 1]])
    text = (pr.OUT / "t.tex").read_text()
    assert r"\caption{Cap\RunTag}" in text and r"x & 1 \\" in text
    assert r"\label{tab:t}" in text
    pr.pending("p", "Cap", "no seeds")
    assert "Pending: no seeds" in (pr.OUT / "p.tex").read_text()
    # A run with no completed seeds or responses yields pending tables, not a crash.
    empty_run = Path(folder) / "run"
    empty_run.mkdir()
    pr.results(empty_run, pilot=True)
    pr.text_stats(empty_run, pilot=True)
    for name in ("main_results", "contrasts", "text_stats"):
        assert "Pending:" in (pr.OUT / f"{name}.tex").read_text(), name
print("paper_results helper checks passed")
