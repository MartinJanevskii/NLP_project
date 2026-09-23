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
assert keys == {
    "director": set(),
    "actor": {"A", "B", "C"},
    "genre set": {"Comedy|Drama"},
}
with tempfile.TemporaryDirectory() as folder:
    pr.OUT = Path(folder)
    pr.table("t", "Cap", ["A", "B"], [["x", 1]])
    text = (pr.OUT / "t.tex").read_text()
    assert r"\caption{Cap\RunTag}" in text and r"x & 1 \\" in text
    assert r"\label{tab:t}" in text
    pr.table("u", "Cap", ["A"], [["x"]], tag=False)
    assert r"\caption{Cap}" in (pr.OUT / "u.tex").read_text()
    pr.pending("p", "Cap", "no seeds")
    assert "Pending: no seeds" in (pr.OUT / "p.tex").read_text()
    # A run with no completed seeds or responses yields pending tables, not a crash.
    empty_run = Path(folder) / "run"
    empty_run.mkdir()
    pr.results(empty_run, pilot=True)
    pr.text_stats(empty_run, pilot=True)
    for name in ("main_results", "contrasts", "text_stats"):
        assert "Pending:" in (pr.OUT / f"{name}.tex").read_text(), name
    # Outputs of an earlier run never survive into a later one.
    (pr.OUT / "heatmap_ndcg20.pdf").write_text("stale")
    pr.clear_outputs()
    assert not [p for p in pr.OUT.iterdir() if p.suffix in (".tex", ".pdf")]
    # Qualitative examples fall back to whichever configurations exist.
    partial = Path(folder) / "partial"
    partial.mkdir()
    columns = [
        "configuration",
        "item_id",
        "triples",
        "entities",
        "context_characters",
        "prompt_characters",
        "response_words",
        "entity_coverage",
        "unsupported_known_entities",
        "input_tokens",
        "output_tokens",
    ]
    (partial / "text_analysis.csv").write_text(
        ",".join(columns) + "\nH1P2,0,5,6,397,669,24,1.0,0,186,39\n"
    )
    (partial / "qualitative.json").write_text(
        '[{"configuration": "H1P2", "title": "Heat (1995)", "response": "A **crime** film."}]'
    )
    pr.text_stats(partial, pilot=False)
    assert r"\item[H1P2] A crime film." in (pr.OUT / "qualitative.tex").read_text()
    # Full runs get a short reference-row label that fits the page width.
    (partial / "aggregated.csv").write_text(
        "configuration,"
        + ",".join(f"{m}_{s}" for m in pr.METRICS for s in ("mean", "std"))
        + "\nH1P1,"
        + ",".join(["0.1"] * 8)
        + "\n"
    )
    pr.results(partial, pilot=False)
    assert "CoLaKG (orig.)" in (pr.OUT / "main_results.tex").read_text()
print("paper_results helper checks passed")
