"""Offline checks for the paper results generator."""

import json
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
    # Reduced overnight runs: reference row from its three seeds, significance and dataset tables.
    reduced = Path(folder) / "reduced"
    (reduced / "reference").mkdir(parents=True)
    for seed, value in ((42, 0.1), (123, 0.2), (2026, 0.3)):
        (reduced / "reference" / f"seed_{seed}.json").write_text(
            json.dumps(
                {"status": "COMPLETE", "metrics": {m: value for m in pr.METRICS}}
            )
        )
    assert pr.reference_row(reduced)[1] == "0.2000 $\\pm$ 0.1000"
    assert pr.reference_row(Path(folder) / "missing") is None
    pr.significance(reduced)
    pr.dataset_summary(reduced)
    assert "Pending:" in (pr.OUT / "significance.tex").read_text()
    assert "Pending:" in (pr.OUT / "dataset_summary.tex").read_text()
    (reduced / "significance.csv").write_text(
        "comparison,metric,mean_difference,ci_low,ci_high,seed_differences,p,users,p_holm,verdict\n"
        'RQ1_H2_minus_H1,ndcg@20,0.001,0.0005,0.0015,"[0.001, 0.002, 0.001]",0.001,5758,0.008,supported\n'
        'RQ1_H2_minus_H1,recall@20,0.001,0.0,0.002,"[0.001, 0.001, 0.001]",0.2,5758,0.9,secondary\n'
    )
    (reduced / "data").mkdir()
    (reduced / "data" / "dataset.json").write_text(
        json.dumps(
            {
                "items": 1000,
                "users": 5758,
                "train_interactions": 10,
                "test_interactions": 5,
                "density": 0.05,
                "val_interactions": 1,
                "sampled_items": 1000,
                "sample_seed": 2026,
            }
        )
    )
    pr.significance(reduced)
    pr.dataset_summary(reduced)
    text = (pr.OUT / "significance.tex").read_text()
    assert "supported" in text and "secondary" not in text and "+++" in text
    assert "5,758" in (pr.OUT / "dataset_summary.tex").read_text()
    pr.learning_curves(reduced, pilot=False)  # no curves saved: must not crash
print("paper_results helper checks passed")
