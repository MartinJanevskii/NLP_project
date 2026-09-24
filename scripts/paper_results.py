"""Write the paper's LaTeX tables and PDF figures from one saved experiment run.

Reads saved artifacts only: no LLM calls, no training. Works on pilots and full/.
Usage: python scripts/paper_results.py artifacts/experiments/<protocol>/<run>
"""

import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path

from experiment import DATA, build_kg, context, make_payload, read_json
from run_baseline import ROOT, load_manifest

OUT = ROOT / "paper/generated"
SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex(value):
    return "".join(SPECIAL.get(c, c) for c in str(value))


def plain(markdown):
    """Drop Markdown emphasis, headings and bullets so responses read as prose."""
    text = re.sub(r"[*#`]+", "", markdown)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


def table(name, caption, header, rows, align=None, tag=True):
    """Header and rows are LaTeX already; callers tex() free text.

    tag=False for run-independent tables (KG and context statistics).
    """
    align = align or "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}\RunTag}}" if tag else rf"\caption{{{caption}}}",
        rf"\label{{tab:{name}}}",
        r"\end{table}",
    ]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def clear_outputs():
    """Remove an earlier run's files so stale figures never reach the paper."""
    for path in OUT.glob("*"):
        if path.suffix in (".tex", ".pdf"):
            path.unlink()


def pending(name, caption, reason):
    table(
        name,
        caption,
        [r"\textbf{Status}"],
        [[r"\textit{Pending: " + tex(reason) + "}"]],
        "l",
    )


def neighbour_agreement(vectors, keys, k=10):
    """Mean share of each item's k nearest cosine neighbours sharing at least one key."""
    import numpy as np

    x = np.asarray(vectors, dtype=float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    k = min(k, len(x) - 1)
    nearest = np.argsort(-similarity, axis=1)[:, :k]
    return float(
        np.mean(
            [
                sum(bool(keys[i] & keys[j]) for j in row) / k
                for i, row in enumerate(nearest)
            ]
        )
    )


def movie_keys(row):
    def clean(values):
        return {v for v in values if v and v.casefold() != "unknown"}

    genres = (row["genres"] or row["Genres"] or "").split("|")
    return {
        "director": clean([row["director"]]),
        "actor": clean((row["actors"] or "").split("|")[:3]),
        "genre set": clean(["|".join(sorted(g for g in genres if g))]),
    }


CONFIGS = [h + p for h in ("H1", "H2", "H3") for p in ("P1", "P2", "P3")]
METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
METRIC_TEX = {
    m: m.split("@")[0].capitalize().replace("Ndcg", "NDCG") + "@" + m[-2:]
    for m in METRICS
}
CONTRASTS = {
    "RQ1_H2_minus_H1": r"RQ1: H2 $-$ H1 (mean over P1--P3)",
    "RQ2_P2_minus_P1_over_H1_H2": r"RQ2: P2 $-$ P1 (mean over H1, H2)",
    "RQ2_P3_minus_P1_over_H1_H2": r"RQ2: P3 $-$ P1 (mean over H1, H2)",
    "RQ3_depth_effect_P2_minus_P1": r"RQ3: depth effect under P2 $-$ under P1",
    "RQ3_depth_effect_P3_minus_P1": r"RQ3: depth effect under P3 $-$ under P1",
    "RQ4_H3_minus_H2_P1": r"RQ4: H3 $-$ H2 (P1)",
    "RQ4_H3_minus_H2_P2": r"RQ4: H3 $-$ H2 (P2)",
    "RQ4_H3_minus_H2_P3": r"RQ4: H3 $-$ H2 (P3)",
}
SHORT = {
    "RQ1_H2_minus_H1": r"RQ1 H2$-$H1",
    "RQ2_P2_minus_P1_over_H1_H2": r"RQ2 P2$-$P1",
    "RQ2_P3_minus_P1_over_H1_H2": r"RQ2 P3$-$P1",
    "RQ3_depth_effect_P2_minus_P1": r"RQ3 depth$\times$P2",
    "RQ3_depth_effect_P3_minus_P1": r"RQ3 depth$\times$P3",
    "RQ4_H3_minus_H2_P1": r"RQ4 H3$-$H2 (P1)",
    "RQ4_H3_minus_H2_P2": r"RQ4 H3$-$H2 (P2)",
    "RQ4_H3_minus_H2_P3": r"RQ4 H3$-$H2 (P3)",
}
PILOT_TITLE = "ENGINEERING PILOT (not research results)"


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as file:
        return list(csv.DictReader(file))


def save(fig, name, pilot):
    import matplotlib.pyplot as plt

    if pilot:
        fig.suptitle(PILOT_TITLE, fontsize=8, color="firebrick")
    fig.savefig(
        OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None}
    )
    plt.close(fig)


def run_info(run, manifest, pilot):
    lines = [
        rf"\def\RunName{{{tex(run.name)}}}",
        rf"\def\RunItems{{{manifest['items']}}}",
        rf"\def\RunEpochs{{{manifest['epochs']}}}",
        r"\def\RunTag{ \textbf{[ENGINEERING PILOT]}}" if pilot else r"\def\RunTag{}",
    ]
    if pilot:
        lines.append(r"\pilotruntrue")
    if manifest["purpose"] == "reduced":
        lines.append(r"\reducedruntrue")
    (OUT / "run_info.tex").write_text("\n".join(lines) + "\n")


def relation_kind(relation):
    return {"directed": "director", "starred in": "actor"}.get(relation, "genre")


def kg_tables(labels, adjacency):
    import matplotlib.pyplot as plt
    import numpy as np

    relations = Counter(r for edges in adjacency.values() for _, r, _ in edges)
    movies = sum(node.startswith("movie:") for node in labels)
    rows = [
        ["Movie nodes", f"{movies:,}"],
        ["Attribute nodes", f"{len(labels) - movies:,}"],
        ["Directed triples", f"{sum(relations.values()):,}"],
        [r"\multicolumn{2}{l}{\textit{Triples per relation}}"],
    ]
    rows += [[r"\quad " + tex(r), f"{n:,}"] for r, n in sorted(relations.items())]
    table(
        "kg_stats",
        "Statistics of the reconstructed MovieLens knowledge graph",
        ["Quantity", "Value"],
        rows,
        tag=False,
    )
    degree = {
        e: len(edges) for e, edges in adjacency.items() if e.startswith("entity:")
    }
    kind = {
        e: relation_kind(edges[0][1])
        for e, edges in adjacency.items()
        if e.startswith("entity:")
    }
    top = sorted(degree, key=lambda e: (-degree[e], labels[e]))[:10]
    table(
        "kg_top",
        "The ten highest-degree attribute nodes (number of movies they connect to)",
        ["Attribute", "Type", "Degree"],
        [[tex(labels[e]), kind[e], degree[e]] for e in top],
        "llr",
        tag=False,
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    # Integer-aligned log bins, so degrees 1, 2, 3 each get their own bar.
    bins = (
        np.unique(np.round(np.logspace(0, np.log10(max(degree.values()) + 1), 25)))
        - 0.5
    )
    for name in ("actor", "director", "genre"):
        values = [d for e, d in degree.items() if kind[e] == name]
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            label=f"{name} ({len(values):,} nodes)",
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Attribute degree (movies connected)",
        ylabel="Number of attributes",
    )
    ax.legend()
    save(fig, "degree_distribution", False)


def context_sizes(mapping, labels, adjacency):
    import matplotlib.pyplot as plt

    sizes = {}
    for strategy in ("H1", "H2", "H3"):
        facts = [context(raw, strategy, adjacency) for raw, _ in mapping]
        sizes[strategy] = {
            "first": [len(f["first_hop"]) for f in facts],
            "second": [len(f["second_hop"]) for f in facts],
            "chars": [
                len(
                    make_payload("deepseek-chat", "P1", raw, f, labels)["messages"][1][
                        "content"
                    ]
                )
                for (raw, _), f in zip(mapping, facts)
            ],
        }
    rows = []
    for strategy, s in sizes.items():
        total = [a + b for a, b in zip(s["first"], s["second"])]
        rows.append(
            [
                strategy,
                f"{statistics.mean(s['first']):.2f}",
                f"{statistics.mean(s['second']):.2f}",
                f"{statistics.median(total):.0f}",
                max(total),
                f"{statistics.mean(s['chars']):,.0f}",
            ]
        )
    table(
        "context_sizes",
        f"Context size per strategy over all {len(mapping):,} recommender items (P1 user message)",
        [
            "Context",
            "1-hop (mean)",
            "2-hop (mean)",
            "Total (median)",
            "Total (max)",
            "Characters (mean)",
        ],
        rows,
        tag=False,
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.boxplot(
        [[a + b for a, b in zip(s["first"], s["second"])] for s in sizes.values()],
        tick_labels=list(sizes),
        showfliers=False,
    )
    ax.set(ylabel="Triples in context", xlabel="Context strategy")
    save(fig, "context_sizes", False)
    raw = 1  # Toy Story (1995), the running example used throughout the paper
    h1, h2, h3 = (context(raw, s, adjacency) for s in ("H1", "H2", "H3"))

    def triple(edge):
        head, relation, tail = edge
        return (
            f"({tex(labels[head])}, \\textit{{{tex(relation)}}}, {tex(labels[tail])})"
        )

    rows = [[r"H1 (1-hop)", triple(e)] for e in h1["first_hop"]]
    rows += [[r"H3 (kept 2-hop)", triple(e)] for e in h3["second_hop"]]
    table(
        "example_context",
        f"Context for \\textit{{{tex(labels[f'movie:{raw}'])}}}: all H1 triples and the H3 second-hop triples "
        f"(H2 adds {len(h2['second_hop'])} sampled second-hop triples, of which H3 keeps {len(h3['second_hop'])})",
        ["Part", "Triple"],
        rows,
        "lp{11cm}",
        tag=False,
    )


def results(run, pilot):
    import matplotlib.pyplot as plt
    import numpy as np

    summary = {r["configuration"]: r for r in read_csv(run / "aggregated.csv")}
    caption = "Recommendation accuracy per configuration (mean $\\pm$ sample SD over three seeds)"
    caption += "; bold: best of the nine configurations"
    if (run / "reference").exists():
        caption += "; CoLaKG (orig.): published embeddings, same data and seeds"
    elif not pilot:
        caption += "; CoLaKG (orig.): published embeddings, seed 2020"
    if not summary:
        pending("main_results", caption, "no configuration has three completed seeds")
    else:
        best = {
            m: max(float(s[m + "_mean"]) for s in summary.values()) for m in METRICS
        }

        def cell(s, m):
            text = f"{float(s[m + '_mean']):.4f} $\\pm$ {float(s[m + '_std']):.4f}"
            return rf"\textbf{{{text}}}" if float(s[m + "_mean"]) == best[m] else text

        rows = [
            [c] + [cell(summary[c], m) for m in METRICS]
            for c in CONFIGS
            if c in summary
        ]
        reference = reference_row(run)
        if reference:
            rows.append(reference)
        elif not pilot:
            baseline = load_manifest(ROOT / "artifacts/EXPERIMENT_MANIFEST.json")[
                "baseline"
            ]
            label = "CoLaKG (orig.)"
            if baseline.get("status") == "COMPLETE":
                rows.append(
                    [label] + [f"{baseline['metrics'][m]:.4f}" for m in METRICS]
                )
            else:
                rows.append([label] + [r"\textit{pending}"] * 4)
        table(
            "main_results",
            caption,
            ["Config."] + [METRIC_TEX[m] for m in METRICS],
            rows,
        )
    grid = np.full((3, 3), np.nan)
    for i, h in enumerate(("H1", "H2", "H3")):
        for j, p in enumerate(("P1", "P2", "P3")):
            if h + p in summary:
                grid[i, j] = float(summary[h + p]["ndcg@20_mean"])
    if summary:
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        image = ax.imshow(grid, cmap="viridis")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    "–" if np.isnan(grid[i, j]) else f"{grid[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color="black" if grid[i, j] > np.nanmean(grid) else "white",
                    fontsize=9,
                )
        ax.set(
            xticks=range(3),
            xticklabels=["P1", "P2", "P3"],
            yticks=range(3),
            yticklabels=["H1", "H2", "H3"],
            xlabel="Prompt strategy",
            ylabel="Context strategy",
        )
        fig.colorbar(image, ax=ax, label="NDCG@20 (mean)")
        save(fig, "heatmap_ndcg20", pilot)
    effects = read_csv(run / "contrasts.csv")
    caption = "Paired contrasts (mean $\\pm$ sample SD of per-seed differences)"
    if not effects:
        pending(
            "contrasts",
            caption,
            "contrasts need all involved configurations with three seeds",
        )
        return
    lookup = {(e["comparison"], e["metric"]): e for e in effects}
    rows = []
    for name, label in CONTRASTS.items():
        if (name, "ndcg@20") in lookup:
            rows.append(
                [label]
                + [
                    f"{float(lookup[name, m]['mean_difference']):+.5f} $\\pm$ {float(lookup[name, m]['sample_std']):.5f}"
                    for m in ("ndcg@20", "recall@20")
                ]
            )
    table(
        "contrasts",
        caption,
        ["Comparison", "$\\Delta$ NDCG@20", "$\\Delta$ Recall@20"],
        rows,
    )


def reference_row(run):
    """CoLaKG with its published embeddings, trained in the same run (reduced runs)."""
    seeds = sorted((run / "reference").glob("seed_*.json"))
    results = [read_json(p) for p in seeds]
    results = [r for r in results if r.get("status") == "COMPLETE"]
    if len(results) < 2:
        return None
    cells = []
    for m in METRICS:
        values = [r["metrics"][m] for r in results]
        cells.append(
            f"{statistics.mean(values):.4f} $\\pm$ {statistics.stdev(values):.4f}"
        )
    return ["CoLaKG (orig.)"] + cells


def significance(run):
    rows = [r for r in read_csv(run / "significance.csv") if r["metric"] == "ndcg@20"]
    caption = (
        "Pre-registered per-user tests on NDCG@20: mean difference, 95\\% bootstrap CI, "
        "Holm-adjusted Wilcoxon $p$, sign of the three per-seed differences, verdict"
    )
    if not rows:
        pending("significance", caption, "no per-user results yet")
        return
    body = []
    for r in rows:
        signs = "".join(
            "+" if d > 0 else "$-$" for d in json.loads(r["seed_differences"])
        )
        body.append(
            [
                SHORT.get(r["comparison"], tex(r["comparison"])),
                f"{float(r['mean_difference']):+.5f}",
                f"[{float(r['ci_low']):+.5f}, {float(r['ci_high']):+.5f}]",
                f"{float(r['p_holm']):.3g}",
                signs,
                r"\textbf{" + r["verdict"] + "}"
                if r["verdict"] != "inconclusive"
                else r["verdict"],
            ]
        )
    table(
        "significance",
        caption,
        [
            "Comparison",
            "$\\Delta$",
            "95\\% CI",
            "$p_{\\text{Holm}}$",
            "Seeds",
            "Verdict",
        ],
        body,
        "lrrrcl",
    )


def dataset_summary(run):
    path = run / "data" / "dataset.json"
    caption = (
        "The reduced dataset (random item sample of the supplied MovieLens-1M split)"
    )
    if not path.exists():
        pending("dataset_summary", caption, "no reduced dataset in this run")
        return
    d = read_json(path)
    table(
        "dataset_summary",
        caption,
        ["Quantity", "Value"],
        [
            [
                "Sampled items (seed " + str(d["sample_seed"]) + ")",
                f"{d['sampled_items']:,}",
            ],
            ["Items kept", f"{d['items']:,}"],
            ["Users kept", f"{d['users']:,}"],
            ["Training interactions", f"{d['train_interactions']:,}"],
            ["Test interactions", f"{d['test_interactions']:,}"],
            [
                "Validation interactions (held out of training)",
                f"{d['val_interactions']:,}",
            ],
            ["Density", f"{100 * d['density']:.2f}\\%"],
        ],
        tag=False,
    )


def learning_curves(run, pilot):
    import matplotlib.pyplot as plt

    budget = run / "epoch_budget.json"
    curves = {}
    for config in ["reference"] + CONFIGS:
        path = run / config / "seed_42.json"
        if path.exists() and read_json(path).get("curve"):
            curves[config] = read_json(path)["curve"]
    if not budget.exists() and not curves:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    if budget.exists():
        b = read_json(budget)
        left.plot(
            [p["epoch"] for p in b["curve"]],
            [p["ndcg@20"] for p in b["curve"]],
            marker=".",
        )
        left.axvline(
            b["epochs"],
            color="firebrick",
            linestyle="--",
            label=f"chosen E = {b['epochs']}",
        )
        left.set(
            xlabel="Epoch",
            ylabel="Validation NDCG@20",
            title="Epoch budget (validation split)",
        )
        left.legend()
    for config, curve in curves.items():
        right.plot(
            [p["epoch"] for p in curve],
            [p["ndcg@20"] for p in curve],
            label=config,
            color="black" if config == "reference" else None,
            linewidth=2 if config == "reference" else 1,
        )
    right.set(
        xlabel="Epoch",
        ylabel="Test NDCG@20 (seed 42)",
        title="Training curves (not used for selection)",
    )
    if curves:
        right.legend(fontsize=6, ncol=2)
    save(fig, "learning_curves", pilot)


def text_stats(run, pilot):
    import matplotlib.pyplot as plt

    rows = read_csv(run / "text_analysis.csv")
    by = {c: [r for r in rows if r["configuration"] == c] for c in CONFIGS}
    by = {c: v for c, v in by.items() if v}
    caption = (
        "Context and generated-text statistics per configuration (means over items)"
    )
    if not by:
        pending("text_stats", caption, "no responses saved")
        return
    columns = (
        "triples",
        "context_characters",
        "response_words",
        "entity_coverage",
        "unsupported_known_entities",
        "input_tokens",
        "output_tokens",
    )
    means = {
        c: {k: statistics.mean(float(r[k]) for r in v) for k in columns}
        for c, v in by.items()
    }
    table(
        "text_stats",
        caption,
        [
            "Config.",
            "Triples",
            "Context chars",
            "Resp. words",
            "Coverage",
            "Unsupported",
            "In tok.",
            "Out tok.",
        ],
        [
            [
                c,
                f"{m['triples']:.1f}",
                f"{m['context_characters']:,.0f}",
                f"{m['response_words']:.1f}",
                f"{m['entity_coverage']:.3f}",
                f"{m['unsupported_known_entities']:.2f}",
                f"{m['input_tokens']:,.0f}",
                f"{m['output_tokens']:,.0f}",
            ]
            for c, m in means.items()
        ],
    )
    names = list(means)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))
    left.bar(names, [means[c]["entity_coverage"] for c in names], color="tab:blue")
    left.set(ylabel="Entity coverage", ylim=(0, 1))
    right.bar(
        names, [means[c]["unsupported_known_entities"] for c in names], color="tab:red"
    )
    right.set(ylabel="Unsupported known entities / response")
    for ax in (left, right):
        ax.tick_params(axis="x", rotation=45)
    save(fig, "grounding", pilot)
    summary = {r["configuration"]: r for r in read_csv(run / "aggregated.csv")}
    shared = [c for c in names if c in summary]
    if shared:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        xs = [means[c]["input_tokens"] + means[c]["output_tokens"] for c in shared]
        ys = [float(summary[c]["ndcg@20_mean"]) for c in shared]
        ax.scatter(xs, ys)
        for c, x, y in zip(shared, xs, ys):
            ax.annotate(
                c, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8
            )
        ax.set(xlabel="Mean tokens per item (input + output)", ylabel="NDCG@20 (mean)")
        save(fig, "cost_quality", pilot)
    examples = (
        read_json(run / "qualitative.json")
        if (run / "qualitative.json").exists()
        else []
    )
    if examples:
        title = examples[0]["title"]
        same = [e for e in examples if e["title"] == title]
        # Prefer the diagonal H1P1/H2P2/H3P3; partial runs show what exists.
        chosen = [
            e for e in same if e["configuration"] in ("H1P1", "H2P2", "H3P3")
        ] or same[:3]
        lines = [r"\begin{description}[style=nextline]"]
        for e in chosen:
            words = plain(e["response"]).split()
            text = " ".join(words[:80]) + (" \\dots" if len(words) > 80 else "")
            lines.append(
                rf"\item[{e['configuration']}] "
                + tex(text).replace(r" \textbackslash{}dots", r" \dots")
            )
        lines.append(r"\end{description}")
        (OUT / "qualitative.tex").write_text(
            rf"\noindent Generated descriptions of \textit{{{tex(title)}}}\RunTag:"
            + "\n"
            + "\n".join(lines)
            + "\n"
        )


def embedding_space(run, rows_by_raw, pilot):
    import matplotlib.pyplot as plt
    import torch
    from sklearn.manifold import TSNE

    sets = {}
    for config in CONFIGS:
        folder = run / config
        if (folder / "embeddings.pt").exists() and (
            folder / "embedding_rows.json"
        ).exists():
            sets[config] = (
                torch.load(
                    folder / "embeddings.pt", map_location="cpu", weights_only=True
                ).numpy(),
                read_json(folder / "embedding_rows.json"),
            )
    caption = (
        "Share of each item's 10 nearest embedding neighbours with a shared attribute"
    )
    if not sets:
        pending("embedding_neighbours", caption, "no embeddings saved")
        return
    order = next(iter(sets.values()))[1]
    published = torch.load(
        DATA / "movie_embeddings_simcse_kg.pt", map_location="cpu", weights_only=True
    )
    # Rows carry raw MovieIDs; reduced runs re-index item_id, so map through item_map.txt.
    original = dict(
        tuple(map(int, line.split()))
        for line in (DATA / "item_map.txt").read_text().splitlines()
    )
    sets = {
        "Published": (
            published[[original[r["raw_movie_id"]] for r in order]].numpy(),
            order,
        ),
        **sets,
    }
    kinds = ("director", "actor", "genre set")
    rows = []
    for name, (vectors, mapping) in sets.items():
        keys = [movie_keys(rows_by_raw[r["raw_movie_id"]]) for r in mapping]
        rows.append(
            [name]
            + [
                f"{neighbour_agreement(vectors, [k[kind] for k in keys]):.3f}"
                for kind in kinds
            ]
        )
    table(
        "embedding_neighbours",
        caption + f" ({len(order)} items)",
        ["Embeddings", "Director", "Actor", "Genre set"],
        rows,
    )
    first = [
        (rows_by_raw[r["raw_movie_id"]]["genres"] or "unknown").split("|")[0]
        for r in order
    ]
    common = [g for g, _ in Counter(first).most_common(6)]
    colours = [f"C{common.index(g)}" if g in common else "lightgrey" for g in first]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    for ax, (name, (vectors, _)) in zip(axes.flat, sets.items()):
        points = TSNE(
            perplexity=min(30, (len(vectors) - 1) // 3), init="pca", random_state=0
        ).fit_transform(vectors)
        ax.scatter(points[:, 0], points[:, 1], c=colours, s=8)
        ax.set_title(name, fontsize=9)
        ax.set(xticks=[], yticks=[])
    for ax in list(axes.flat)[len(sets) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=f"C{i}", label=g)
        for i, g in enumerate(common)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    save(fig, "tsne", pilot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing manifest.json")
    parser.add_argument(
        "--skip-report", action="store_true", help="Reuse the run's existing CSVs"
    )
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "manifest.json").exists():
        parser.error(f"Not a run directory (missing manifest.json): {run}")
    os.environ.setdefault("MPLCONFIGDIR", str(run / "matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    if not args.skip_report:
        from experiment_report import report

        report(run)
    OUT.mkdir(parents=True, exist_ok=True)
    clear_outputs()
    manifest = read_json(run / "manifest.json")
    pilot = manifest["purpose"] == "engineering_only"
    with (DATA / "ml1m_extended_movie.csv").open() as file:
        movies = list(csv.DictReader(file))
    labels, adjacency = build_kg(movies)
    mapping = sorted(
        (
            tuple(map(int, line.split()))
            for line in (DATA / "item_map.txt").read_text().splitlines()
        ),
        key=lambda pair: pair[1],
    )
    run_info(run, manifest, pilot)
    kg_tables(labels, adjacency)
    context_sizes(mapping, labels, adjacency)
    results(run, pilot)
    significance(run)
    dataset_summary(run)
    learning_curves(run, pilot)
    text_stats(run, pilot)
    embedding_space(run, {int(m["MovieID"]): m for m in movies}, pilot)
    print("Paper inputs written to", OUT)


if __name__ == "__main__":
    main()
