"""Tables, plots and text diagnostics computed only from saved experiment files."""

import csv
import re
import statistics
from functools import lru_cache
from pathlib import Path

from llm_subset import atomic_json


@lru_cache(maxsize=20000)
def entity_pattern(name):
    return re.compile(r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)")


def mentions(text, names):
    # ponytail: vocabulary matching misses novel entities and aliases; use validated NER if needed.
    text = text.casefold()
    return {
        name
        for name in names
        if len(name) > 2
        and name.casefold() != "unknown"
        and name.casefold() in text
        and entity_pattern(name).search(text)
    }


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def contrasts(raw):
    """Descriptive paired differences; never infer significance from three seeds."""
    lookup = {(r["configuration"], r["seed"]): r for r in raw}
    comparisons = {
        "RQ1_H2_minus_H1": [("H2" + p, 1 / 3) for p in ("P1", "P2", "P3")]
        + [("H1" + p, -1 / 3) for p in ("P1", "P2", "P3")],
    }
    for p in ("P2", "P3"):
        comparisons[f"RQ2_{p}_minus_P1_over_H1_H2"] = [
            (h + p, 0.5) for h in ("H1", "H2")
        ] + [(h + "P1", -0.5) for h in ("H1", "H2")]
        comparisons[f"RQ3_depth_effect_{p}_minus_P1"] = [
            ("H2" + p, 1),
            ("H1" + p, -1),
            ("H2P1", -1),
            ("H1P1", 1),
        ]
    for p in ("P1", "P2", "P3"):
        comparisons[f"RQ4_H3_minus_H2_{p}"] = [("H3" + p, 1), ("H2" + p, -1)]
    results = []
    for name, terms in comparisons.items():
        if not all(
            (config, seed) in lookup for config, _ in terms for seed in (42, 123, 2026)
        ):
            continue
        for metric in ("recall@10", "recall@20", "ndcg@10", "ndcg@20"):
            values = [
                sum(weight * lookup[config, seed][metric] for config, weight in terms)
                for seed in (42, 123, 2026)
            ]
            results.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "mean_difference": statistics.mean(values),
                    "sample_std": statistics.stdev(values),
                }
            )
    return results


def report(run):
    from experiment import SEEDS, file_hash, read_json
    from llm_subset import request_id

    manifest = read_json(run / "manifest.json")
    labels = read_json(run.parent / "entity_labels.json")
    protocol = read_json(run.parent / "protocol.json")
    vocabulary = set(labels.values())
    pilot = manifest["purpose"] == "engineering_only"
    raw, text_rows, qualitative, summaries = [], [], [], []
    for config in sorted(manifest["configurations"]):
        folder = run / config
        for seed in SEEDS:
            path = folder / f"seed_{seed}.json"
            if path.exists():
                result = read_json(path)
                if result["status"] != "COMPLETE" or result["metadata"]["inputs"][
                    "embeddings.pt"
                ] != file_hash(folder / "embeddings.pt"):
                    raise ValueError(f"Stale result: {path}")
                raw.append(
                    {
                        "configuration": config,
                        "seed": seed,
                        "purpose": manifest["purpose"],
                        **result["metrics"],
                    }
                )
        if not (folder / "responses.json").exists():
            continue
        responses = read_json(folder / "responses.json")
        contexts = read_json(folder / "contexts.json")
        requests = read_json(folder / "requests.json")
        if not len(responses) == len(contexts) == len(requests) == manifest["items"]:
            raise ValueError(f"Mismatched artifact lengths: {folder}")
        for facts, response, request in zip(contexts, responses, requests):
            assert facts["item_id"] == response["item_id"]
            if response["request_id"] != request_id(protocol["endpoint"], request):
                raise ValueError("Response fingerprint no longer matches prompt")
            edges = facts["first_hop"] + facts["second_hop"]
            entities = {
                labels[node] for head, _, tail in edges for node in (head, tail)
            }
            mentioned = mentions(response["text"], vocabulary)
            covered = mentioned & entities
            unsupported = mentioned - entities
            eligible_entities = {
                name
                for name in entities
                if len(name) > 2 and name.casefold() != "unknown"
            }
            row = {
                "configuration": config,
                "item_id": facts["item_id"],
                "triples": len(edges),
                "entities": len(entities),
                "context_characters": len(request["messages"][1]["content"]),
                "prompt_characters": sum(
                    len(m["content"]) for m in request["messages"]
                ),
                "response_words": len(response["text"].split()),
                "entity_coverage": len(covered) / len(eligible_entities)
                if eligible_entities
                else 0,
                "unsupported_known_entities": len(unsupported),
                "input_tokens": (response.get("usage") or {}).get("prompt_tokens", 0),
                "output_tokens": (response.get("usage") or {}).get(
                    "completion_tokens", 0
                ),
            }
            text_rows.append(row)
            if facts["item_id"] in [r["item_id"] for r in contexts[:3]]:
                qualitative.append(
                    {
                        "configuration": config,
                        "title": labels[f"movie:{facts['raw_movie_id']}"],
                        "response": response["text"],
                        "covered_entities": sorted(covered),
                        "unsupported_known_entities": sorted(unsupported),
                    }
                )
        seeds = [r for r in raw if r["configuration"] == config]
        if len(seeds) == 3:
            summary = {
                "configuration": config,
                "purpose": manifest["purpose"],
                "seeds": 3,
            }
            for metric in ("recall@10", "recall@20", "ndcg@10", "ndcg@20"):
                values = [r[metric] for r in seeds]
                summary[metric + "_mean"] = statistics.mean(values)
                summary[metric + "_std"] = statistics.stdev(values)
            for metric in (
                "triples",
                "response_words",
                "entity_coverage",
                "unsupported_known_entities",
            ):
                summary[metric] = statistics.mean(
                    r[metric] for r in text_rows if r["configuration"] == config
                )
            summaries.append(summary)
    write_csv(run / "per_seed.csv", raw)
    write_csv(run / "text_analysis.csv", text_rows)
    write_csv(run / "aggregated.csv", summaries)
    effects = contrasts(raw)
    write_csv(run / "contrasts.csv", effects)
    atomic_json(run / "qualitative.json", qualitative)
    plots = []
    if summaries:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(run / "matplotlib_cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for metric, label in (
            ("ndcg@20_mean", "NDCG@20"),
            ("recall@20_mean", "Recall@20"),
            ("triples", "KG triples per context"),
            ("response_words", "Response words"),
            ("entity_coverage", "Entity coverage"),
        ):
            fig, ax = plt.subplots(figsize=(8, 4))
            error = (
                [s[metric.replace("_mean", "_std")] for s in summaries]
                if metric.endswith("_mean")
                else None
            )
            ax.bar(
                [s["configuration"] for s in summaries],
                [s[metric] for s in summaries],
                yerr=error,
                capsize=3,
            )
            ax.set_ylabel(label)
            ax.set_title(
                "ENGINEERING PILOT — not research results"
                if pilot
                else "MovieLens context × prompt"
            )
            fig.tight_layout()
            name = metric.replace("@", "_") + ".png"
            fig.savefig(run / name, dpi=160)
            plt.close(fig)
            plots.append(name)
    lines = [
        "# Engineering pilot report" if pilot else "# MovieLens experiment results",
        "",
        "These are real measured diagnostics, not evidence of recommendation improvements."
        if pilot
        else "Results below are computed from saved seed runs.",
        "",
        "| Configuration | Recall@20 mean ± sample SD | NDCG@20 mean ± sample SD |",
        "|---|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['configuration']} | {s['recall@20_mean']:.5f} ± {s['recall@20_std']:.5f} | {s['ndcg@20_mean']:.5f} ± {s['ndcg@20_std']:.5f} |"
        )
    pending = [
        c
        for c in manifest["configurations"]
        if c not in {s["configuration"] for s in summaries}
    ]
    lines += [
        "",
        "Pending configurations: "
        + (", ".join(pending) if pending else "none in this run"),
        "",
        "Entity diagnostics use exact vocabulary matching, miss unseen names/aliases, and can match ambiguous titles. Unsupported mentions are a proxy, not a hallucination rate.",
        "User embeddings stay fixed. Pilot runs replace only selected item embeddings and train one subset batch per epoch; the original full candidate catalog remains in use.",
        "",
    ]
    lines += [
        "## Descriptive RQ contrasts",
        "",
        "Paired seed differences, in metric units; sample SD is not a confidence interval. RQ2/RQ3 use H1/H2 only; H3 is a separate filtering comparison.",
        "",
        "| Comparison | NDCG@20 difference ± sample SD |",
        "|---|---:|",
    ]
    lines += [
        f"| {r['comparison']} | {r['mean_difference']:.6f} ± {r['sample_std']:.6f} |"
        for r in effects
        if r["metric"] == "ndcg@20"
    ]
    lines.append("")
    for plot in plots:
        lines.append(f"![{plot}]({plot})")
    lines += ["", "## Qualitative examples", ""]
    for example in qualitative:
        lines += [
            f"### {example['title']} — {example['configuration']}",
            "",
            example["response"],
            "",
        ]
    (run / "report.md").write_text("\n".join(lines))
    # A manuscript is deliberately gated on all nine real, fully trained configurations.
    if not pilot and len(summaries) == 9 and not pending:
        sections = [
            "# Knowledge Graph Context Meets Prompt Design: An Experimental Study of LLM-Enhanced Recommendation",
            "",
            "Draft generated from saved full-run artifacts; requires scientific review.",
            "",
            "## Introduction",
            "We study how KG neighborhood depth and prompt instructions affect recommendation.",
            "## Related Work and CoLaKG Background",
            "CoLaKG's model, semantic fusion and neighbor retrieval are prior contributions, not ours. See https://arxiv.org/abs/2410.12229 .",
            "## Methods and Experimental Setup",
            "Our modifications concern context extraction and prompt strategy. See protocol.json, kg_inspection.json, seed metadata and README for exact settings.",
            "## RQ1–RQ5 Results",
            "The tables, plots and qualitative examples below report saved measurements. Small differences alone do not establish statistical significance.",
            "## Discussion and Limitations",
            "Exact-name grounding proxies miss aliases and unseen entities. Only MovieLens and one generator/encoder pair are studied. Interpret differences using the three-seed dispersion.",
            "## Conclusion",
            "This draft reports descriptive comparisons; causal or generalization claims require review.",
            "",
        ]
        (run / "paper_draft.md").write_text("\n".join(sections + lines[2:]))
    print("Report:", run / "report.md", flush=True)


if __name__ == "__main__":
    import sys

    report(Path(sys.argv[1]).resolve())
