"""Tables, plots and text diagnostics computed only from saved run files."""

import csv
import os
import re
import statistics
from functools import lru_cache
from pathlib import Path

from llm_knowledge_enhancement.design import CONTRASTS, METRICS, SEEDS
from llm_knowledge_enhancement.files import atomic_json, file_hash, read_json
from llm_knowledge_enhancement.llm import request_id

TEXT_METRICS = (
    "triples",
    "response_words",
    "entity_coverage",
    "unsupported_known_entities",
)
PLOTS = (
    ("ndcg@20_mean", "NDCG@20"),
    ("recall@20_mean", "Recall@20"),
    ("triples", "KG triples per context"),
    ("response_words", "Response words"),
    ("entity_coverage", "Entity coverage"),
)
QUALITATIVE_ITEMS = 3


@lru_cache(maxsize=20000)
def entity_pattern(name: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)")


def countable(name: str) -> bool:
    return len(name) > 2 and name.casefold() != "unknown"


def mentions(text: str, names: set[str]) -> set[str]:
    """Known entity names that appear as whole words in the text (case-insensitive).

    Exact vocabulary matching: it misses aliases and entities outside the KG.
    """
    text = text.casefold()
    return {
        name
        for name in names
        if countable(name)
        and name.casefold() in text
        and entity_pattern(name).search(text)
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def contrasts(raw: list[dict]) -> list[dict]:
    """Descriptive paired differences over seeds; not a significance test."""
    lookup = {(r["configuration"], r["seed"]): r for r in raw}
    results = []
    for name, terms in CONTRASTS.items():
        if not all((config, seed) in lookup for config, _ in terms for seed in SEEDS):
            continue
        for metric in METRICS:
            values = [
                sum(weight * lookup[config, seed][metric] for config, weight in terms)
                for seed in SEEDS
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


def seed_results(folder: Path, config: str, purpose: str) -> list[dict]:
    raw = []
    for seed in SEEDS:
        path = folder / f"seed_{seed}.json"
        if not path.exists():
            continue
        result = read_json(path)
        if result["status"] != "COMPLETE" or result["metadata"]["inputs"][
            "embeddings.pt"
        ] != file_hash(folder / "embeddings.pt"):
            raise ValueError(f"Stale result: {path}")
        raw.append(
            {
                "configuration": config,
                "seed": seed,
                "purpose": purpose,
                **result["metrics"],
            }
        )
    return raw


def text_diagnostics(
    folder: Path,
    config: str,
    items: int,
    labels: dict[str, str],
    endpoint: str,
) -> tuple[list[dict], list[dict]]:
    """Per-item context/response statistics and the first items' example responses."""
    responses = read_json(folder / "responses.json")
    contexts = read_json(folder / "contexts.json")
    requests = read_json(folder / "requests.json")
    if not len(responses) == len(contexts) == len(requests) == items:
        raise ValueError(f"Mismatched artifact lengths: {folder}")
    vocabulary = set(labels.values())
    examples = {r["item_id"] for r in contexts[:QUALITATIVE_ITEMS]}
    rows, qualitative = [], []
    for facts, response, request in zip(contexts, responses, requests, strict=True):
        assert facts["item_id"] == response["item_id"]
        if response["request_id"] != request_id(endpoint, request):
            raise ValueError("Response fingerprint no longer matches prompt")
        edges = facts["first_hop"] + facts["second_hop"]
        entities = {labels[node] for head, _, tail in edges for node in (head, tail)}
        mentioned = mentions(response["text"], vocabulary)
        covered = mentioned & entities
        unsupported = mentioned - entities
        eligible = {name for name in entities if countable(name)}
        usage = response.get("usage") or {}
        rows.append(
            {
                "configuration": config,
                "item_id": facts["item_id"],
                "triples": len(edges),
                "entities": len(entities),
                "context_characters": len(request["messages"][1]["content"]),
                "prompt_characters": sum(
                    len(m["content"]) for m in request["messages"]
                ),
                "response_words": len(response["text"].split()),
                "entity_coverage": len(covered) / len(eligible) if eligible else 0,
                "unsupported_known_entities": len(unsupported),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        )
        if facts["item_id"] in examples:
            qualitative.append(
                {
                    "configuration": config,
                    "title": labels[f"movie:{facts['raw_movie_id']}"],
                    "response": response["text"],
                    "covered_entities": sorted(covered),
                    "unsupported_known_entities": sorted(unsupported),
                }
            )
    return rows, qualitative


def summarize(config: str, purpose: str, seeds: list[dict], text_rows: list[dict]):
    summary = {"configuration": config, "purpose": purpose, "seeds": len(SEEDS)}
    for metric in METRICS:
        values = [r[metric] for r in seeds]
        summary[metric + "_mean"] = statistics.mean(values)
        summary[metric + "_std"] = statistics.stdev(values)
    for metric in TEXT_METRICS:
        summary[metric] = statistics.mean(
            r[metric] for r in text_rows if r["configuration"] == config
        )
    return summary


def bar_plots(run: Path, summaries: list[dict], pilot: bool) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(run / "matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    for metric, label in PLOTS:
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
        names.append(name)
    return names


def markdown(
    manifest: dict,
    summaries: list[dict],
    effects: list[dict],
    plots: list[str],
    qualitative: list[dict],
    pilot: bool,
) -> list[str]:
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
    lines += [
        f"| {s['configuration']} | {s['recall@20_mean']:.5f} ± {s['recall@20_std']:.5f} | {s['ndcg@20_mean']:.5f} ± {s['ndcg@20_std']:.5f} |"
        for s in summaries
    ]
    pending = pending_configurations(manifest, summaries)
    lines += [
        "",
        "Pending configurations: "
        + (", ".join(pending) if pending else "none in this run"),
        "",
        "Entity diagnostics use exact vocabulary matching, miss unseen names/aliases, and can match ambiguous titles. Unsupported mentions are a proxy, not a hallucination rate.",
        "User embeddings stay fixed. Pilot runs replace only selected item embeddings and train one subset batch per epoch; the original full candidate catalog remains in use.",
        "",
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
    lines += [f"![{plot}]({plot})" for plot in plots]
    lines += ["", "## Qualitative examples", ""]
    for example in qualitative:
        lines += [
            f"### {example['title']} — {example['configuration']}",
            "",
            example["response"],
            "",
        ]
    return lines


def pending_configurations(manifest: dict, summaries: list[dict]) -> list[str]:
    done = {s["configuration"] for s in summaries}
    return [c for c in manifest["configurations"] if c not in done]


PAPER_DRAFT = [
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


def report(run: Path) -> None:
    """Write per_seed/text_analysis/aggregated/contrasts CSVs, plots and report.md."""
    manifest = read_json(run / "manifest.json")
    labels = read_json(run.parent / "entity_labels.json")
    endpoint = read_json(run.parent / "protocol.json")["endpoint"]
    purpose = manifest["purpose"]
    pilot = purpose == "engineering_only"
    raw, text_rows, qualitative, summaries = [], [], [], []
    for config in sorted(manifest["configurations"]):
        folder = run / config
        raw += seed_results(folder, config, purpose)
        if not (folder / "responses.json").exists():
            continue
        rows, examples = text_diagnostics(
            folder, config, manifest["items"], labels, endpoint
        )
        text_rows += rows
        qualitative += examples
        seeds = [r for r in raw if r["configuration"] == config]
        if len(seeds) == len(SEEDS):
            summaries.append(summarize(config, purpose, seeds, text_rows))
    write_csv(run / "per_seed.csv", raw)
    write_csv(run / "text_analysis.csv", text_rows)
    write_csv(run / "aggregated.csv", summaries)
    effects = contrasts(raw)
    write_csv(run / "contrasts.csv", effects)
    atomic_json(run / "qualitative.json", qualitative)
    plots = bar_plots(run, summaries, pilot) if summaries else []
    lines = markdown(manifest, summaries, effects, plots, qualitative, pilot)
    (run / "report.md").write_text("\n".join(lines))
    complete = len(summaries) == len(manifest["configurations"]) == 9
    if not pilot and complete:
        (run / "paper_draft.md").write_text("\n".join(PAPER_DRAFT + lines[2:]))
    print("Report:", run / "report.md", flush=True)
