# Research paper + results generator — design

Date: 2026-09-23. Status: awaiting review.

## Goal

A LaTeX research paper (NLP course, FINKI, 4th-year student level) describing the
KG-context × prompt-strategy study on top of the original CoLaKG recommender, plus
one command that turns any experiment run directory into the paper's results
tables and figures. Every section is written now **except Results**, which contains
only generated inputs and "to discuss" placeholders. Discussion/Conclusion get full
structure; their result-dependent sentences are marked `\todo{}`.

Authors: Martin Janevski, Viktor Hristovski, David Dukoski.
`reference.tex` is a structural/stylistic model only; none of its content is used.

## Out of scope

- Running the full experiment or baseline (needs Linux/CUDA; see PROJECT_GUIDE §8).
- Any change to `experiment.py`, `experiment_train.py`, `experiment_report.py`, or
  upstream code. Changing `experiment_train.py` alters every seed fingerprint.
- Per-user significance testing (listed as future work for that reason).

## Layout

```text
paper/
├── main.tex            Paper; \input{generated/...} for every measured number
├── references.bib      Real citations (CoLaKG, LightGCN, KGAT, KGIN, SimCSE,
│                       MovieLens, CoT, KAR, RLMRec, BPR, NDCG, t-SNE, DeepSeek)
├── figures/            Hand-made figures; placeholder boxes where images belong
└── generated/          Written only by scripts/paper_results.py; committed so
                        the paper compiles without re-running anything
scripts/paper_results.py
```

## Paper sections

1. **Abstract.**
2. **Introduction** — motivation (LLM text as item side information), RQ1 depth
   (H2 vs H1), RQ2 prompt (P2/P3 vs P1), RQ3 depth×prompt interaction, RQ4
   filtering (H3 vs H2), RQ5 grounding of generated text; contributions (we study
   context/prompt; CoLaKG is prior work).
3. **Background and Related Work** — collaborative filtering and BPR; graph CF
   (LightGCN); KG-aware recommenders (KGAT, KGIN); LLMs for recommendation (KAR,
   RLMRec, CoLaKG); prompting (chain-of-thought / structured extraction); sentence
   embeddings (SimCSE).
4. **Data and Knowledge Graph** — MovieLens-1M and the supplied split
   (6,040 users, 3,260 items, 801,218 / 197,321 interactions); KG construction
   (director, first three actors, genre/genre-pair, inverses; raw-MovieID nodes);
   generated KG statistics table; generated degree distribution figure; the
   indexing rule raw MovieID → recommender row.
5. **Method** — CoLaKG recap with equations (LightGCN propagation, semantic
   fusion/neighbour aggregation, BPR loss) at the level the upstream code
   implements; formal set definitions of H1, H2 (≤10 deterministic second-hop
   triples per attribute, target excluded), H3 (role-matched director/actor,
   no genre/unknown bridges, ≤3 per attribute, ≤1 path per destination movie,
   director first); P1–P3 prompt listing with the common grounding instruction;
   generation settings (temperature 0, 512-token cap, truncation rejected);
   SimCSE encoding (pooler output, 1,024-d); TikZ pipeline diagram; TikZ
   H1/H2/H3 neighbourhood example for one movie.
6. **Experimental Setup** — seeds 42/123/2026, 2,000 epochs, last scheduled
   evaluation (no test-based selection); Recall@K and NDCG@K formulas; contrast
   definitions exactly as `experiment_report.contrasts()` computes them;
   grounding proxies (entity coverage, unsupported known entities — explicitly
   not a hallucination rate); embedding-space metric (below); caching and
   fingerprinting for reproducibility; environment.
7. **Results** — `\input{generated/*.tex}` slots only, each followed by a
   `\todo{discuss}` placeholder.
8. **Discussion** — structure per RQ with `\todo{}`s.
9. **Limitations and Future Work** — written in full: one dataset, one
   generator/encoder, three seeds, exact-match grounding, KG reconstruction
   differs from the paper's reported KG, fixed user vectors, per-user
   significance testing, pilot ordering bias.
10. **Conclusion** — structure with `\todo{}`s.

Image placeholders: a framed `\placeholder{caption}` macro box wherever a real
image belongs (e.g. CoLaKG architecture sketch, qualitative example screenshot).

## `scripts/paper_results.py <run_dir>`

Reads only saved artifacts plus the KG built by the existing `build_kg`/`context`.
Calls the existing `experiment_report.report(run)` first so CSVs are current.
Writes to `paper/generated/`:

| Output | Source | Needs training? |
|---|---|---|
| `kg_stats.tex` table | `kg_inspection.json` | no |
| `degree_distribution.pdf` | `build_kg` over metadata | no |
| `context_sizes.tex` + `context_sizes.pdf` (H1/H2/H3 triples per item, full 3,260-item catalogue) | `context()` | no |
| `prompt_lengths.tex` | `requests.json` / `text_analysis.csv` | no |
| `main_results.tex` (Recall/NDCG@10/20 mean ± SD per config; original-CoLaKG row when the baseline manifest is COMPLETE, else "pending") | `aggregated.csv`, manifest | yes |
| `contrasts.tex` (RQ1–RQ4, NDCG@20 and Recall@20) | `contrasts.csv` | yes |
| `heatmap_ndcg20.pdf` (3×3 H × P) | `aggregated.csv` | yes |
| `grounding.tex` + `grounding.pdf` | `text_analysis.csv` | no (needs responses) |
| `cost_quality.tex` (tokens per item vs NDCG@20) | `text_analysis.csv`, `aggregated.csv` | partly |
| `embedding_neighbours.tex` | `embeddings.pt` per config + published embeddings | no |
| `tsne.pdf` (per-config t-SNE coloured by primary genre) | `embeddings.pt` | no |
| `run_info.tex` (`\newcommand`s: run name, items, epochs, pilot flag) | manifest | no |

Embedding-space metric: for each item with a fresh vector, the fraction of its
10 nearest cosine neighbours (among the items evaluated in the run) that share
a director, one of the first three actors, or a genre with it. Also computed for
the published CoLaKG embeddings on the same items, as a reference.

Pilot runs: every table caption and figure title carries "ENGINEERING PILOT",
driven by `run_info.tex`. Missing inputs produce a "pending" table rather
than a crash. The same command works unchanged on `full/`.

One runnable check: `scripts/test_paper_results.py` (plain `assert`s) for the
neighbour metric and LaTeX escaping on a tiny synthetic input.

## Environment

- Rebuild `.venv` (`uv sync`, plus pipeline requirements: transformers,
  matplotlib, scikit-learn already used by the report).
- Install `tectonic` via Homebrew to compile `paper/main.tex` locally.

## Done when

- `python scripts/paper_results.py artifacts/experiments/e35fd36b944318b1/pilot100_e2`
  runs without error and fills `paper/generated/`.
- `tectonic paper/main.tex` compiles without errors or undefined references.
- Existing tests (`test_experiment.py`, `test_llm_subset.py`, `test_baseline.py`)
  and the new check still pass; `ruff check scripts` is clean.
