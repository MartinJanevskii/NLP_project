# Knowledge Graph Context Meets Prompt Design

NLP course project, FINKI.

We study how the knowledge-graph context and the prompt given to an LLM change the
movie descriptions it writes, and whether that changes recommendation quality in the
CoLaKG recommender. The recommender, data split, encoder and user vectors stay the
same; only the LLM input changes:

- **Context:** H1 direct facts, H2 direct + sampled two-hop facts, H3 filtered two-hop facts
- **Prompt:** P1 generic description, P2 extract-then-describe, P3 recommendation-oriented

That gives nine configurations. Each is trained with three seeds and compared to CoLaKG
with its original descriptions. The paper is in [`paper/`](paper/).

## Requirements

- macOS or Linux, with `git`
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A [DeepSeek API key](https://platform.deepseek.com/) (the run makes about 9,000 requests)
- Around 13 hours for the full run on an Apple M-series Mac; slower on CPU only
- A few GB of free disk space (the sentence encoder alone is about 1.4 GB)

## Quick start

All commands run from the repository root.

**1. Get the code and the original CoLaKG repository.** The MovieLens data comes with CoLaKG.

```sh
git clone https://github.com/MartinJanevskii/NLP_project.git
cd NLP_project
git clone https://github.com/ziqiangcui/CoLaKG-SIGIR25.git vendor/CoLaKG
git -C vendor/CoLaKG checkout be3aa19b59419d197dd4e20f44db737041821f58
```

**2. Install the dependencies.**

```sh
uv sync
```

**3. Add your API key.**

```sh
cp .env.sample .env
# set DEEPSEEK_API_KEY in .env
```

**4. Prepare.** This builds the knowledge graph and writes the experiment protocol.
It makes no API calls.

```sh
uv run python scripts/experiment.py
uv run python scripts/overnight.py prepare
```

**5. Check the key.** This sends one tiny request.

```sh
uv run --env-file .env python scripts/overnight.py check-key
```

**6. Run the experiment.** It generates the descriptions, encodes them, picks the
epoch budget on a validation split, trains the CoLaKG reference and the nine
configurations with three seeds each, and runs the statistics.

```sh
caffeinate -i uv run --env-file .env python scripts/overnight.py run
```

The run is resumable; cached responses are never requested again.

**7. Progress:**

```sh
uv run python scripts/overnight.py status
```

**8. Build the paper.** Use the run folder printed by `status`. You need
[tectonic](https://tectonic-typesetting.github.io/) (`brew install tectonic`).

```sh
uv run python scripts/paper_results.py artifacts/experiments/<protocol>/overnight_<id> --skip-report
cd paper && tectonic main.tex
```

## Results

Everything is written to `artifacts/experiments/<protocol>/overnight_<id>/`:

| File | Contents |
|---|---|
| `findings.json` | Summary: metrics per configuration, contrasts, text statistics |
| `significance.csv` | Per-user Wilcoxon tests with Holm correction and the verdict per hypothesis |
| `aggregated.csv`, `per_seed.csv` | Recall@10/20 and NDCG@10/20, averaged and per seed |
| `text_analysis.csv` | Context size, response length, entity coverage and token counts per item |
| `report.md` | A readable summary with plots and example descriptions |

Your numbers can differ slightly from the paper: DeepSeek may answer the same prompt
a little differently over time, and training on Apple GPUs is not bit-for-bit repeatable.

`paper/generated/` holds the tables and figures the paper includes. They are produced
by step 8, never edited by hand.

## Tests

Quick offline checks. They need no API key and do no training.

```sh
for test in tests/test_*.py; do uv run python "$test"; done
```

## Project layout

```
scripts/                       command-line entry points
  overnight.py                 the reduced study above (prepare / run / status / check-key)
  experiment.py                20- and 100-item pilots and the full 2,000-epoch protocol
  paper_results.py             paper tables and figures from a run
  experiment_train.py          trains CoLaKG for one configuration and seed
  experiment_report.py         CSV tables, plots and report.md for a run
  run_baseline.py              the unmodified CoLaKG training (CUDA machine)
  llm_subset.py, smoke_colakg.py  checks that the original CoLaKG pipeline is reproduced
src/llm_knowledge_enhancement/ the shared code
  kg.py, prompts.py            knowledge graph, the H1-H3 contexts and the P1-P3 prompts
  llm.py, encoder.py           DeepSeek client with response cache, SimCSE encoder
  training.py                  CoLaKG training and per-user evaluation
  overnight.py, reduced.py     the reduced study and its dataset
  report.py, stats.py, paper.py  tables, significance tests and paper output
tests/                         offline checks
paper/                         LaTeX source of the paper
vendor/CoLaKG/                 the original CoLaKG code and data (cloned in step 1)
artifacts/                     everything a run produces (not in git)
```

The original 2,000-epoch CoLaKG baseline (`run_baseline.py`) needs Python 3.8 and CUDA;
its environment is described in `requirements-baseline-cuda.txt` and
`requirements-pipeline-cuda.txt`.
