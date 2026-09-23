# Research Paper + Results Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A compilable LaTeX paper (all sections except Results written) plus `scripts/paper_results.py`, which turns any saved experiment run into the paper's tables and figures.

**Architecture:** `paper/main.tex` never contains measured numbers. It `\input`s files from `paper/generated/`, which only `scripts/paper_results.py <run_dir>` writes. It reuses `experiment.build_kg`/`context` and the existing `experiment_report.report`. Macros fall back to placeholder boxes when a generated file is missing, so the paper always compiles.

**Tech Stack:** Python 3.13 (`.venv`), numpy, torch (tensor loading only), scikit-learn (t-SNE), matplotlib; LaTeX compiled with tectonic (XeTeX); natbib + BibTeX.

**Spec:** `docs/superpowers/specs/2026-09-23-research-paper-design.md`

## Global Constraints

- Do not modify `scripts/experiment.py`, `scripts/experiment_train.py`, `scripts/experiment_report.py`, or anything under `vendor/`.
- No LLM API calls and no training in any new code.
- Authors: Martin Janevski, Viktor Hristovski, David Dukoski.
- No content from `reference.tex`; only its structure/style (article, 11pt, a4, 2.54cm margins, listings colours, `[H]` floats).
- Every pilot-derived table/figure caption carries `\RunTag` (`[ENGINEERING PILOT]` on pilots).
- Work on branch `paper-draft` (created from `CoLaKG-experiment`).

## Review Focus

1. Run directory with some configurations lacking seeds or responses → affected tables become "Pending" and nothing crashes (Task 3 step: partial-run check).
2. Baseline manifest not COMPLETE → the original-CoLaKG row says pending, and pilots never show a full-run baseline next to pilot numbers (Task 3).
3. Movie titles and responses containing `& _ % # $` or Markdown `**` → escaped and stripped, so the LaTeX compiles (Task 2 test plus Task 7 compile).
4. Compiling the paper before any run → every `\gentable`/`\genfigure` shows a placeholder box instead of erroring (Task 4 step: compile with an empty `generated/`).
5. Running the command twice → identical outputs, with no appended duplicates (Task 3 step: rerun and diff).

---

### Task 1: Environment and branch

**Files:** Modify `.gitignore`; delete `reference.md`.

- [ ] Create the branch: `git checkout -b paper-draft`
- [ ] Rebuild the venv:
  `uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python torch numpy scipy pandas scikit-learn matplotlib transformers tensorboardX ruff`
- [ ] Install tectonic: `brew install tectonic`
- [ ] Run the existing checks. All must pass before new work starts:
  `.venv/bin/python scripts/test_experiment.py && .venv/bin/python scripts/test_llm_subset.py && .venv/bin/python scripts/test_baseline.py`
- [ ] Remove the throwaway conversion (`rm reference.md`) and append to `.gitignore`:
  ```
  # LaTeX build outputs
  /paper/*.pdf
  /paper/*.aux
  /paper/*.log
  ```
- [ ] Commit: `git add .gitignore && git commit -m "chore: ignore paper build outputs"`

### Task 2: Pure helpers with test

**Files:** Create `scripts/paper_results.py` (helpers only) and `scripts/test_paper_results.py`.

**Produces:** `tex(value) -> str`, `plain(markdown) -> str`, `table(name, caption, header, rows, align=None) -> None`, `pending(name, caption, reason) -> None`, `neighbour_agreement(vectors, keys, k=10) -> float`, `movie_keys(row) -> dict[str, set[str]]`, `OUT: Path`.

- [ ] Write the test:
  ```python
  """Offline checks for the paper results generator."""
  import tempfile
  from pathlib import Path

  import paper_results as pr

  assert pr.tex("Tom & Jerry_50% #1 $") == r"Tom \& Jerry\_50\% \#1 \$"
  assert pr.plain("**Target:** *Heat* (1995)\n- Genre: Drama") == "Target: Heat (1995) Genre: Drama"
  close = [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]]
  assert pr.neighbour_agreement(close, [{"a"}, {"a"}, {"b"}, {"b"}], k=1) == 1.0
  assert pr.neighbour_agreement(close, [{"a"}, {"b"}, {"a"}, {"b"}], k=1) == 0.0
  keys = pr.movie_keys({"director": "unknown", "actors": "A|B|C|D", "genres": "Drama|Comedy", "Genres": ""})
  assert keys == {"director": set(), "actor": {"A", "B", "C"}, "genre set": {"Comedy|Drama"}}
  with tempfile.TemporaryDirectory() as folder:
      pr.OUT = Path(folder)
      pr.table("t", "Cap", ["A", "B"], [["x", 1]])
      text = (pr.OUT / "t.tex").read_text()
      assert r"\caption{Cap\RunTag}" in text and r"x & 1 \\" in text and r"\label{tab:t}" in text
      pr.pending("p", "Cap", "no seeds")
      assert "Pending: no seeds" in (pr.OUT / "p.tex").read_text()
  print("paper_results helper checks passed")
  ```
- [ ] Run it: `.venv/bin/python scripts/test_paper_results.py`. Expected: `ModuleNotFoundError: paper_results`.
- [ ] Implement the helpers in `scripts/paper_results.py`:
  ```python
  """Write the paper's LaTeX tables and PDF figures from one saved experiment run.

  Reads saved artifacts only: no LLM calls, no training. Works on pilots and full/.
  """

  import re
  from pathlib import Path

  from run_baseline import ROOT

  OUT = ROOT / "paper/generated"
  SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


  def tex(value):
      return "".join(SPECIAL.get(c, c) for c in str(value))


  def plain(markdown):
      """Drop Markdown emphasis, headings and bullets so responses read as prose."""
      text = re.sub(r"[*#`]+", "", markdown)
      text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
      return " ".join(text.split())


  def table(name, caption, header, rows, align=None):
      """Header and rows are LaTeX already; callers tex() free text."""
      align = align or "l" + "r" * (len(header) - 1)
      lines = [r"\begin{table}[H]", r"\centering", r"\small", rf"\begin{{tabular}}{{{align}}}",
               r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
      lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
      lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}\RunTag}}",
                rf"\label{{tab:{name}}}", r"\end{table}"]
      (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


  def pending(name, caption, reason):
      table(name, caption, [r"\textbf{Status}"], [[r"\textit{Pending: " + tex(reason) + "}"]], "l")


  def neighbour_agreement(vectors, keys, k=10):
      """Mean share of each item's k nearest cosine neighbours sharing at least one key."""
      import numpy as np

      x = np.asarray(vectors, dtype=float)
      x /= np.linalg.norm(x, axis=1, keepdims=True)
      similarity = x @ x.T
      np.fill_diagonal(similarity, -np.inf)
      k = min(k, len(x) - 1)
      nearest = np.argsort(-similarity, axis=1)[:, :k]
      return float(np.mean([sum(bool(keys[i] & keys[j]) for j in row) / k
                            for i, row in enumerate(nearest)]))


  def movie_keys(row):
      def clean(values):
          return {v for v in values if v and v.casefold() != "unknown"}

      genres = (row["genres"] or row["Genres"] or "").split("|")
      return {
          "director": clean([row["director"]]),
          "actor": clean((row["actors"] or "").split("|")[:3]),
          "genre set": clean(["|".join(sorted(g for g in genres if g))]),
      }
  ```
- [ ] Run the test. Expected: `paper_results helper checks passed`.
- [ ] Commit: `git add scripts/paper_results.py scripts/test_paper_results.py && git commit -m "feat: add paper results helpers"`

### Task 3: Generator `main()` producing every artifact

**Files:** Modify `scripts/paper_results.py`.

**Consumes:** the Task 2 helpers; `experiment.DATA`, `build_kg`, `context`, `make_payload`, `read_json`, `SEEDS`; `experiment_report.report`; `run_baseline.load_manifest`.

**Produces in `paper/generated/`:**
- tables: `run_info.tex`, `kg_stats.tex`, `kg_top.tex`, `context_sizes.tex`, `example_context.tex`, `main_results.tex`, `contrasts.tex`, `text_stats.tex`, `embedding_neighbours.tex`, `qualitative.tex`
- figures: `degree_distribution.pdf`, `context_sizes.pdf`, `heatmap_ndcg20.pdf`, `grounding.pdf`, `cost_quality.pdf`, `tsne.pdf`

- [ ] Implement `main()` with these functions, each writing its files. Any function whose inputs are missing calls `pending(...)` or skips its figure:
  - `run_info(manifest, run)`: writes `\def\RunName{..}\def\RunItems{..}\def\RunEpochs{..}\def\RunTag{..}`, plus `\pilotruntrue` for pilots. `\RunTag` is ` \textbf{[ENGINEERING PILOT]}` on pilots and empty on the full run.
  - `kg_tables(labels, adjacency)`:
    - `kg_stats`: rows are the movie count, attribute count, triple count, then each relation's count.
    - `kg_top`: the top 10 attributes by degree.
    - `degree_distribution.pdf`: log-log histogram of attribute degrees, one series per relation (`directed`, `starred in`, genre).
  - `context_sizes(mapping, adjacency, labels)`: covers all 3,260 `item_map.txt` items.
    - For H1/H2/H3, compute the mean, median and max of first-hop and second-hop triples, plus the mean user-message characters via `make_payload("deepseek-chat", "P1", raw, facts, labels)`.
    - Also writes a boxplot `context_sizes.pdf`.
    - `example_context.tex` lists Toy Story's (raw MovieID 1) H1 triples, the H2 count and the H3 triples.
  - `results(run, manifest, pilot)`:
    - `main_results` comes from `aggregated.csv`: mean±SD for 4 metrics per configuration, with the best per column in `\textbf`.
    - On the full run only, add the baseline row "CoLaKG (published embeddings, seed 2020)" from `load_manifest(...)["baseline"]["metrics"]` when its status is COMPLETE; otherwise that row reads "pending".
    - `contrasts` comes from `contrasts.csv` for ndcg@20 and recall@20, with readable names.
    - `heatmap_ndcg20.pdf` is a 3×3 heatmap, with NaN and "–" for missing configurations.
  - `text_stats(run, pilot)`:
    - The table comes from `text_analysis.csv`, as per-configuration means of triples, context chars, response words, entity coverage, unsupported known entities, and input/output tokens.
    - `grounding.pdf`: bars for coverage and unsupported.
    - `cost_quality.pdf`: scatter of mean input tokens against NDCG@20 mean, labelled by configuration. Only drawn when `aggregated.csv` exists.
    - `qualitative.tex`: for the first item in `qualitative.json`, the H1P1, H2P2 and H3P3 responses, `plain()`→`tex()`, truncated to 80 words.
  - `embedding_space(run, rows_by_raw, pilot)`:
    - For each configuration with `embeddings.pt`+`embedding_rows.json`, and for the published tensor rows at the same item IDs, compute `neighbour_agreement` at k=10 for the director, actor and genre-set keys.
    - `tsne.pdf` is a 2×5 grid (published + 9) with `TSNE(perplexity=min(30,(n-1)//3), init="pca", random_state=0)`, coloured by the first genre (top 6 coloured, rest grey).
  - CLI: `run` positional plus `--skip-report`. Error if `manifest.json` is missing. Call `report(run)` unless `--skip-report`. Set `MPLCONFIGDIR` to `run/"matplotlib_cache"` and use `matplotlib.use("Agg")`. Figures use `figsize=(7,3.5)` and `savefig(..., bbox_inches="tight")`, with the pilot title suffix "ENGINEERING PILOT".
- [ ] Run on the pilot:
  `.venv/bin/python scripts/paper_results.py artifacts/experiments/e35fd36b944318b1/pilot100_e2`
  Expected: it exits 0, `paper/generated/` holds 11 `.tex` files and 6 `.pdf` files, and the main results show H1P1 NDCG@20 = 0.18216 ± 0.00070 (matching `report.md`).
- [ ] Rerun check: `cp -r paper/generated /tmp/g1`, rerun the command, then `diff -r paper/generated /tmp/g1` to confirm the `.tex` files are identical. PDF differences only in timestamps are fine; set `metadata={"CreationDate": None}` to avoid even those.
- [ ] Partial-run check: copy `pilot100_e2` to the scratchpad, delete `H3P3/seed_*.json` and `H3P3/responses.json`, run with that copy. Expected: exits 0; H3P3 is absent from the main results and `heatmap` shows "–".
- [ ] Run `.venv/bin/ruff check scripts` and the Task 2 test.
- [ ] Commit: `git add scripts/paper_results.py paper/generated && git commit -m "feat: generate paper tables and figures from a run"`

### Task 4: Paper skeleton, macros, bibliography

**Files:** Create `paper/main.tex` and `paper/references.bib`.

- [ ] Preamble:
  - article, a4, 11pt, `geometry` 2.54cm.
  - `iftex`: fontspec under XeTeX, else T1+utf8.
  - Packages: amsmath, amssymb, graphicx, booktabs, float, xcolor, hyperref, listings (same colour scheme as the reference), `tikz` (libraries `positioning,arrows.meta,fit,shapes`), `natbib`, `enumitem`, `caption`.
- [ ] Macros:
  ```latex
  \newif\ifpilotrun
  \def\RunTag{}\def\RunName{none}\def\RunItems{--}\def\RunEpochs{--}
  \InputIfFileExists{generated/run_info.tex}{}{}
  \newcommand{\todo}[1]{\textcolor{red!70!black}{\textbf{[TODO:} #1\textbf{]}}}
  \newcommand{\placeholder}[2][0.9]{\begin{center}\fbox{\parbox[c][4cm][c]{#1\linewidth}{\centering\textit{Image placeholder: #2}}}\end{center}}
  \newcommand{\gentable}[1]{\InputIfFileExists{generated/#1.tex}{}{\placeholder{run scripts/paper\_results.py to generate table #1}}}
  \newcommand{\genfigure}[3][0.85]{\begin{figure}[H]\centering
    \IfFileExists{generated/#2.pdf}{\includegraphics[width=#1\linewidth]{generated/#2.pdf}}{\placeholder{run scripts/paper\_results.py to generate #2}}
    \caption{#3\RunTag}\label{fig:#2}\end{figure}}
  ```
- [ ] The title, the three authors with "Faculty of Computer Science and Engineering (FINKI), Ss. Cyril and Methodius University in Skopje", and the date, followed by empty `\section`s in the spec order.
- [ ] `references.bib` entries:
  - `cui2025colakg` (arXiv:2410.12229, SIGIR'25)
  - `he2020lightgcn`, `wang2019kgat`, `wang2021kgin`, `gao2021simcse`, `liu2019roberta`
  - `harper2015movielens`, `rendle2009bpr`, `jarvelin2002ndcg`, `wei2022cot`
  - `xi2024kar`, `ren2024rlmrec`, `wei2024llmrec`, `maaten2008tsne`, `deepseek2024v3`, `ji2023hallucination`
- [ ] Compile with an empty `generated/`: `mv paper/generated /tmp/gen && (cd paper && tectonic main.tex); mv /tmp/gen paper/generated`. Expected: a PDF is built with placeholder boxes and no errors.
- [ ] Commit: `git add paper/main.tex paper/references.bib && git commit -m "docs: paper skeleton, macros and bibliography"`

### Task 5: Abstract, Introduction, Related Work, Data and KG

**Files:** Modify `paper/main.tex`.

Content requirements (a 4th-year student voice: explains each concept before using it, first person plural, no hype):
- [ ] **Abstract** (~180 words): the problem (LLM-generated item text as side information), what is varied (H1/H2/H3 × P1/P2/P3), what is kept fixed (CoLaKG, split, encoder, user vectors), the evaluation (Recall/NDCG@10/20, 3 seeds, grounding proxies, embedding-space analysis), and "results \todo{}".
- [ ] **Introduction:** motivation; the gap (CoLaKG fixes one prompt and one subgraph recipe); RQ1–RQ5 in an `enumerate`; contributions in an `itemize` that explicitly credits CoLaKG to \citet{cui2025colakg}; paper outline.
- [ ] **Background and Related Work:** subsections for Collaborative filtering and BPR (with the BPR loss equation), Graph-based CF (LightGCN propagation equation), KG-aware recommendation (KGAT, KGIN), LLMs for recommendation (KAR, RLMRec, LLMRec, CoLaKG), Prompting and grounding (CoT, hallucination), and Sentence embeddings (SimCSE contrastive objective equation, RoBERTa).
- [ ] **Data and Knowledge Graph:**
  - MovieLens-1M figures: 6,040 users, 3,260 items, 801,218/197,321 interactions, density ≈ 5.07%, metadata for 3,883 movies.
  - A formal KG definition $\mathcal{G}=\{(h,r,t)\}$ and the 8 relations.
  - A listing of `build_kg` (≈20 lines, lightly trimmed).
  - `\gentable{kg_stats}`, `\gentable{kg_top}`, `\genfigure{degree_distribution}{...}`.
  - A paragraph on hub attributes (genre pairs), which motivates H3.
  - The raw MovieID → row indexing rule with the Toy Story example.
- [ ] Compile; there should be no undefined citations.
- [ ] Commit: `git commit -am "docs: paper introduction, related work, data"`

### Task 6: Method with diagrams

**Files:** Modify `paper/main.tex`.

- [ ] **Pipeline overview:** a TikZ flow diagram (metadata → KG → context selector H → prompt P → DeepSeek → SimCSE → row mapping → CoLaKG ×3 seeds → evaluation → report), with the fixed inputs (split, user vectors) as a side branch.
- [ ] **CoLaKG recap:** equations matching `vendor/CoLaKG/rec_code/model.py::CoLaKG.computer`, covering:
  - the semantic projection $\tilde{\mathbf{s}}_i=\mathrm{ELU}(\mathbf{W}_s\mathbf{s}_i+\mathbf{b})$
  - the fusion $\mathbf{e}_i^{(0)}=(\mathbf{e}_i+\tilde{\mathbf{s}}_i)/2$, and the same for users
  - the top-30 semantic neighbours $\mathcal{N}_i$ by cosine similarity
  - the attention $\alpha_{ij}=\mathrm{softmax}_j(\mathrm{LeakyReLU}(\mathbf{a}^\top[\mathbf{W}\mathbf{s}_j\Vert\mathbf{W}\mathbf{s}_i]))$ and $\mathbf{h}_i=\mathrm{ELU}(\sum_j\alpha_{ij}\mathbf{e}_j^{(0)})$
  - the item update $(\mathbf{e}_i^{(0)}+\mathbf{h}_i)/2$, followed by 3-layer LightGCN mean pooling and the BPR loss with L2 regularisation
  - `\placeholder{CoLaKG architecture sketch}`.
- [ ] **Context strategies:** set definitions:
  - $\mathcal{C}_1(m)$
  - $\mathcal{C}_2(m)=\mathcal{C}_1\cup\bigcup_{a}\mathrm{Sample}_{10}(\{(a,r,m'):m'\neq m\})$ with a seeded RNG
  - $\mathcal{C}_3$ with the four filter rules as an `enumerate`
  - a TikZ figure of one target movie with its attribute nodes (H1), a fan of second-hop movies (H2), and the kept director/actor paths highlighted (H3)
  - `\gentable{example_context}`, `\gentable{context_sizes}`, `\genfigure{context_sizes}{...}`.
- [ ] **Prompt strategies:** `lstlisting` with the COMMON instruction and P1–P3 verbatim from `experiment.py`; a paragraph on why each exists (P2 is structured extraction, CoT-like; P3 is task-oriented); an example user message format; the generation settings (temperature 0, top_p 0.001, max 512 tokens, truncated responses rejected, response caching by SHA-256 request fingerprint).
- [ ] **Encoding and integration:** SimCSE-RoBERTa-large pooler output (1,024-d) at the pinned revision; `embedding_rows.json` mapping; the neighbours are recomputed; user vectors stay fixed.
- [ ] Compile and commit: `git commit -am "docs: paper method and diagrams"`

### Task 7: Setup, Results slots, Discussion, Limitations, Conclusion; final verification

**Files:** Modify `paper/main.tex` and `README.md` (one "Paper" section).

- [ ] **Experimental Setup:**
  - A hyperparameter table: dim 64, 3 layers, lr 1e-3, decay 1e-4, batch 4096, dropouts 0.6, neighbours 30, 2,000 epochs, seeds 42/123/2026, evaluation at the last scheduled epoch with no test-based selection.
  - Recall@K and NDCG@K equations, with training items masked.
  - The contrast definitions as equations mirroring `contrasts()` (RQ1 average over P; RQ2 average over H1,H2; RQ3 difference-in-differences; RQ4 per-P H3−H2), with SD over seeds stated as spread, not significance.
  - Grounding proxies defined (coverage = |mentioned ∩ context| / |context entities|; unsupported = mentioned known entities not in the context).
  - The neighbour-agreement metric equation.
  - Reproducibility (fingerprints, caches, resumable checkpoints).
  - Environment: `\todo{GPU model, CUDA, runtime}`.
- [ ] **Results:**
  - Subsections RQ1–RQ5 plus Embedding space, each containing only `\gentable`/`\genfigure` calls and `\todo{discuss}`.
  - `\ifpilotrun` note box: "The numbers below come from the engineering pilot \RunName{} and are not research results."
  - Includes `main_results`, `heatmap_ndcg20`, `contrasts`, `text_stats`, `grounding`, `cost_quality`, `qualitative`, `embedding_neighbours`, `tsne`.
- [ ] **Discussion:** a subsection per RQ, each with one sentence of what would count as support and `\todo{}` for the finding.
- [ ] **Limitations and Future Work** (written in full):
  - a single dataset, generator and encoder
  - three seeds, so no significance claims; per-user paired tests are future work because they need per-user score logging, which changes training fingerprints
  - exact-match grounding misses aliases
  - the reconstructed KG differs from the paper's reported KG
  - H3 is a hand-designed filter, not learned
  - fixed user vectors
  - cost and provider alias drift (`deepseek-chat` served as `deepseek-flash`)
  - pilots use an ordered first-N subset
- [ ] **Conclusion:** structure plus `\todo{}`.
- [ ] README: add a "Paper" section with the two commands:
  `.venv/bin/python scripts/paper_results.py <run_dir>` and `cd paper && tectonic main.tex`.
- [ ] Final verification:
  - Run the generator on `pilot100_e2`, then `cd paper && tectonic main.tex 2>&1 | grep -Ei "error|undefined|missing"`. Expected: no output.
  - All four test scripts and `ruff check scripts` pass.
  - Look through the PDF pages visually (convert to PNG and view).
- [ ] Commit: `git add -A paper README.md && git commit -m "docs: complete paper sections except results"`
