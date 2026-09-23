# Overnight Reduced Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Native, as chosen for the previous plan). Steps use checkbox (`- [ ]`) syntax.

**Goal:** A prepared, verified overnight run of the full 9-configuration study on a 1,000-item subsample on the M4. It gets pre-registered per-user significance tests and machine-readable findings, and is proven to start and run live before the user leaves it.

**Architecture:**
- New `scripts/overnight.py` covers dataset derivation, the epoch budget, orchestration, status, statistics and findings. It reuses `experiment.build_kg/context/make_payload/encode`, `llm_subset.generate`, and `experiment_report.report`.
- `scripts/experiment_train.py` gains optional reduced-mode flags; pilot behaviour is unchanged.
- Run directory: `artifacts/experiments/<protocol>/overnight_<hash>[_rehearsal]/`, so `report()` finds `entity_labels.json` and `protocol.json` in its parent.

**Tech Stack:** Python 3.13 `.venv`, torch MPS, scipy (Wilcoxon), numpy; tectonic.

**Spec:** `docs/superpowers/specs/2026-09-23-overnight-reduced-run-design.md`

## Global Constraints

- `vendor/` is pinned and unmodified. `experiment.py` and `experiment_report.py` are unchanged; `experiment_train.py` gets new optional flags only.
- No test-set-based choice anywhere.
- The API key is read only from the environment. It is never printed, logged or committed.
- Claude may start the real run only to prove it works, then stops it (the user asked for this); the full run is the user's.

## Review Focus

1. Resume after kill (Ctrl-C or sleep) mid-epoch or mid-generation → rerun continues and no stage is repeated (Task 6 live start → stop → restart).
2. The Mac sleeps → `caffeinate -i` in the documented command, and `status` flags a heartbeat older than 15 min (Task 4).
3. An API error or truncation for one item → the run stops with `last_error` in `status.json` and exit ≠ 0; a rerun retries only the missing items (Task 4 test with a failing fake endpoint).
4. A missing or empty key → `run` exits in < 5 s with a clear message, before any long stage (Task 6).
5. The per-user metric differs from upstream → an assertion at the end of each run compares the means with upstream `test_one_batch` (Task 1).

---

### Task 1: Reduced mode in `experiment_train.py`

**Files:** Modify `scripts/experiment_train.py`. Test `scripts/test_overnight.py` (created here).

**Produces:**
- `rank_metrics(top, truth, ks=(10, 20)) -> dict[str, float]`, for one user: `top` are ranked item IDs, `truth` the test items. It matches upstream `RecallPrecision_ATk` and `NDCGatK_r`.
- CLI flags `--data-dir PATH`, `--reduced`, `--eval-every K`, `--allow-partial`.
- Output `seed_<s>_users.npz` (keys `users`, `recall@10`, `recall@20`, `ndcg@10`, `ndcg@20`), plus a `curve` list in `seed_<s>.json`.
- stdout lines `EPOCH <n>/<E> loss=<x> seconds=<t>` and `CURVE {"epoch":…, "ndcg@20":…, "recall@20":…}`.

- [ ] Test first (in `test_overnight.py`): `rank_metrics([5,1,9], {1,7})` with ks=(2,3). The recall@2 hit is 1 of 2 → 0.5; ndcg@2 = (1/log2 3)/(1+1/log2 3). Also assert equality with upstream `utils.NDCGatK_r` / `RecallPrecision_ATk` on a random 50-user case.
- [ ] Run it → fails (no `rank_metrics`).
- [ ] Implement:
  - `rank_metrics`.
  - The flags and their validation: `--epochs` is free in pilot or reduced mode; `--eval-every` and `--allow-partial` require `--reduced`.
  - Reduced mode sets `world.device` to MPS if available, else CPU, **before** `Loader`.
  - Base item and user tensors come from `<data-dir>/item_emb_published.pt` and `user_emb_published.pt` when present. `raw_to_item` comes from `<data-dir>/item_map.json` when present.
  - The full-coverage check applies unless `--allow-partial`.
  - Data-dir files are included in the fingerprint.
  - The training epoch uses upstream `BPR_train_original` and parses the loss.
  - The curve is evaluated every K epochs, and the curve is stored in the checkpoint so it survives a resume.
  - Final per-user evaluation in batches of 1,024 users, with training positives masked, followed by an assertion that the means equal the upstream `test_one_batch` means within 1e-9.
- [ ] Tests pass. Existing `test_experiment.py` (including its resume path) passes.
- [ ] Commit.

### Task 2: Dataset derivation (`overnight.py`: `sample_items`, `derive`, `holdout`)

**Produces:**
- `sample_items(item_ids, n, seed) -> sorted list`.
- `derive(train_rows, test_rows, keep_items) -> (train, test, user_map, item_map)`. The rows are `{user: [items]}`; users are dropped unless they have both train and test interactions; IDs are re-indexed contiguously in sorted order.
- `holdout(train, fraction, seed) -> (fit, val)`: at least 1 held out, only for users with ≥ 2 interactions, and every user keeps ≥ 1.
- `write_upstream(path, rows)` / `read_upstream(path)`.

- [ ] Tests first:
  - A toy of 4 users × 6 items: sampled items only; a user whose test items were all dropped is removed; IDs are contiguous; train and test stay disjoint; the maps round-trip.
  - Holdout: counts, disjointness, and the same seed gives the same output.
- [ ] Implement, run the tests, commit.

### Task 3: Statistics (`overnight.py`: `CONTRASTS`, `contrast_vector`, `holm`, `verdict`, `analyse`)

**Produces:**
- `CONTRASTS: dict[name, list[(config, weight)]]`, identical to `experiment_report.contrasts()`.
- `contrast_vector(per_user: dict[config, ndarray]) -> dict[name, ndarray]`.
- `holm(p: list) -> list`.
- `verdict(p_holm, delta, seed_deltas) -> "supported"|"opposite"|"inconclusive"`.
- `analyse(run) -> writes significance.csv, findings.json`.

- [ ] Tests first:
  - Toy per-seed metrics fed through both `experiment_report.contrasts()` and `CONTRASTS` give the same means.
  - `holm([0.01, 0.04, 0.03])` equals `[0.03, 0.06, 0.06]`.
  - The verdict truth table.
  - Wilcoxon on an all-zero vector gives p = 1 without crashing.
- [ ] Implement, run the tests, commit.

### Task 4: Orchestration and observability (`overnight.py`: `prepare`, `run`, `rehearse`, `status`, `check-key`)

- [ ] Tests first:
  - `Progress` writes `progress.jsonl` and `status.json`. The ETA is derived from the mean seconds per epoch.
  - `status` reports "STALE" when `updated` is more than 15 min old.
  - A generation failure through a local fake HTTP endpoint that returns 500 sets `last_error` and exits non-zero.
- [ ] `prepare`:
  - Recompute the experiment protocol exactly as `experiment.py` does and require it to equal the existing `protocol.json`.
  - Sample 1,000 items; derive `data/` and `data_val/`, with published slices, `dataset.json`, `item_map.json` and `user_map.json`.
  - Write per-configuration `contexts.json` and `requests.json`, and a `reference/` folder (published embeddings + `embedding_rows.json`).
  - Write `scale.json` with total and cached requests plus a token estimate.
  - Timing probe: a 1-epoch `experiment_train --reduced` on `reference/` into `probe/`.
  - `eta.json` = (600 + 30 × 400) epoch-equivalents × measured seconds + generation estimate (misses × 1.5 s ÷ 4 workers) + encoding estimate (misses × measured encoder seconds per text).
  - Print a summary.
- [ ] `run` stages, each skipped if already complete:
  1. **Preflight:** the key is non-empty, vendor is pinned, prepare outputs are fresh.
  2. **Budget:** `reference` on `data_val/`, 600 epochs, eval-every 10, early stop after 10 evaluations without improvement (validation-only). `E` by the spec rule → `epoch_budget.json`.
  3. **Per configuration:** generate (4 workers, shared cache) → `responses.json` → `encode()`.
  4. **Train:** `reference` + 9 configurations × 3 seeds for E epochs, streaming the subprocess stdout into `seed_*.log` and `Progress`.
  5. `report(run)` → `analyse(run)` → `status: COMPLETE`.
- [ ] `rehearse`: the same `run` code in `…_rehearsal/`.
  - 10 sampled items that have cached responses for all 9 configurations; the other rows keep the published vectors (`--allow-partial`).
  - Budget: 20 epochs, eval-every 10; training: 2 epochs; 3 seeds.
  - No key required (errors if any request is not cached).
- [ ] `check-key`: one 5-token request, printing only OK/FAIL and the served model name.
- [ ] Commit.

### Task 5: Paper and results generator

- [ ] `paper_results.py`:
  - Detect `purpose` `reduced` / `rehearsal`: `RunTag` stays empty for reduced and is ` [REHEARSAL]` for rehearsal; set `\reducedruntrue`.
  - Build the reference row from `reference/seed_*.json`.
  - New outputs: `dataset_summary.tex` (from `data/dataset.json`), `significance.tex` (from `significance.csv`) and `learning_curves.pdf` (the budget validation curve plus the test curves).
  - Fix the embedding-space published slice to map raw MovieIDs through `item_map.txt`.
  - Test the new tables' pending paths.
- [ ] `main.tex`:
  - A "Reduced protocol" subsection with the pre-registered test and verdict rule.
  - The results box switches on `\ifreducedrun`.
  - Slots for `dataset_summary`, `significance` and `learning_curves`.
- [ ] Compile, commit.

### Task 6: Verification (Claude runs all of this)

- [ ] `overnight.py prepare` → report the ETA. If it is over 10 h, reduce `N_ITEMS` (a ruling) and re-prepare.
- [ ] `overnight.py rehearse` → exit 0; `findings.json` exists; `paper_results.py` on the rehearsal plus the paper compile.
- [ ] Once the user has filled in `.env`: `overnight.py check-key` → OK.
- [ ] Live start: launch the real command in the background, watch `status` until the budget stage is running and at least one real generation batch has completed (≥ 10 min). Then stop it and rerun `status`. Restart briefly to confirm it resumes without repeating work, then stop again.
- [ ] Full suite (all 5 test scripts, ruff on the new and changed files, paper compile). Final review by an opus subagent, then fixes.
