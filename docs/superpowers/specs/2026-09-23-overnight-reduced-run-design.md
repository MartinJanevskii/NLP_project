# Overnight reduced experiment on the Mac M4 — design

Date: 2026-09-23. Status: awaiting review.

## Goal

Run the complete 9-configuration study (H1–H3 × P1–P3, 3 seeds) on this Mac M4
overnight, with enough statistical power to confirm or reject RQ1–RQ5. Prepare and
verify it; the user starts it. The results must be machine-readable, so Claude can
fill in the paper's Results, Discussion, Abstract and Conclusion afterwards.

Measured on this M4 (upstream model, full data): 25 s per training epoch on MPS,
43 s on CPU, 2.4 s Python negative sampling per epoch, 0.1 s per full scoring
pass. The full protocol would take ~18 days, so it is reduced as described below.

## Constraints

- `vendor/CoLaKG` stays pinned and unmodified. The hyperparameters, prompts,
  encoder and generation settings stay unchanged.
- `experiment.py` and `experiment_report.py` are unchanged. `experiment_train.py`
  gets new **optional** flags only. Pilot behaviour with the existing flags is
  identical (its source hash changes, so rerunning a pilot needs a new
  `--run-tag`; the existing pilot reports stay valid).
- The test set is never used for any choice (epochs, items, configuration).
- Claude does not start the real run.

## 1. Reduced dataset

- Sample `N_ITEMS = 1000` of the 3,260 recommender items uniformly with
  `random.Random(2026)`, from `item_map.txt` order.
- Keep only the supplied `train.txt` / `test.txt` interactions with sampled items.
  Drop users who are left without a train or without a test interaction.
- Re-index users (sorted old IDs) and items (sorted old item IDs) contiguously.
  Write to `artifacts/overnight/<protocol-hash>/data/`:
  - `train.txt` and `test.txt` in upstream format;
  - `item_map.json` (new item → raw MovieID, old item), `user_map.json`;
  - `item_emb_published.pt` and `user_emb_published.pt` (slices of the published
    tensors in new order);
  - `dataset.json` (counts, density, sample seed, sha256 of inputs).
- LLM requests are built with the existing `context()` and `make_payload()` on
  raw MovieIDs, so the response cache in `artifacts/experiments/e35fd36b944318b1/responses`
  and the vector cache are reused as they are (same protocol hash, because the
  prompts, model and inputs are unchanged).

## 2. Epoch budget (validation, never test)

- A validation run: the reference model (published embeddings), seed 2026.
  Hold out 10% of each user's training interactions (at least 1, and only users
  with ≥ 2 training interactions) as validation, with `random.Random(2026)`.
  Train on the rest.
- Evaluate validation NDCG@20 every 10 epochs, up to 600 epochs.
- `E` = the first evaluated epoch with val NDCG@20 ≥ 0.99 × best, capped to
  [50, 400]. Saved in `epoch_budget.json` together with the curve.
- All following runs train on the full reduced training set for exactly `E`
  epochs and are evaluated once on test at the end.

## 3. Runs

- 30 training runs: the reference model (published item embeddings) plus 9
  configurations, × seeds 42, 123, 2026. User semantic vectors are always the
  published ones (sliced).
- Device: MPS if available, else CPU. Recorded per run.
- Each run also logs test NDCG@20 every 50 epochs, **for the learning-curve
  figure only**, labelled as such in the paper.
- Per-user test Recall/NDCG@10/20 saved as `seed_<s>_users.npz` (user IDs plus
  metric arrays).
- Everything can be resumed: completed runs are skipped by fingerprint, and
  interrupted runs resume from per-epoch checkpoints (the existing mechanism).

## 4. Statistics (pre-registered; also written into the paper now)

- Primary metric: per-user NDCG@20, averaged over the 3 seeds per configuration.
- For each of the 8 contrasts of `experiment_report.contrasts()`, apply the same
  weights per user. This gives a per-user difference vector `d`.
- Test: two-sided paired Wilcoxon signed-rank test on `d` (scipy), a Holm
  correction across the 8 contrasts, the mean Δ, and a 95% bootstrap CI (2,000
  resamples, seed 0). Also the 3 per-seed mean Δs.
- Verdict:
  - `supported`: Holm p < 0.05, Δ > 0, and all 3 per-seed Δ > 0;
  - `opposite`: Holm p < 0.05, Δ < 0, and all 3 per-seed Δ < 0;
  - otherwise `inconclusive`.
- The same analysis, secondary, on Recall@20 (reported, no separate verdict).
- RQ5 stays descriptive (existing grounding proxies).

## 5. Observability

- `progress.jsonl`: one JSON line per event (`stage`, `config`, `seed`,
  `epoch`, `loss`, `seconds`, `time`).
- `status.json`: a heartbeat rewritten on every event with `stage`,
  `runs_done/total`, current run, `eta_hours`, `last_error`, `started`, `updated`.
- `python scripts/overnight.py status`: a human summary of `status.json`
  (done / running / failed, ETA, stale-heartbeat warning if > 15 min old, the
  last 5 log lines).
- At the end: `significance.csv` and `findings.json` (per-RQ verdicts, Δ, CI, p,
  per-seed Δ, main metrics per configuration, the epoch budget, the dataset
  summary, runtime).
- `paper_results.py` reads an overnight run directory. It adds
  `significance.tex`, `learning_curves.pdf`, and a dataset-summary table, and
  its existing tables work on the new layout.

## 6. Commands

| Command | Effect | API | Claude runs it? |
|---|---|---|---|
| `overnight.py prepare` | dataset, contexts, requests, scale, 5-batch timing probe, ETA | no | yes |
| `overnight.py rehearse` | the `run` code path into `…/rehearsal/`, on 10 sampled items that are already cached, 2 epochs, 1 seed, 9 configs + reference; then analysis + `paper_results` | no | yes (must pass) |
| `overnight.py check-key` | one tiny API request | 1 call | the user, after adding `.env` |
| `caffeinate -i uv run --no-sync --env-file .env python scripts/overnight.py run` | full overnight run | yes | the user |
| `overnight.py status` | progress summary | no | anyone |

`run` fails in its first seconds if the API key is missing, `vendor` is not
pinned, or `prepare` outputs are missing or stale.

## 7. Paper

- Setup: add a "Reduced protocol" subsection (subsample, validation epoch
  budget, device, pre-registered test and verdict rule). Keep the full-protocol
  text as the design; state clearly which protocol produced the results.
- Results: add `significance` and `learning_curves` slots.

## Done when

- The offline unit tests for re-indexing, validation holdout, epoch choice,
  contrast weights per user, Holm correction and verdict pass.
- `prepare` gives a measured ETA ≤ 10 h (otherwise reduce `N_ITEMS` and say so).
- `rehearse` completes with exit 0, produces `findings.json`, and `paper_results`
  plus the paper compile on it.
- The existing test suite passes.
