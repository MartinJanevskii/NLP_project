# Overnight reduced run — record (23–24 September 2026)

This is the run that produced every result in the paper (`paper/main.tex`, Sections 6–7).
The run artifacts live outside Git. Copy the run folder together with the paper if the
results need to be regenerated on another machine.

## What was run

| Item | Value |
|---|---|
| Run folder | `artifacts/experiments/e35fd36b944318b1/overnight_7961f4ac/` |
| Protocol | the same prompts, model alias, encoder and context seed as the pilots (`protocol.json` in the parent folder) |
| Items | 1,000 of 3,260, sampled uniformly with seed 2026 |
| Users | 5,758 (users without a train or test interaction after sampling are removed) |
| Interactions | 230,912 train / 57,528 test (supplied split, restricted); density 5.01% |
| Validation | 23,493 train interactions held out (10% per user), used only for the epoch budget |
| LLM requests | 9,000 (9 configurations × 1,000 items); 243 were already cached from the pilots |
| Served model | `deepseek-flash` (requested alias `deepseek-chat`) |
| Tokens | ≈ 4.1 M input, ≈ 1.4 M output |
| Truncated responses | 0 of 9,000 |
| Epoch budget | E = 400 (the cap); validation NDCG@20 was still rising at epoch 600 (best 0.2274) |
| Training runs | 30 = (CoLaKG reference + 9 configurations) × seeds 42, 123, 2026 |
| Hardware | Apple M4, 10-core CPU, 24 GB; PyTorch 2.14 (MPS), Python 3.13 |
| Wall-clock | 13.5 h in total: ≈ 2 h generation and encoding, 0.5 h budget, 10.4 h training (≈ 3.1 s per epoch) |

## Commands

```sh
uv run --no-sync --env-file .env python scripts/overnight.py check-key
caffeinate -i uv run --no-sync --env-file .env python scripts/overnight.py run
.venv/bin/python scripts/overnight.py status
.venv/bin/python scripts/paper_results.py artifacts/experiments/e35fd36b944318b1/overnight_7961f4ac --skip-report
cd paper && tectonic main.tex
```

## Timeline and incidents

1. **First start (user).** The run stopped during H2P2 generation because one provider
   response was empty. This was a transient error: a 16-request sample of the same
   configuration all finished normally, and none of the final 9,000 responses is truncated.
   *Fix (`dacf13c`):* retries with backoff for transient API errors. A response that stops at
   the 512-token limit would be kept and flagged `truncated: true` (it repeats at temperature 0).
2. **Second start.** Generation and encoding of all nine configurations completed.
3. **Double-training bug found by profiling.** The wall-clock time per epoch was twice the
   measured in-epoch time. `cProfile` showed two `BPR_train_original` calls per epoch: the
   research-mode `else:` branch also ran in reduced mode. *Fix (`e79d033`):* the branch is now
   `elif not args.reduced`, verified by profiling (3 epochs → 3 calls). The run was stopped;
   only the budget run and the partial first reference run were deleted. Generation and
   encoding caches were kept.
4. **Third start.** Budget: 600 epochs, E = 400. Then all 30 runs, then the analysis.
   It completed without errors.

Resuming works throughout: responses and vectors are cached per item, and training
checkpoints after every epoch. A training run resumed on MPS is valid but not bit-identical
(the MPS RNG state is not saved). No completed run in this record was resumed mid-training.

## Outputs in the run folder

| File | Content |
|---|---|
| `findings.json` | everything the paper cites: dataset, epoch budget, metrics per configuration, contrasts, RQ5 text statistics, truncation counts |
| `significance.csv` | pre-registered per-user Wilcoxon tests (Holm), bootstrap CIs, per-seed differences, verdicts |
| `per_seed.csv`, `aggregated.csv`, `contrasts.csv`, `text_analysis.csv`, `qualitative.json`, `report.md` | the standard report (`experiment_report.py`) |
| `progress.jsonl`, `status.json` | the event log and the final heartbeat |
| `epoch_budget.json` | the validation curve and the chosen E |
| `<config>/seed_<s>.json`, `seed_<s>_users.npz`, `seed_<s>.log`, `seed_<s>.pt` | per-run metrics, per-user metrics, log, checkpoint |
| `data/`, `data_val/` | the derived dataset in upstream format, ID maps, sliced published embeddings |

## Results (NDCG@20, pre-registered test, Holm-adjusted)

| Contrast | Δ | 95% CI | p_Holm | Seeds | Verdict |
|---|---|---|---|---|---|
| RQ1 H2−H1 | −0.00052 | [−0.00123, +0.00015] | 0.124 | − − − | inconclusive |
| RQ2 P2−P1 | −0.00053 | [−0.00121, +0.00013] | 0.13 | − + − | inconclusive |
| RQ2 P3−P1 | −0.00001 | [−0.00083, +0.00079] | 1 | + + − | inconclusive |
| RQ3 depth×P2 | −0.00094 | [−0.00218, +0.00032] | 0.13 | − + − | inconclusive |
| RQ3 depth×P3 | +0.00032 | [−0.00102, +0.00163] | 1 | − + − | inconclusive |
| RQ4 H3−H2 (P1) | −0.00080 | [−0.00177, +0.00019] | 0.024 | + − − | inconclusive |
| **RQ4 H3−H2 (P2)** | **+0.00169** | [+0.00076, +0.00268] | **0.004** | + + + | **supported** |
| RQ4 H3−H2 (P3) | −0.00039 | [−0.00146, +0.00061] | 1 | − + + | inconclusive |

Mean NDCG@20 ranges from 0.3356 (H2P2) to 0.3373 (H3P2). The CoLaKG reference is 0.3356.
The paper interprets these numbers in Sections 6–7.
