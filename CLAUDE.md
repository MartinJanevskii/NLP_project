# CLAUDE.md

Research code for an NLP course paper: how knowledge-graph context (H1–H3) and prompt
strategy (P1–P3) for an LLM change item descriptions and CoLaKG recommendation quality.
See README.md for setup and the full run.

## Layout

- `src/llm_knowledge_enhancement/` — all shared logic (installed editable by `uv sync`)
- `scripts/` — thin CLI entry points; keep their flags and output unchanged
- `tests/` — plain assert scripts: `for t in tests/test_*.py; do uv run python "$t"; done`
- `vendor/CoLaKG/` — upstream CoLaKG pinned at `be3aa19`; never modify it (runs check this)
- `artifacts/` — run outputs and paid response caches, not in git
- `paper/` — LaTeX; `paper/generated/` is written only by `scripts/paper_results.py`

## Rules

- Never change anything that feeds a hash unless a new experiment is intended:
  `prompts.py` strings and payload layout (response-cache key), `protocol.build_protocol`
  (experiment folder name), `overnight.SETTINGS` (run folder name), the context seed in
  `kg.py`, and the encoder name/revision (vector-cache key). A change silently means new
  paid generations and new folders.
- `training.py` hashes its own source into every run fingerprint; editing it makes
  existing run folders non-resumable (results stay readable).
- Keep JSON/CSV key order and formatting stable: the paper is regenerated from them.
- Upstream modules parse `sys.argv` on import: call `colakg.configure(...)` before
  importing `world`, `utils`, `model`, `dataloader` or `Procedure`.
- Style: `uv run ruff check` and `uv run ruff format`; comments only for non-obvious reasons.
