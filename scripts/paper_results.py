"""Write the paper's LaTeX tables and PDF figures from one saved run.

Reads saved files only: no LLM calls, no training.
Usage: python scripts/paper_results.py artifacts/experiments/<protocol>/<run>
"""

import argparse
import os
from pathlib import Path

from llm_knowledge_enhancement import paper


def main() -> None:
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
    if not args.skip_report:
        from llm_knowledge_enhancement.report import report

        report(run)
    paper.generate(run)
    print("Paper inputs written to", paper.OUT)


if __name__ == "__main__":
    main()
