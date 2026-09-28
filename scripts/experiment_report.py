"""Write the CSV tables, plots and report.md of one run directory."""

import argparse
from pathlib import Path

from llm_knowledge_enhancement.report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing manifest.json")
    report(parser.parse_args().run.resolve())


if __name__ == "__main__":
    main()
