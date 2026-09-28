"""Overnight reduced experiment: 1,000-item subsample, validation-chosen epoch
budget, CoLaKG reference + 9 configurations x 3 seeds, per-user significance tests.

Commands (run from the repository root):
  prepare    build the subsample, contexts and requests (no API, no training)
  run        everything, resumable: prepare -> generate -> encode -> budget -> train -> analyse
  status     human-readable progress (safe to run any time, from another terminal)
  check-key  one tiny API request to confirm the key works
"""

import argparse

from llm_knowledge_enhancement import overnight
from llm_knowledge_enhancement.llm import api_key
from llm_knowledge_enhancement.progress import status_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("prepare", "run", "status", "check-key"))
    args = parser.parse_args()
    if args.command == "status":
        print(status_text(overnight.run_dir()))
        return
    key = api_key()
    if args.command in ("run", "check-key") and not key:
        parser.error(
            "No API key: put DEEPSEEK_API_KEY in .env and use 'uv run --env-file .env'"
        )
    if args.command == "prepare":
        overnight.prepare(overnight.run_dir())
    elif args.command == "check-key":
        overnight.check_key(key)
    else:
        overnight.check_vendor()
        overnight.run_all(overnight.run_dir(), key)


if __name__ == "__main__":
    main()
