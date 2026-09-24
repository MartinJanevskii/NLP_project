# Recommendation system 

## Intro

Project for the course NLP at university FINKI. 

## Required tooling 

The project, atm uses the following tools: 

- postgres(vector db)
- neo4j(graph db)
- uv 
- dvc
- docker 

Please install them so that you can start working on the project. 

## Quickstart 

When first starting off with the project, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and execute `uv sync `. This will download all the necessary packages, defined in the [pyproject.toml](./pyproject.toml). Also don't forget to activate the python venv. 

The project structure is the following: 
- /model 
    - This is the folder for versioning model weights. Files like `pkl`,`pt` will be stored here. 
- /reports 
    - The folder will be store `csv`,`json` files which contain results, from the experiment runs, for the models. 
- /notebooks 
    - The folder will store notebooks that do some kind of EDA either on datasets(the data folder) or the reports
- /src 
    - The folder will store all the code logic. Logic for defining data pipelines, training loops, test loops...; will be stored here
- /data
    - The folder will contain all the raw and processed that on which the model will be trained. 
- /db 
    - The folder will contain configuration files affecting the vector / graph databases. For example, we have the [init.sql](./db/vector/init.sql)  

Every folder, listed from above, will be versioned with dvc and stored remotely on the `DagsHub` [backend](https://dagshub.com/viki123v/llm-knowledge-enhancement/src/feature/NLP-13/s3:/llm-knowledge-enhancement). This will allow us to bypass git's default max file size and store indiscriminate number of files of enormous sizes(hopefully).

To be in sync with the data stored in the `DagsHub` backend use `dvc`. `DVC` works similar to git, so all the commands you know and love will still work. For example, for pulling changes for the data, you should use `dvc pull`. The configuration for dvc is defined in the [this](./.dvc/config) config file. For security reasons you need to fill in the [config.local.sample](./.dvc/config.local.sample) to access the backend. If you a contributor to the project, you can find the access credentials [here](https://dagshub.com/viki123v/llm-knowledge-enhancement/src/feature/NLP-13/s3:/llm-knowledge-enhancement). 

To start the databases, firstly fill in the placeholder values in [.env.sample](.env.sample). After that run `docker compose up` and this should start the databases with all the configuration. 

## Paper

The LaTeX paper lives in [paper/](paper/). Every table and figure in it is generated
from a saved experiment run; nothing is typed in by hand.

```sh
# 1. Regenerate tables/figures from a run (pilot now, full/ later). No API calls, no training.
.venv/bin/python scripts/paper_results.py artifacts/experiments/<protocol>/<run>
# 2. Compile (install once with: brew install tectonic)
cd paper && tectonic main.tex
```

On a pilot run every generated caption is marked **[ENGINEERING PILOT]**. The paper
compiles without any run too: missing inputs show as placeholder boxes. Result-dependent
text is marked with red `\todo{}` notes. Offline check: `.venv/bin/python scripts/test_paper_results.py`.

## Overnight reduced run (Mac M4)

A complete study (9 configurations + CoLaKG reference, 3 seeds) on a random 1,000-item
subsample with a validation-chosen epoch budget; ~13 h on an M4. Resumable at any point.

```sh
uv run --no-sync --env-file .env python scripts/overnight.py check-key   # one tiny API call
caffeinate -i uv run --no-sync --env-file .env python scripts/overnight.py run
.venv/bin/python scripts/overnight.py status                             # from another terminal
.venv/bin/python scripts/paper_results.py artifacts/experiments/<protocol>/overnight_<hash> --skip-report
```

Results: `findings.json` and `significance.csv` in the run folder (pre-registered per-user
Wilcoxon tests with Holm correction). Offline checks: `.venv/bin/python scripts/test_overnight.py`.
