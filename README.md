# Alpaca Analysis

Analysis and research code for the alpaca hum study.

## Layout

- `corpus_code_research`: corpus cleaning, duplicate removal, and supporting research workflows.
- `spectrotemporal-visualization-analysis-subset`: H1/H2 analysis backend, scripts, tests, and analysis notes.

## Setup

From this repository's root:

```bat
python -m pip install -e "spectrotemporal-visualization-analysis-subset/backend[dev]"
set PYTHONPATH=%CD%\corpus_code_research;%PYTHONPATH%
```

Run the analysis tests with:

```bat
python -m pytest spectrotemporal-visualization-analysis-subset\backend\tests
```

See each subdirectory's README for its specific workflows.
