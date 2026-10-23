# Alpaca Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23599881.svg)](https://doi.org/10.5281/zenodo.23599881)

Research software for corpus curation and the statistical analyses used to
study relationships between alpaca hum vocalizations, GPS-derived movement,
and group dispersion. This repository supports the analysis stage of the
research; detection and initial curation are implemented separately in
[`alpaca-pipelines`](https://github.com/darizae/alpaca-pipelines).

## Repository layout

- `corpus_code_research`: deterministic corpus cleaning, metadata backfill,
  human-speech contamination review, duplicate-removal workflows, and an R
  acoustic-profile workflow.
- `spectrotemporal-visualization-analysis-subset/backend`: the Python package
  (`spectrotemporal-backend`, version 0.1.0) that prepares analysis-base tables
  and runs the retained H1 and H2 scientific workflows.
- `spectrotemporal-visualization-analysis-subset/docs`: analysis plans and
  artifact contracts.
- `spectrotemporal-visualization-analysis-subset/scripts`: Bash/SLURM wrappers
  for long-running analysis stages.

No source audio, GPS research records, cleaned corpus, trained models, manual
review packages, or manuscript result bundles are distributed here. The
separate research-data deposit is maintained by the data custodians; this
repository does not claim a dataset DOI.

## Requirements and installation

The retained analysis backend requires Python 3.12.x. The lockfile is the
authoritative Python environment; [uv](https://docs.astral.sh/uv/) is the
recommended installer.

From the repository root:

```bash
uv sync \
  --project spectrotemporal-visualization-analysis-subset/backend \
  --extra dev \
  --locked
```

The corpus workflows have a separate pinned requirements file:

```bash
python3.12 -m venv corpus_code_research/.venv
source corpus_code_research/.venv/bin/activate
python -m pip install \
  -r corpus_code_research/research/human_contamination_cleaning/requirements.txt
```

The final statistical model stages and acoustic-profile workflow also require
R plus the packages installed by
`corpus_code_research/research/hum_acoustic_profile/install_deps.R`. HPC
wrappers require Bash, OpenSSH/rsync, a SLURM client, and access to a configured
cluster.

## Configure external inputs

Copy `spectrotemporal-visualization-analysis-subset/backend/config/default.toml`
to an untracked local path and replace the values in `[paths]`, `[remote]`, and
`[analysis_hpc]`. In particular, `raw_corpus_dir` must point to the externally
obtained corpus-construction deposit and its metadata-backfill tables. The
checked-in paths and cluster names are templates, not public resources.

Likewise, the YAML files in `corpus_code_research/config/` are templates. Set
their `input_table` and `source_wav_index` to local copies of the required
external Parquet tables before running a deduplication workflow.

## Analysis backend workflow

Use the local config path in every command:

```bash
BACKEND=spectrotemporal-visualization-analysis-subset/backend
CONFIG=/absolute/path/to/local.toml

uv run --project "$BACKEND" stv-backend --config "$CONFIG" build-dataset
uv run --project "$BACKEND" stv-backend --config "$CONFIG" build-analysis-base
uv run --project "$BACKEND" stv-backend --config "$CONFIG" verify-dataset
```

Run `fetch-sidecars` first only when the required AudioMoth recorder sidecars
are not already present locally; it requires SSH access to the configured raw
audio store.

Create an immutable H1 or H2 launch from the analysis-base manifest:

```bash
ANALYSIS_ID=rq1-post-hum-cohesion  # or rq2-hum-timing-group-dispersion
BASE_MANIFEST=/absolute/path/to/analysis/base/manifest.json

uv run --project "$BACKEND" stv-analysis --config "$CONFIG" \
  launch create "$ANALYSIS_ID" --label release-run --profile full \
  --input-manifest "$BASE_MANIFEST"
```

Then execute stages in the order recorded in the launch manifest:

```bash
uv run --project "$BACKEND" stv-analysis --config "$CONFIG" \
  launch run-stage "$ANALYSIS_ID" --launch-dir /path/to/launch \
  --stage units
```

See `spectrotemporal-visualization-analysis-subset/PRIMER.md` and the per-analysis
artifact contracts for complete stage lists. The Bash wrappers in `scripts/`
prepare, submit, monitor, and retrieve the computationally intensive stages on
SLURM.

## Corpus workflows

With the corpus virtual environment active and the repository root on
`PYTHONPATH`, the deduplication CLI can be inspected or run against a completed
local config:

```bash
export PYTHONPATH="$PWD/corpus_code_research${PYTHONPATH:+:$PYTHONPATH}"
python -m corpus_dedup.cli prepare-features \
  --config corpus_code_research/config/local_smoke.yaml
```

Human-contamination and metadata-backfill entry points live under
`corpus_code_research/research/human_contamination_cleaning/code/`. Their local
READMEs describe required review bundles, metadata tables, and output layouts.

## Tests

The tests use synthetic or minimal fixtures and do not need the restricted
research dataset or an HPC account:

```bash
uv run --project spectrotemporal-visualization-analysis-subset/backend \
  pytest spectrotemporal-visualization-analysis-subset/backend/tests

(
  cd corpus_code_research
  PYTHONPATH=. .venv/bin/python -m pytest \
    tests \
    research/human_contamination_cleaning/tests
)
```

This repository currently has no pre-commit configuration.

## Reproducibility boundaries

The software environment and automated tests are reproducible from this
repository. End-to-end corpus reconstruction and manuscript-result
reproduction are not possible without the separately governed audio, GPS,
manual-review, metadata-backfill, and intermediate analysis artifacts. Full H1
and H2 model fitting additionally depends on the documented R environment and,
for the canonical large runs, site-specific HPC resources. Example configs do
not contain substitute research data.

## Citation and license

Citation metadata are in [`CITATION.cff`](CITATION.cff). Version 0.1.0 is
archived on Zenodo as
[`10.5281/zenodo.23599881`](https://doi.org/10.5281/zenodo.23599881).

This software is licensed under the [GNU General Public License v3.0
only](LICENSE).
