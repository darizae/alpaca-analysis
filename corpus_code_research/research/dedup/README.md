# Duplicate-Removal Code Notes

This folder contains the code used to study and remove duplicate recordings of the same alpaca hum.

The cleaned corpus and its source audio are not included in this software
repository. Obtain the separately governed research-data deposit, then set
`input_table` and `source_wav_index` in a copy of one of the YAML templates
under `../../config/`.

From the `alpaca-analysis` repository root, install the corpus dependencies as
described in the root README and run, for example:

```bash
export PYTHONPATH="$PWD/corpus_code_research${PYTHONPATH:+:$PYTHONPATH}"
python -m corpus_dedup.cli prepare-features \
  --config corpus_code_research/config/local_smoke.yaml
```

Continue with `run-features`, `finalize-features`, `prepare-grid`, `run-grid`,
`summarize-run`, and `export-top-exemplars`; use `python -m
corpus_dedup.cli --help` for arguments. The `local_smoke.yaml` file is only a
configuration template and needs local external inputs. The HPC templates and
submission scripts additionally require a configured SLURM cluster.
