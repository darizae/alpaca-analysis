# Hum Acoustic Profile Code

This folder contains R code for measuring acoustic features of alpaca hums.

It is supporting code. The source audio and final research data are not
included in this repository. Configure the script inputs to point to an
authorized local copy of the separately deposited corpus.

Install the required R packages once:

```bash
Rscript research/hum_acoustic_profile/install_deps.R
```

## Run

From inside `corpus_code_research` in the `alpaca-analysis` repository, use R:

```bash
Rscript research/hum_acoustic_profile/run_hum_acoustic_profile.R
```

To process every hum instead of the default sample:

```bash
Rscript research/hum_acoustic_profile/run_hum_acoustic_profile.R --sample-size all
```

Use `--bundle-dir` and `--output-dir` to select authorized external inputs and
an output location before processing audio.
