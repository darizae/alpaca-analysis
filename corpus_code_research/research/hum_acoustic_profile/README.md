# Hum Acoustic Profile Code

This folder contains R code for measuring acoustic features of alpaca hums.

It is supporting code. The final thesis data is already in `Published\Data`.

## Run

From inside `corpus_code_research` in the `alpaca-analysis` repository, use R:

```bat
Rscript research\hum_acoustic_profile\run_hum_acoustic_profile.R
```

To process every hum instead of the default sample:

```bat
Rscript research\hum_acoustic_profile\run_hum_acoustic_profile.R --sample-size all
```
