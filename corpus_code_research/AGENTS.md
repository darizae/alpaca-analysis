# AGENTS.md

At the start of every new session in this repo, before making changes or running checks:

1. Confirm that this `AGENTS.md` is active for the current working directory.
2. Restate these rules in 5 bullets or fewer:
   - Never run `npm run build` without my permission.
   - No hacks.
   - No fallback mechanisms.
   - Ask before commit.
   - Ask before push.
3. State that noisy non-interactive shell output must go through `distill` using strict output contracts.
4. For repeated workflows, prefer the predefined distill shapes in this file instead of inventing a new loose prompt.
---

## Project context

This repo is a sandbox for bioacoustics data wrangling and scientific data analysis around the Alpaca deep-learning workflow.

Use this repo for smaller, messier, research-facing work that does not belong directly in the pipeline or product UI: inspecting WAVs and review bundles, reorganizing mappings, validating artifact folders, analyzing prediction/RF outputs, preparing or importing Raven/selection-table/continuous-review files, counting and plotting labels, and leaving reproducible markdown plans, notes, results, and generated artifacts.

Treat source WAVs, JSON/TSV manifests, review packages, derived audio, tables, plots, and markdown writeups as first-class research artifacts. Prefer explicit, auditable scripts and markdown-tracked reasoning/results over hidden one-off steps. Keep outputs compatible with downstream Alpaca training/curation ingestion; do not silently overwrite artifacts or invent path mappings when existing metadata can be used.

Every process should be reproducible and repeatable, with deterministic outputs organized into hierarchical folders.

The current codebase already contains a review-organizer style app for loading a review root, reading `mappings.json`, grouping source WAVs by collection, tracking/uploading review artifacts, saving notes/status, validating folder state, and exporting/importing continuous-review artifacts. Extend that direction pragmatically: this repo should stay useful as an agent-assisted lab notebook plus artifact workspace for bioacoustics curation and analysis.

---

## Current research-artifact state

The repo now has a Python metadata-backfill workflow for inference review bundles:

* CLI entrypoint: `research/human_contamination_cleaning/code/backfill_inference_metadata.py`
* Package: `research/human_contamination_cleaning/code/labeller_helper_metadata_backfill/`
* Default outputs: `artifacts/full_inference_runs/<bundle-name>/metadata_backfill/`
* Core outputs include:

  * `enriched_detections.csv/.parquet`
  * `source_wav_metadata_index.csv/.parquet`
  * `source_wavs.csv`
  * `join_audit.md`
  * `manifest.json`
  * pulled recorder metadata files and SHA-256 checksums

The backfill links review-bundle detections to HPC raw-recording metadata: source WAV, collection, recorder CSV metadata file, absolute detection time, GPS-nearest row, recorder/device metadata, firmware, settings hash, QC fields, and metadata source notes.

There is also a human-contamination cleaning workflow:

* CLI entrypoint: `research/human_contamination_cleaning/code/build_human_contamination_cleaning.py`
* Package: `research/human_contamination_cleaning/code/labeller_helper_human_contamination/`
* Default outputs: `research/human_contamination_cleaning/<bundle-name>/human_contamination_cleaning/`

There is also a cleaned-backfill verifier:

* CLI entrypoint: `research/human_contamination_cleaning/code/verify_cleaned_metadata_backfill.py`
* Purpose: audit the hum-only retained backfill for coverage and integrity of pulled metadata, GPS, and absolute timestamps.

---

## Research workflow principles

For new analysis modules:

* Keep original review artifacts immutable.
* Prefer consuming existing canonical outputs, especially `enriched_detections.parquet`, rather than reparsing or re-pulling unless necessary.
* Put run-scoped derived artifacts under the relevant run folder or a clearly linked deterministic artifact folder.
* Record input paths, hashes, command arguments, schema versions, and output paths in a `manifest.json`.
* Use stable row keys, stable sorting, stable column ordering, fixed random seeds, and explicit configs.
* Make outputs both machine-readable and human-readable: Parquet/CSV/JSON plus Markdown reports and plots.
* Favor small modular Python packages over large one-off scripts.
* Design tables so future GPS/time analyses and contamination-review work can be added without rewriting the pipeline.

A useful emerging pattern is:

```text
review bundle
  → metadata backfill
  → detection-level metadata map
  → acoustic features / clustering / reports / visual exploration
```

The next layer should formalize the detection-level metadata map as a named, queryable research artifact rather than leaving it implicit inside backfill outputs.
