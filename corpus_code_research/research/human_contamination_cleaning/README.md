# Human-Speech Cleaning Code

This folder contains code used to find and remove detections contaminated by human speech.

Plain meaning: some detector hits were not alpaca hums. This code helped identify those bad hits so they could be removed before the final corpus was used.

No review bundle, source audio, metadata table, or cleaned result is included
in this repository. These workflows require the separately governed research
artifacts produced by the detection/curation process.

From the `alpaca-analysis` repository root, install the pinned requirements
listed in this directory as shown in the root README. Example entry points are:

```bash
python corpus_code_research/research/human_contamination_cleaning/code/backfill_inference_metadata.py \
  --review-bundle /path/to/review-bundle \
  --ssh-host your-hpc-login

python corpus_code_research/research/human_contamination_cleaning/code/build_human_contamination_cleaning.py \
  --review-bundle /path/to/review-bundle
```

Metadata backfill requires either reachable local metadata inputs or authorized
SSH access to the configured raw-recording store. Audio exports are optional
and must only be generated where the data permissions allow them.

Useful files:

- `code/build_human_contamination_cleaning.py`: builds the cleaning tables.
- `code/apply_posthoc_confirmed_human_overrides.py`: applies manually confirmed removals.
- `code/verify_cleaned_metadata_backfill.py`: audits the retained hum-only backfill.
- `tests`: checks for this workflow.
