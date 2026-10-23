# Human-Speech Cleaning Code

This folder contains code used to find and remove detections contaminated by human speech.

Plain meaning: some detector hits were not alpaca hums. This code helped identify those bad hits so they could be removed before the final corpus was used.

Most users do not need to run this. The cleaned result is already included in:

`Published\Data\Corpus construction`

Useful files for developers:

- `code\build_human_contamination_cleaning.py`: builds the cleaning tables.
- `code\apply_posthoc_confirmed_human_overrides.py`: applies manually confirmed removals.
- `tests`: checks for this workflow.
