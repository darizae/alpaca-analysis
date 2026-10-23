# Data Contract v4

**Any backend output contract change requires updating this file in the same change. There is no backwards compatibility guarantee. Frontend code should treat this document and `manifest.json` as the current source of truth.**

## Purpose

This document explains the processed backend outputs under `backend/data/processed/` for the frontend implementer.

Read this first, then inspect:

- [manifest.json](data/processed/manifest.json)
- [paddock_zones.geojson](data/processed/paddock_zones.geojson)
- [paddock_zone_evidence.geojson](data/processed/paddock_zone_evidence.geojson)
- [paddock_inference_boundary.geojson](data/processed/paddock_inference_boundary.geojson)
- one day index, for example [2025-02-03/index.json](data/processed/days/2025-02-03/index.json)
- one recorder shard, for example [2025-02-03/recorders/audio_collection_388_m32_20250213.json](data/processed/days/2025-02-03/recorders/audio_collection_388_m32_20250213.json)
- the build logic in [pipeline.py](src/stv_backend/pipeline.py) and [paddock_zones.py](src/stv_backend/paddock_zones.py)

## Core Model

- Canonical display timezone: `Australia/Adelaide`
- Contract version: `5`
- UI collection identity: `recorder_id`
- Day index: one file per `date_local`
- Recorder shard: one file per `date_local` + `recorder_id`
- Segment: one source WAV recording window, keyed by `segment_id`
- Session: one alive recorder window, keyed by `session_id`
- Day session: one day-level alive-window identity, keyed by `day_session_id`
- Event: one hum detection, keyed by `event_id`
- Trace: one GPS array payload per segment, keyed by `segment_id` and `session_id`
- Audio: one precomputed WAV clip per event
- Spectrogram: one precomputed PNG per event
- Paddock zone: one app-facing polygon feature keyed by `zone_id`

Identity semantics:

- `recorder_id`: stable selector key, currently the full `audio_collection_*` slug
- `recorder_display_name`: short display token such as `388`, `TINA`, `PINK`
- `segment_id`: `${recorder_id}:${wav_basename}`
- `session_id`: `${date_local}:${recorder_id}:session-NNN`
- `day_session_id`: `${date_local}:session-NNN`
- `event_id`: stable event key, sourced from `review_row_key`
- `zone_id`: stable zone key inside the generated zone layer

## File Layout

The frontend should read only from:

- `backend/data/processed/manifest.json`
- `backend/data/processed/paddock_zones.geojson` when `manifest.paddock_zones` is present
- `backend/data/processed/days/YYYY-MM-DD/index.json`
- `backend/data/processed/days/YYYY-MM-DD/recorders/<recorder_id>.json`
- `/api/audio/clips/<clip_key>` for playback

The current app should not load `paddock_hull.geojson`. That file remains a backend inference/source-record artifact.

The frontend should not depend on raw corpus files, cached recorder metadata files, or intermediate parquet files.

## Shared Types

### `counts`

Appears in `manifest.json` and day indexes.

Fields:

- `recorders`
- `segments`
- `trace_points`
- `trace_segments`
- `events`

### `availability`

Appears in day indexes and recorder shards.

Fields:

- `date_local`
- `timezone`
- `alive_windows`
- `dead_windows`

Each window has:

- `session_id`
- `start_s`
- `end_s`
- `duration_s`
- `source_count`

Semantics:

- `alive_windows` are local-day second ranges where recording data exists.
- `dead_windows` are complementary local-day ranges where no recording should be displayed.
- Day index availability is merged across recorders for timeline display.
- Recorder shard availability is recorder-specific and is authoritative for traces, events, and current positions.

### `trace_encoding`

Appears in day indexes and recorder shards.

Fields:

- `kind`: currently `per-segment-array`
- `time_unit`: currently `seconds_since_midnight_local`
- `coordinate_order`: currently `["latitude", "longitude"]`

### `spatial_thresholds`

Top-level manifest field.

- `polygon_link_meters`: `{ min, max, step, default, low_zoom_min }`
- `company_distance_meters`: `{ min, max, step, default }`

Frontend controls should use these values instead of hardcoded thresholds.

## `manifest.json`

Top-level keys:

- `dataset_name`
- `version`
- `timezone`
- `source_date_generated_utc`
- `available_days`
- `recorders`
- `day_files`
- `spatial_thresholds`
- `counts`
- `bbox`
- `paddock_hull_path`
- `audio_export`
- `audio_clips`
- `paddock_zones`

Important fields:

- `version`: must be `5`
- `available_days`: ordered Adelaide-local day strings
- `recorders`: selector metadata
- `day_files`: day index paths and per-day counts
- `bbox`: initial map fit bounds
- `paddock_hull_path`: backend source-record path only
- `audio_export`: concat WAV source-record only
- `audio_clips`: generated clip directory and count
- `paddock_zones`: either `null` or the zone manifest below

### `recorders[]`

Fields:

- `recorder_id`
- `recorder_display_name`
- `device_id`

### `day_files[]`

Fields:

- `date_local`
- `path`
- `recorder_count`
- `segment_count`
- `trace_point_count`
- `trace_segment_count`
- `event_count`

### `bbox`

Fields:

- `min_lon`
- `max_lon`
- `min_lat`
- `max_lat`

### `audio_export`

Fields:

- `path`
- `row_count`
- `duration_s`
- `gap_duration_s`

### `audio_clips`

Fields:

- `path`
- `count`

### `paddock_zones`

When present, this section points to the generated paddock-zone artifacts.

Fields:

- `method_version`
- `display_path`
- `evidence_path`
- `inference_boundary_path`
- `state_point_path`
- `event_point_path`
- `coverage_summary_path`
- `display_count`
- `evidence_count`
- `excluded_moth_numbers`

## `days/YYYY-MM-DD/index.json`

Top-level keys:

- `date_local`
- `timezone`
- `day_start_local`
- `day_end_local`
- `trace_encoding`
- `availability`
- `recorder_ids`
- `recorder_files`
- `sampled_positions`
- `event_time_summaries`
- `counts`
- `company_summaries`

Frontend use:

- populate the day selector
- render the availability-aware time slider before recorder shards are loaded
- determine which recorder shards to load
- preserve selected recorders across day changes when the recorder still exists that day

### `recorder_files[]`

Fields:

- `recorder_id`
- `recorder_display_name`
- `path`
- `segment_count`
- `trace_point_count`
- `trace_segment_count`
- `event_count`

### `sampled_positions[]`

Fields:

- `recorder_id`
- `recorder_display_name`
- `session_id`
- `day_session_id`
- `sample_s`
- `latitude`
- `longitude`

### `event_time_summaries[]`

Fields:

- `recorder_id`
- `recorder_display_name`
- `day_session_id`
- `event_times_s`

## `days/YYYY-MM-DD/recorders/<recorder_id>.json`

Top-level keys:

- `date_local`
- `timezone`
- `day_start_local`
- `day_end_local`
- `trace_encoding`
- `availability`
- `recorder_id`
- `recorder_display_name`
- `segments`
- `trace_segments`
- `events`
- `sampled_positions`
- `event_time_summaries`
- `movement_summaries`
- `high_hum_intervals`
- `company_summaries`

### `segments[]`

Fields:

- `segment_id`
- `session_id`
- `day_session_id`
- `recorder_id`
- `recorder_display_name`
- `device_id`
- `wav_basename`
- `source_audio_file`
- `recording_start_utc`
- `recording_end_utc`
- `recording_start_local`
- `recording_end_local`
- `recording_start_seconds_local`
- `recording_end_seconds_local`
- `duration_s`
- `gps_valid_fraction`
- `first_valid_latitude`
- `first_valid_longitude`
- `last_valid_latitude`
- `last_valid_longitude`

### `trace_segments[]`

Fields:

- `segment_id`
- `session_id`
- `day_session_id`
- `recorder_id`
- `recorder_display_name`
- `wav_basename`
- `point_count`
- `times_s`
- `latitudes`
- `longitudes`

Array semantics:

- `times_s[n]`, `latitudes[n]`, and `longitudes[n]` refer to the same GPS fix.
- `times_s` is Adelaide-local seconds since midnight.
- Arrays are already ordered.
- Frontend code must suppress trace/current-position state when the current slider time is outside the trace session's alive window.

### `events[]`

Fields:

- `event_id`
- `segment_id`
- `session_id`
- `day_session_id`
- `recorder_id`
- `recorder_display_name`
- `device_id`
- `wav_basename`
- `source_audio_file`
- `timestamp_start_utc`
- `timestamp_end_utc`
- `timestamp_start_local`
- `timestamp_end_local`
- `seconds_since_midnight_local`
- `latitude`
- `longitude`
- `coordinate_source`
- `nearest_trace_offset_s`
- `detection_score`
- `label`
- `source_start_s`
- `source_end_s`
- `export_selection`
- `export_begin_time_s`
- `export_end_time_s`
- `clip_key`
- `clip_path`
- `clip_url_path`
- `clip_duration_s`

Frontend use:

- draw hum markers inside the current active window and alive session
- open event detail drawers
- play audio from `clip_url_path`

### `movement_summaries[]`

One row per recorder session.

Fields:

- `recorder_id`
- `recorder_display_name`
- `session_id`
- `day_session_id`
- `start_s`
- `end_s`
- `distance_m`
- `duration_s`
- `speed_mps`
- `state`: `moving`, `still`, or `unknown`

### `high_hum_intervals[]`

Top hum-density windows per recorder session.

Fields:

- `recorder_id`
- `recorder_display_name`
- `session_id`
- `day_session_id`
- `start_s`
- `end_s`
- `event_count`

### `company_summaries[]`

Nearby-recorder pair summaries.

Fields:

- `day_session_id`
- `recorder_ids`
- `recorder_display_names`
- `threshold_m`
- `together_duration_s`
- `sample_count`
- `min_distance_m`
- `mean_distance_m`
- `duplicate_hum_count`

## `paddock_zones.geojson`

App-facing non-overlapping paddock-zone display layer.

Top-level keys:

- `type`: `FeatureCollection`
- `metadata`
- `features`

Semantics:

- This layer is the only zone layer the current app should render.
- It is a paddock partition for display, not a claim that every polygon is high-confidence behavior truth.
- `uncategorized` polygons are explicit remainder areas and are valid features.

### Zone feature

Each feature has:

- `type`
- `id`
- `properties`
- `geometry`

`geometry.type` may be `Polygon` or `MultiPolygon`.

### Zone properties

- `zone_id`
- `zone_label`
- `dominant_behavior_code`
- `dominant_behavior_name`
- `confidence`: `high`, `medium`, `low`, or `uncategorized`
- `behavior_purity`
- `behavior_lift`
- `n_gps_points`
- `n_intervals`
- `n_sessions`
- `n_animals`
- `n_dates`
- `top_point_events`
- `qa_flags`
- `source_session_ids`
- `source_moth_numbers`
- `toggle_default`
- `zone_color_key`
- `area_m2`
- `evidence_kind`

Frontend use:

- key per-zone toggles by `zone_id`
- resolve color in the frontend from `zone_color_key`
- default `toggle_default=false` zones off unless the UI explicitly enables them

## `paddock_zone_evidence.geojson`

Audit layer for qualifying evidence polygons.

Semantics:

- Features use the same zone-property shape as the display layer.
- This file is not the current app rendering source.
- Multiple evidence polygons may exist for the same dominant behavior family.

## `paddock_inference_boundary.geojson`

Buffered paddock source-record layer used for zone inference.

Semantics:

- This file is backend/audit output only.
- The current app should not render it.

## Contract Guarantees

- `manifest.version` is `4`.
- One day index exists per `available_days` entry.
- One recorder shard exists per `recorder_files[]` entry.
- Each recorder shard contains only its own `recorder_id`.
- `segment.session_id`, `trace_segments.session_id`, and `event.session_id` reference alive windows in the same recorder shard.
- `event.segment_id` and `trace_segments.segment_id` reference segments in the same recorder shard.
- Missing event coordinates are backfilled from the nearest valid trace point during dataset build.
- Trace payloads are already filtered through the checked-in GPS blocker.
- Every event has generated clip metadata and a clip file.
- `manifest.paddock_zones` is `null` immediately after `build-dataset` and populated only after `build-paddock-zones`.
- Zone artifacts must exist only when `manifest.paddock_zones` is not `null`.

## Recommended Frontend Loading Strategy

1. Load `manifest.json`.
2. Load the selected day index.
3. Clamp or snap timeline changes to `day.availability.alive_windows`.
4. Load only selected recorder shards.
5. Load `paddock_zones.geojson` only when `manifest.paddock_zones` is present.
6. Use recorder `availability.alive_windows` to filter traces, current positions, dynamic cluster polygons, hums, and detail state.
7. Keep only the selected day and selected recorder shards resident in memory where practical.

## Audio Notes

- `audio_export.path` points at the concat WAV used to build clips.
- The concat export is large and should not be sent directly to the browser.
- Use `event.clip_url_path` for browser playback.
- `export_begin_time_s` and `export_end_time_s` are retained for checking/debugging, not interactive slicing.
