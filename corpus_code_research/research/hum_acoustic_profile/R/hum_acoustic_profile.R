required_packages <- c("tuneR", "jsonlite", "digest", "knitr")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Missing required R packages: ",
    paste(missing_packages, collapse = ", "),
    ". Run: Rscript research/hum_acoustic_profile/install_deps.R",
    call. = FALSE
  )
}

hum_profile_default_config <- function() {
  bundle_dir <- file.path(
    "artifacts",
    "full_inference_runs",
    "RF_run_on_FULL_Run_1_prediction-review-raven_bundle-e0aa814d-9f9"
  )
  list(
    bundle_dir = bundle_dir,
    corpus_subdir = file.path("audio_exports", "cleaned_alpaca_hum_corpus"),
    split_subdir = file.path("audio_exports", "cleaned_alpaca_hum_corpus", "split_into_6_parts"),
    output_dir = file.path(bundle_dir, "acoustic_profile"),
    sample_size = "1200",
    min_freq_hz = 220,
    max_freq_hz = 5000,
    f0_min_hz = 220,
    f0_max_hz = 600,
    window_n = 2048,
    hop_n = 1024,
    mfcc_count = 12,
    random_seed = 20260726,
    paper = "Wierucka and Leu 2024, Robust encoding of acoustic identity in alpaca hums"
  )
}

run_hum_acoustic_profile <- function(config = hum_profile_default_config()) {
  set.seed(config$random_seed)
  dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(config$output_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

  split_dir <- file.path(config$bundle_dir, config$split_subdir)
  split_manifest_path <- file.path(split_dir, "manifest.json")
  full_selection_path <- file.path(config$bundle_dir, config$corpus_subdir, "review_selection_table.txt")

  assert_file(split_manifest_path)
  assert_file(full_selection_path)

  split_manifest <- jsonlite::fromJSON(split_manifest_path, simplifyVector = FALSE)
  parts <- split_manifest$parts
  if (length(parts) == 0) {
    stop("Split manifest contains no parts: ", split_manifest_path, call. = FALSE)
  }

  all_rows <- read_selection_table(full_selection_path)
  all_rows <- stable_order_rows(all_rows)
  corpus_rows_n <- nrow(all_rows)
  all_rows_dedup <- all_rows[!duplicated(all_rows$review_row_key), , drop = FALSE]

  selected_keys <- select_deterministic_keys(all_rows_dedup, config$sample_size)
  selected_key_set <- stats::setNames(rep(TRUE, length(selected_keys)), selected_keys)

  feature_tables <- list()
  for (part in parts) {
    part_index <- as.integer(part$part_index)
    message("Processing split part ", part_index, "...")

    selection_path <- normalizePath(part$selection_table, mustWork = TRUE)
    wav_path <- normalizePath(part$wav, mustWork = TRUE)
    rows <- read_selection_table(selection_path)
    rows <- rows[rows$review_row_key %in% names(selected_key_set), , drop = FALSE]
    rows <- rows[!duplicated(rows$review_row_key), , drop = FALSE]
    rows <- stable_order_rows(rows)

    if (nrow(rows) == 0) {
      next
    }

    wav <- tuneR::readWave(wav_path)
    feature_tables[[length(feature_tables) + 1]] <- profile_part_rows(wav, rows, part_index, config)
    rm(wav)
    gc()
  }

  if (length(feature_tables) == 0) {
    stop("No rows were selected for acoustic profiling.", call. = FALSE)
  }

  features <- do.call(rbind, feature_tables)
  features <- stable_order_rows(features)

  summary_table <- summarize_feature_table(features)
  duration_summary_table <- summarize_duration_rows(all_rows_dedup)

  features_path <- file.path(config$output_dir, "hum_acoustic_features.csv")
  summary_path <- file.path(config$output_dir, "hum_acoustic_summary.csv")
  duration_summary_path <- file.path(config$output_dir, "corpus_duration_summary.csv")
  report_path <- file.path(config$output_dir, "report.md")
  manifest_path <- file.path(config$output_dir, "manifest.json")

  write.csv(features, features_path, row.names = FALSE, na = "")
  write.csv(summary_table, summary_path, row.names = FALSE, na = "")
  write.csv(duration_summary_table, duration_summary_path, row.names = FALSE, na = "")

  plot_paths <- write_profile_plots(features, config$output_dir)
  write_report(report_path, config, split_manifest, features, summary_table, duration_summary_table, plot_paths)
  write_manifest(manifest_path, config, split_manifest_path, full_selection_path, features_path, summary_path, duration_summary_path, report_path, plot_paths, corpus_rows_n, nrow(all_rows_dedup), nrow(features))

  message("Wrote report: ", report_path)
  invisible(list(
    features = features_path,
    summary = summary_path,
    duration_summary = duration_summary_path,
    report = report_path,
    manifest = manifest_path
  ))
}

assert_file <- function(path) {
  if (!file.exists(path)) {
    stop("Required file does not exist: ", path, call. = FALSE)
  }
}

read_selection_table <- function(path) {
  out <- read.delim(
    path,
    sep = "\t",
    header = TRUE,
    check.names = FALSE,
    stringsAsFactors = FALSE,
    quote = "",
    comment.char = ""
  )
  required <- c("Selection", "Begin Time (s)", "End Time (s)", "review_row_key")
  missing <- setdiff(required, names(out))
  if (length(missing) > 0) {
    stop("Selection table missing required columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  out
}

stable_order_rows <- function(rows) {
  order_cols <- intersect(c("split_part_index", "Selection", "review_row_key"), names(rows))
  if (length(order_cols) == 0) {
    return(rows)
  }
  rows[do.call(order, rows[order_cols]), , drop = FALSE]
}

select_deterministic_keys <- function(rows, sample_size) {
  if (identical(sample_size, "all")) {
    return(rows$review_row_key)
  }
  n <- suppressWarnings(as.integer(sample_size))
  if (is.na(n) || n <= 0) {
    stop("--sample-size must be a positive integer or 'all'.", call. = FALSE)
  }
  n <- min(n, nrow(rows))
  hashes <- vapply(rows$review_row_key, digest::digest, character(1), algo = "sha256", serialize = FALSE)
  rows$deterministic_sample_hash <- hashes
  rows <- rows[order(rows$deterministic_sample_hash, rows$review_row_key), , drop = FALSE]
  rows$review_row_key[seq_len(n)]
}

profile_part_rows <- function(wav, rows, part_index, config) {
  sample_rate <- wav@samp.rate
  if (sample_rate <= 0) {
    stop("Invalid WAV sample rate in part ", part_index, call. = FALSE)
  }
  if (wav@stereo) {
    stop("Expected mono WAV but got stereo in part ", part_index, call. = FALSE)
  }

  scale <- 2^(wav@bit - 1)
  samples_int <- wav@left

  out <- vector("list", nrow(rows))
  for (i in seq_len(nrow(rows))) {
    begin_s <- as.numeric(rows[["Begin Time (s)"]][[i]])
    end_s <- as.numeric(rows[["End Time (s)"]][[i]])
    start_i <- max(1, floor(begin_s * sample_rate) + 1)
    end_i <- min(length(samples_int), ceiling(end_s * sample_rate))
    if (end_i <= start_i) {
      stop("Invalid segment boundaries for review_row_key ", rows$review_row_key[[i]], call. = FALSE)
    }

    segment <- as.numeric(samples_int[start_i:end_i]) / scale
    out[[i]] <- profile_segment(segment, sample_rate, rows[i, , drop = FALSE], part_index, config)
  }
  do.call(rbind, out)
}

profile_segment <- function(segment, sample_rate, row, part_index, config) {
  stft <- stft_power(segment, sample_rate, config$window_n, config$hop_n)
  freqs <- stft$freqs
  power <- stft$power
  band <- freqs >= config$min_freq_hz & freqs <= config$max_freq_hz
  if (!any(band)) {
    stop("No FFT bins inside configured frequency band.", call. = FALSE)
  }

  band_freqs <- freqs[band]
  band_power_by_frame <- power[band, , drop = FALSE]
  spectrum <- rowSums(band_power_by_frame)

  if (sum(spectrum) <= 0) {
    stop("Segment has zero spectral energy: ", row$review_row_key[[1]], call. = FALSE)
  }

  freq_5 <- weighted_frequency_quantile(band_freqs, spectrum, 0.05)
  freq_95 <- weighted_frequency_quantile(band_freqs, spectrum, 0.95)
  peak_freq <- band_freqs[which.max(spectrum)]
  f0 <- estimate_f0_from_harmonics(band_freqs, spectrum, config$f0_min_hz, config$f0_max_hz, config$max_freq_hz)
  entropy <- mean(apply(band_power_by_frame, 2, spectral_entropy), na.rm = TRUE)
  duration_90 <- energy_duration_90(stft$times, colSums(power))

  mfcc_wave <- tuneR::Wave(
    left = as.integer(pmax(-1, pmin(1, segment)) * 32767),
    samp.rate = sample_rate,
    bit = 16
  )
  mfcc <- tuneR::melfcc(
    mfcc_wave,
    sr = sample_rate,
    wintime = config$window_n / sample_rate,
    hoptime = config$hop_n / sample_rate,
    numcep = config$mfcc_count,
    minfreq = config$min_freq_hz,
    maxfreq = config$max_freq_hz,
    dither = FALSE,
    preemph = 0,
    frames_in_rows = TRUE
  )
  mfcc_means <- colMeans(mfcc)
  mfcc_sds <- apply(mfcc, 2, stats::sd)

  base <- data.frame(
    review_row_key = row$review_row_key[[1]],
    selection = as.integer(row$Selection[[1]]),
    split_part_index = part_index,
    begin_time_s = as.numeric(row[["Begin Time (s)"]][[1]]),
    end_time_s = as.numeric(row[["End Time (s)"]][[1]]),
    selection_duration_s = as.numeric(row[["End Time (s)"]][[1]]) - as.numeric(row[["Begin Time (s)"]][[1]]),
    duration_90_ms = duration_90 * 1000,
    f0_hz = f0,
    peak_frequency_hz = peak_freq,
    frequency_5_hz = freq_5,
    frequency_95_hz = freq_95,
    bandwidth_90_hz = freq_95 - freq_5,
    average_entropy_0_1 = entropy,
    stringsAsFactors = FALSE
  )

  metadata_cols <- intersect(
    c("wav_basename", "source_audio_file", "source_start_s", "source_end_s", "audio_collection_name", "device_id", "detection_date", "detection_time_of_day", "model_positive_score", "detection_score"),
    names(row)
  )
  metadata <- row[metadata_cols]
  names(mfcc_means) <- sprintf("mfcc_%02d_mean", seq_along(mfcc_means))
  names(mfcc_sds) <- sprintf("mfcc_%02d_sd", seq_along(mfcc_sds))

  cbind(base, metadata, as.data.frame(as.list(mfcc_means)), as.data.frame(as.list(mfcc_sds)))
}

stft_power <- function(segment, sample_rate, window_n, hop_n) {
  if (length(segment) < window_n) {
    segment <- c(segment, rep(0, window_n - length(segment)))
  }
  starts <- seq.int(1, length(segment) - window_n + 1, by = hop_n)
  last_start <- length(segment) - window_n + 1
  if (tail(starts, 1) != last_start) {
    starts <- c(starts, last_start)
  }

  window <- 0.5 - 0.5 * cos(2 * pi * (0:(window_n - 1)) / (window_n - 1))
  freq_count <- floor(window_n / 2) + 1
  power <- matrix(0, nrow = freq_count, ncol = length(starts))

  for (j in seq_along(starts)) {
    frame <- segment[starts[[j]]:(starts[[j]] + window_n - 1)] * window
    fft_values <- fft(frame)
    power[, j] <- Mod(fft_values[seq_len(freq_count)])^2
  }

  list(
    freqs = seq.int(0, sample_rate / 2, length.out = freq_count),
    times = (starts - 1 + window_n / 2) / sample_rate,
    power = power
  )
}

weighted_frequency_quantile <- function(freqs, weights, prob) {
  total <- sum(weights)
  if (total <= 0) {
    stop("Cannot compute weighted frequency quantile with zero total weight.", call. = FALSE)
  }
  cumulative <- cumsum(weights) / total
  freqs[which(cumulative >= prob)[[1]]]
}

spectral_entropy <- function(power_values) {
  total <- sum(power_values)
  if (total <= 0) {
    return(NA_real_)
  }
  p <- power_values / total
  p <- p[p > 0]
  -sum(p * log(p)) / log(length(power_values))
}

energy_duration_90 <- function(times, frame_energy) {
  total <- sum(frame_energy)
  if (total <= 0) {
    stop("Cannot compute duration 90% with zero frame energy.", call. = FALSE)
  }
  cumulative <- cumsum(frame_energy) / total
  t5 <- times[which(cumulative >= 0.05)[[1]]]
  t95 <- times[which(cumulative >= 0.95)[[1]]]
  max(0, t95 - t5)
}

estimate_f0_from_harmonics <- function(freqs, spectrum, f0_min_hz, f0_max_hz, max_freq_hz) {
  candidates <- freqs[freqs >= f0_min_hz & freqs <= f0_max_hz]
  if (length(candidates) == 0) {
    stop("No F0 candidate bins in configured frequency range.", call. = FALSE)
  }
  scores <- vapply(candidates, function(f0) {
    harmonic_count <- floor(max_freq_hz / f0)
    harmonic_count <- min(harmonic_count, 8)
    harmonics <- f0 * seq_len(harmonic_count)
    bins <- vapply(harmonics, function(h) which.min(abs(freqs - h)), integer(1))
    sum(spectrum[bins] / sqrt(seq_len(harmonic_count)))
  }, numeric(1))
  candidates[which.max(scores)]
}

summarize_feature_table <- function(features) {
  numeric_cols <- names(features)[vapply(features, is.numeric, logical(1))]
  numeric_cols <- setdiff(numeric_cols, c("selection", "split_part_index", "begin_time_s", "end_time_s"))
  rows <- lapply(numeric_cols, function(col) {
    x <- features[[col]]
    data.frame(
      feature = col,
      n = sum(!is.na(x)),
      mean = mean(x, na.rm = TRUE),
      sd = stats::sd(x, na.rm = TRUE),
      median = stats::median(x, na.rm = TRUE),
      q05 = as.numeric(stats::quantile(x, 0.05, na.rm = TRUE, names = FALSE)),
      q25 = as.numeric(stats::quantile(x, 0.25, na.rm = TRUE, names = FALSE)),
      q75 = as.numeric(stats::quantile(x, 0.75, na.rm = TRUE, names = FALSE)),
      q95 = as.numeric(stats::quantile(x, 0.95, na.rm = TRUE, names = FALSE)),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

summarize_duration_rows <- function(rows) {
  durations <- as.numeric(rows[["End Time (s)"]]) - as.numeric(rows[["Begin Time (s)"]])
  data.frame(
    corpus_unique_rows = nrow(rows),
    selection_duration_mean_ms = mean(durations) * 1000,
    selection_duration_sd_ms = stats::sd(durations) * 1000,
    selection_duration_median_ms = stats::median(durations) * 1000,
    selection_duration_q05_ms = as.numeric(stats::quantile(durations, 0.05, names = FALSE)) * 1000,
    selection_duration_q95_ms = as.numeric(stats::quantile(durations, 0.95, names = FALSE)) * 1000
  )
}

write_profile_plots <- function(features, output_dir) {
  plot_dir <- file.path(output_dir, "plots")
  paths <- c(
    duration_90 = file.path(plot_dir, "duration_90_hist.png"),
    f0 = file.path(plot_dir, "f0_hist.png"),
    peak = file.path(plot_dir, "peak_frequency_hist.png"),
    f0_peak = file.path(plot_dir, "f0_vs_peak_frequency.png")
  )

  png(paths[["duration_90"]], width = 1200, height = 800, res = 150)
  hist(features$duration_90_ms, breaks = 40, col = "#6a8d73", border = "white", main = "Duration 90%", xlab = "ms")
  dev.off()

  png(paths[["f0"]], width = 1200, height = 800, res = 150)
  hist(features$f0_hz, breaks = 40, col = "#c17c54", border = "white", main = "Estimated F0", xlab = "Hz")
  dev.off()

  png(paths[["peak"]], width = 1200, height = 800, res = 150)
  hist(features$peak_frequency_hz, breaks = 40, col = "#537188", border = "white", main = "Peak Frequency", xlab = "Hz")
  dev.off()

  png(paths[["f0_peak"]], width = 1200, height = 800, res = 150)
  plot(features$f0_hz, features$peak_frequency_hz, pch = 16, col = "#2f485860", xlab = "Estimated F0 (Hz)", ylab = "Peak frequency (Hz)", main = "F0 vs Peak Frequency")
  dev.off()

  paths
}

write_report <- function(path, config, split_manifest, features, summary_table, duration_summary_table, plot_paths) {
  core_features <- c(
    "f0_hz",
    "duration_90_ms",
    "peak_frequency_hz",
    "frequency_5_hz",
    "frequency_95_hz",
    "bandwidth_90_hz",
    "average_entropy_0_1"
  )
  core <- summary_table[summary_table$feature %in% core_features, , drop = FALSE]
  core$mean <- round(core$mean, 3)
  core$sd <- round(core$sd, 3)
  core$median <- round(core$median, 3)
  core$q05 <- round(core$q05, 3)
  core$q95 <- round(core$q95, 3)

  duration_summary <- duration_summary_table
  duration_summary[] <- lapply(duration_summary, function(x) if (is.numeric(x)) round(x, 3) else x)

  lines <- c(
    "# Alpaca Hum Acoustic Profile",
    "",
    paste0("Bundle: `", config$bundle_dir, "`"),
    "",
    "## Method",
    "",
    paste0("- Source corpus: cleaned alpaca hum export, ", split_manifest$source$selection_rows, " selection rows, ", round(split_manifest$source$duration_s, 2), " s concatenated WAV at ", split_manifest$source$sample_rate_hz, " Hz."),
    paste0("- Acoustic rows processed: ", nrow(features), " deterministic row-key sample (`--sample-size ", config$sample_size, "`)."),
    paste0("- Spectral window: ", config$window_n, " samples, hop ", config$hop_n, " samples, Hann window."),
    paste0("- Frequency band: ", config$min_freq_hz, "-", config$max_freq_hz, " Hz, matching the paper's hum range after excluding 0-220 Hz low-frequency noise."),
    "- Feature basis from Wierucka and Leu 2024: F0, duration 90%, peak frequency, frequency 5%, frequency 95%, bandwidth 90%, average entropy, and MFCC 1-12.",
    "",
    "## Typical Hum Values",
    "",
    knitr::kable(core[, c("feature", "n", "mean", "sd", "median", "q05", "q95")], format = "pipe", row.names = FALSE),
    "",
    "## Corpus Selection Durations",
    "",
    "These duration values use all unique selection rows, not only the acoustic sample.",
    "",
    knitr::kable(duration_summary, format = "pipe", row.names = FALSE),
    "",
    "## Plots",
    "",
    paste0("- Duration 90%: `", relative_path(plot_paths[["duration_90"]], dirname(path)), "`"),
    paste0("- Estimated F0: `", relative_path(plot_paths[["f0"]], dirname(path)), "`"),
    paste0("- Peak frequency: `", relative_path(plot_paths[["peak"]], dirname(path)), "`"),
    paste0("- F0 vs peak frequency: `", relative_path(plot_paths[["f0_peak"]], dirname(path)), "`"),
    "",
    "## Notes",
    "",
    "- F0 is estimated from harmonic support in the configured band; it is not the exact `soundgen` lowest-dominant-band implementation from the paper.",
    "- Frequency percentiles and duration 90% are Raven-like energy-percentile measurements computed from the audio rather than RavenPro exports.",
    "- MFCC columns are written to `hum_acoustic_features.csv`; summary rows are in `hum_acoustic_summary.csv`."
  )
  writeLines(lines, path)
}

relative_path <- function(path, start) {
  normalizePath(path, mustWork = FALSE)
  sub(paste0("^", normalizePath(start, mustWork = FALSE), "/?"), "", normalizePath(path, mustWork = FALSE))
}

write_manifest <- function(path, config, split_manifest_path, full_selection_path, features_path, summary_path, duration_summary_path, report_path, plot_paths, corpus_rows_n, corpus_unique_rows_n, acoustic_rows_n) {
  manifest <- list(
    artifact_name = "alpaca_hum_acoustic_profile",
    schema_version = 1,
    created_at_utc = format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ"),
    config = config,
    inputs = list(
      split_manifest_path = normalizePath(split_manifest_path, mustWork = TRUE),
      split_manifest_sha256 = digest::digest(file = split_manifest_path, algo = "sha256"),
      selection_table_path = normalizePath(full_selection_path, mustWork = TRUE),
      selection_table_sha256 = digest::digest(file = full_selection_path, algo = "sha256")
    ),
    counts = list(
      corpus_rows = corpus_rows_n,
      corpus_unique_review_row_keys = corpus_unique_rows_n,
      acoustic_rows_processed = acoustic_rows_n
    ),
    outputs = list(
      features_csv = normalizePath(features_path, mustWork = TRUE),
      summary_csv = normalizePath(summary_path, mustWork = TRUE),
      duration_summary_csv = normalizePath(duration_summary_path, mustWork = TRUE),
      report_md = normalizePath(report_path, mustWork = TRUE),
      plots = as.list(normalizePath(plot_paths, mustWork = TRUE))
    ),
    r = list(
      version = R.version.string,
      packages = as.list(stats::setNames(
        vapply(required_packages, function(pkg) as.character(utils::packageVersion(pkg)), character(1)),
        required_packages
      ))
    )
  )
  jsonlite::write_json(manifest, path, pretty = TRUE, auto_unbox = TRUE)
}
