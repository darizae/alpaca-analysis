# RQ2 speed sensitivity workflow.
#
# Fits only the requested NB1 interaction H2 model for each supplied GPS
# aggregation interval.

options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  out <- list(
    input = "backend/data/processed/analysis/rq2_hum_timing_group_dispersion/latest/exploratory/speed_sensitivity_attenuation_conservative_nb1_interaction_20260823/inputs/rq2_speed_sensitivity_model_input_60s.csv",
    output_dir = "backend/data/processed/analysis/rq2_hum_timing_group_dispersion/latest/exploratory/speed_sensitivity_attenuation_conservative_nb1_interaction_20260823",
    speed_intervals = "1,5,10,20",
    nsim = 1000L,
    seed = 20260823L,
    mode = "full",
    max_rows = NA_integer_
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    value <- if (i < length(args)) args[[i + 1L]] else NA_character_
    if (key == "--input") {
      out$input <- value; i <- i + 2L
    } else if (key == "--output-dir") {
      out$output_dir <- value; i <- i + 2L
    } else if (key == "--speed-intervals") {
      out$speed_intervals <- value; i <- i + 2L
    } else if (key == "--nsim") {
      out$nsim <- as.integer(value); i <- i + 2L
    } else if (key == "--seed") {
      out$seed <- as.integer(value); i <- i + 2L
    } else if (key == "--mode") {
      out$mode <- value; i <- i + 2L
    } else if (key == "--max-rows") {
      out$max_rows <- suppressWarnings(as.integer(value)); i <- i + 2L
    } else {
      stop(paste("unknown argument", key))
    }
  }
  if (!out$mode %in% c("smoke", "full")) stop(paste("unknown --mode", out$mode))
  if (!is.finite(out$nsim) || out$nsim < 1L) stop("--nsim must be a positive integer")
  out
}

cfg <- parse_args(args)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else sys.frame(1)$ofile
script_dir <- dirname(normalizePath(script_path))
repo_root <- normalizePath(file.path(script_dir, "../../../.."))
source(file.path(script_dir, "glmmtmb_random_slope_helpers.R"))

resolve_path <- function(path_value) {
  if (grepl("^/", path_value)) return(normalizePath(path_value, mustWork = FALSE))
  normalizePath(file.path(repo_root, path_value), mustWork = FALSE)
}

input_path <- resolve_path(cfg$input)
output_dir <- resolve_path(cfg$output_dir)
tables_dir <- file.path(output_dir, "tables")
figures_dir <- file.path(output_dir, "figures")
reports_dir <- file.path(output_dir, "reports")
cache_dir <- file.path(output_dir, "cache", "speed_sensitivity_nb1_interaction")
objects_dir <- file.path(cache_dir, "objects")

dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(reports_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(objects_dir, recursive = TRUE, showWarnings = FALSE)

require_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop(paste("missing required R package:", pkg))
}

for (pkg in c("glmmTMB", "DHARMa", "ggplot2")) require_package(pkg)
suppressPackageStartupMessages({
  library(glmmTMB)
  library(DHARMa)
  library(ggplot2)
})

WINDOW_LENGTH_S <- 60L
SPEED_INTERVALS <- as.integer(strsplit(cfg$speed_intervals, ",", fixed = TRUE)[[1]])
RAW_PREDICTORS <- c("area_m2", "spread_change_m2_per_min", "speed_mps", "centroid_change_in_position_m")
Z_COLUMNS <- c("z_area", "z_spread_change", "z_speed", "z_centroid_change")
REQUIRED <- c(
  "threshold_id", "speed_interval_s", "speed_metric_id", "rq2_window_id",
  "day_session_id", "session_id", "window_length_s", "window_start_seconds_local",
  "window_length_minutes", "hum_count", "primary_hum_count", RAW_PREDICTORS
)
FORMULA_INTERACTION <- paste(
  rq2_formula_for_surface("interaction")
)

safe <- function(expr) tryCatch(expr, error = function(e) e)

safe_name <- function(value) gsub("[^A-Za-z0-9_]+", "_", value)

with_warnings <- function(expr) {
  warnings_seen <- character()
  value <- withCallingHandlers(
    tryCatch(expr, error = function(e) e),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = unique(warnings_seen))
}

bind_rows_fill <- function(items) {
  items <- Filter(function(x) is.data.frame(x) && nrow(x) > 0L, items)
  if (!length(items)) return(data.frame())
  columns <- unique(unlist(lapply(items, names), use.names = FALSE))
  items <- lapply(items, function(x) {
    missing <- setdiff(columns, names(x))
    for (column in missing) x[[column]] <- NA
    x[, columns, drop = FALSE]
  })
  do.call(rbind, items)
}

fmt_num <- function(x, digits = 4) {
  ifelse(is.na(x), "", formatC(as.numeric(x), digits = digits, format = "fg", flag = "#"))
}

model_specs <- function(input) {
  rows <- list()
  for (interval_s in SPEED_INTERVALS) {
    metric_values <- unique(as.character(input$speed_metric_id[input$speed_interval_s == interval_s]))
    metric_id <- if (length(metric_values)) metric_values[[1]] else paste0("speed_", interval_s, "s")
    rows[[length(rows) + 1L]] <- data.frame(
      threshold_id = unique(as.character(input$threshold_id))[1],
      speed_interval_s = interval_s,
      speed_metric_id = metric_id,
      model_id = paste0("nb1_interaction_speed_", interval_s, "s"),
      family = "NB1",
      fixed_surface = "interaction",
      formula = FORMULA_INTERACTION,
      random_effects = rq2_random_effects_for_surface("interaction"),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

prepare_model_data <- function(input, spec) {
  data <- input[input$speed_interval_s == spec$speed_interval_s & input$window_length_s == WINDOW_LENGTH_S, , drop = FALSE]
  for (column in c("window_start_seconds_local", "window_length_minutes", "hum_count", "primary_hum_count", RAW_PREDICTORS)) {
    data[[column]] <- as.numeric(data[[column]])
  }
  data$hum_count <- as.integer(data$hum_count)
  data$primary_hum_count <- as.integer(data$primary_hum_count)
  data$session_id <- as.character(data$session_id)
  data$day_session_id <- as.character(data$day_session_id)
  data <- data[stats::complete.cases(data[, c("hum_count", "window_length_minutes", "session_id", RAW_PREDICTORS)]) &
    data$window_length_minutes > 0, , drop = FALSE]
  data <- data[order(data$session_id, data$window_start_seconds_local, data$rq2_window_id), , drop = FALSE]
  if (cfg$mode == "smoke") data <- rq2_deterministic_smoke_sample(data, cfg$max_rows)
  data$session_id <- droplevels(factor(data$session_id))
  data$model_row_index <- seq_len(nrow(data))
  data$window_number <- as.integer(ave(data$window_start_seconds_local, data$session_id, FUN = seq_along))

  standardization <- list()
  for (i in seq_along(RAW_PREDICTORS)) {
    raw <- RAW_PREDICTORS[[i]]
    z <- Z_COLUMNS[[i]]
    values <- as.numeric(data[[raw]])
    center <- mean(values, na.rm = TRUE)
    scale_value <- stats::sd(values, na.rm = TRUE)
    if (!is.finite(scale_value) || scale_value == 0) {
      data[[z]] <- 0
      note <- "constant predictor; z-score set to 0"
    } else {
      data[[z]] <- as.numeric((values - center) / scale_value)
      note <- "raw predictor mean-centered and divided by one SD"
    }
    standardization[[length(standardization) + 1L]] <- data.frame(
      threshold_id = spec$threshold_id,
      speed_interval_s = spec$speed_interval_s,
      speed_metric_id = spec$speed_metric_id,
      model_id = spec$model_id,
      predictor = z,
      raw_column = raw,
      window_length_s = WINDOW_LENGTH_S,
      transform = "none_then_z",
      center = center,
      scale = scale_value,
      raw_min = min(values, na.rm = TRUE),
      raw_max = max(values, na.rm = TRUE),
      note = note,
      stringsAsFactors = FALSE
    )
  }
  data <- rq2_add_interaction_columns(data)
  list(data = data, standardization = do.call(rbind, standardization))
}

fit_status <- function(fit) {
  if (inherits(fit, "error")) return("failed")
  if (!isTRUE(fit$sdr$pdHess)) return("bad_hessian")
  if (!identical(fit$fit$convergence, 0L)) return("optimizer_not_converged")
  if (!is.finite(as.numeric(stats::logLik(fit)))) return("nonfinite_loglik")
  "acceptable"
}

fit_one_model_uncached <- function(spec, data) {
  formula <- stats::as.formula(spec$formula)
  started <- proc.time()[[3]]
  captured <- with_warnings(
    glmmTMB::glmmTMB(
      formula = formula,
      data = data,
      family = glmmTMB::nbinom1(link = "log"),
      control = glmmTMB::glmmTMBControl(optCtrl = list(iter.max = 10000, eval.max = 10000))
    )
  )
  list(
    model_id = spec$model_id,
    family = spec$family,
    fixed_surface = spec$fixed_surface,
    formula_text = spec$formula,
    random_effects = spec$random_effects,
    random_effects_structure_version = RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION,
    fit = captured$value,
    runtime_seconds = proc.time()[[3]] - started,
    warnings = captured$warnings,
    cache_status = "computed"
  )
}

fit_one_model <- function(spec, data) {
  cache_path <- rq2_fit_cache_path(objects_dir, "rq2_speed_sensitivity", spec$model_id)
  progress_path <- file.path(cache_dir, "rq2_speed_sensitivity_model_fit_progress.csv")
  cached <- if (file.exists(cache_path)) safe(readRDS(cache_path)) else NULL
  if (!inherits(cached, "error") && rq2_cache_valid(cached, spec)) {
    cached$cache_status <- "reused"
    cached$model_cache_path <- cache_path
    message("Reusing cached RQ2 speed sensitivity model fit: ", spec$model_id)
    rq2_write_progress_row(progress_path, list(
      timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
      model_id = spec$model_id,
      step = "fit",
      status = "reused",
      cache_path = cache_path
    ))
    return(cached)
  }
  fit_result <- fit_one_model_uncached(spec, data)
  fit_result$model_cache_path <- cache_path
  rq2_atomic_save_rds(fit_result, cache_path)
  rq2_write_progress_row(progress_path, list(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    model_id = spec$model_id,
    step = "fit",
    status = fit_status(fit_result$fit),
    cache_path = cache_path
  ))
  fit_result
}

manifest_row <- function(spec, fit_result, data) {
  fit <- fit_result$fit
  status <- fit_status(fit)
  row <- data.frame(
    threshold_id = spec$threshold_id,
    speed_interval_s = spec$speed_interval_s,
    speed_metric_id = spec$speed_metric_id,
    model_id = spec$model_id,
    window_length_s = WINDOW_LENGTH_S,
    family = spec$family,
    fixed_surface = spec$fixed_surface,
    response = "attenuation_thresholded_hum_count",
    offset = "log(window_length_minutes)",
    random_effects = spec$random_effects,
    random_effects_structure_version = RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION,
    formula = spec$formula,
    n_rows = nrow(data),
    n_sessions = length(unique(data$session_id)),
    hum_count_total = sum(data$hum_count),
    primary_hum_count_total = sum(data$primary_hum_count),
    removed_hum_count_total = sum(data$primary_hum_count - data$hum_count),
    zero_hum_fraction = mean(data$hum_count == 0),
    fit_status = status,
    runtime_seconds = fit_result$runtime_seconds,
    warning_notes = paste(fit_result$warnings, collapse = " | "),
    cache_status = fit_result$cache_status,
    fit_object_path = fit_result$model_cache_path,
    model_package = "glmmTMB",
    run_context = rq2_run_context(),
    stringsAsFactors = FALSE
  )
  if (!inherits(fit, "error")) {
    row$log_likelihood <- as.numeric(stats::logLik(fit))
    row$n_parameters <- attr(stats::logLik(fit), "df")
    row$AIC <- stats::AIC(fit)
    row$BIC <- stats::BIC(fit)
    row$dispersion_parameter <- tryCatch(as.numeric(sigma(fit)), error = function(e) NA_real_)
  } else {
    row$failure_reason <- conditionMessage(fit)
  }
  row
}

fixed_effect_table <- function(spec, fit) {
  if (inherits(fit, "error")) return(data.frame())
  co <- as.data.frame(summary(fit)$coefficients$cond)
  co$term <- rownames(co)
  rownames(co) <- NULL
  std_col <- grep("^Std", names(co), value = TRUE)[[1]]
  z_col <- grep("^z value$", names(co), value = TRUE)
  p_col <- grep("^Pr", names(co), value = TRUE)[[1]]
  if (!length(z_col)) z_col <- "z value"
  estimate <- co$Estimate
  std_error <- co[[std_col]]
  data.frame(
    threshold_id = spec$threshold_id,
    speed_interval_s = spec$speed_interval_s,
    speed_metric_id = spec$speed_metric_id,
    model_id = spec$model_id,
    family = spec$family,
    fixed_surface = spec$fixed_surface,
    term = co$term,
    estimate = estimate,
    std_error = std_error,
    z_value = co[[z_col]],
    p_value = co[[p_col]],
    incidence_rate_ratio = exp(estimate),
    irr_95_low = exp(estimate - stats::qnorm(0.975) * std_error),
    irr_95_high = exp(estimate + stats::qnorm(0.975) * std_error),
    p_value_method = "glmmTMB conditional-model coefficient Wald z test",
    stringsAsFactors = FALSE
  )
}

extract_htest <- function(test, spec, test_name, n_obs, n_sim) {
  statistic <- if (!is.null(test$statistic) && length(test$statistic)) unname(test$statistic[[1]]) else NA_real_
  statistic_name <- if (!is.null(test$statistic) && length(test$statistic)) names(test$statistic)[[1]] else ""
  estimate <- if (!is.null(test$estimate) && length(test$estimate)) unname(test$estimate[[1]]) else NA_real_
  data.frame(
    threshold_id = spec$threshold_id,
    speed_interval_s = spec$speed_interval_s,
    speed_metric_id = spec$speed_metric_id,
    model_id = spec$model_id,
    family = spec$family,
    fixed_surface = spec$fixed_surface,
    test = test_name,
    method = if (!is.null(test$method)) test$method else "",
    statistic_name = statistic_name,
    statistic = statistic,
    estimate = estimate,
    alternative = if (!is.null(test$alternative)) test$alternative else "",
    p_value = if (!is.null(test$p.value)) test$p.value else NA_real_,
    n_obs = n_obs,
    n_sim = n_sim,
    status = "computed",
    stringsAsFactors = FALSE
  )
}

test_row_or_error <- function(expr, spec, test_name, n_obs, n_sim) {
  result <- safe(expr)
  if (inherits(result, "error")) {
    return(data.frame(
      threshold_id = spec$threshold_id,
      speed_interval_s = spec$speed_interval_s,
      speed_metric_id = spec$speed_metric_id,
      model_id = spec$model_id,
      family = spec$family,
      fixed_surface = spec$fixed_surface,
      test = test_name,
      method = paste("ERROR:", conditionMessage(result)),
      statistic_name = "",
      statistic = NA_real_,
      estimate = NA_real_,
      alternative = "",
      p_value = NA_real_,
      n_obs = n_obs,
      n_sim = n_sim,
      status = "failed",
      stringsAsFactors = FALSE
    ))
  }
  extract_htest(result, spec, test_name, n_obs, n_sim)
}

dharma_outputs <- function(spec, fit, data, seed) {
  if (inherits(fit, "error")) {
    return(list(standard_tests = data.frame(), temporal_tests = data.frame(), residuals = data.frame(), acf = data.frame(), figures = data.frame()))
  }
  suffix <- paste0("speed_", spec$speed_interval_s, "s")
  cache_path <- rq2_dharma_cache_path(objects_dir, "rq2_speed_sensitivity", spec$model_id)
  cached <- if (file.exists(cache_path)) safe(readRDS(cache_path)) else NULL
  if (!inherits(cached, "error") && rq2_cache_valid(cached, spec, nsim = cfg$nsim, seed = seed)) {
    figure_paths <- if ("figures" %in% names(cached) && nrow(cached$figures) && "path" %in% names(cached$figures)) cached$figures$path else character()
    if (!length(figure_paths) || all(file.exists(figure_paths))) {
      message("Reusing cached RQ2 speed sensitivity DHARMa outputs: ", spec$model_id)
      rq2_write_progress_row(file.path(cache_dir, "rq2_speed_sensitivity_model_fit_progress.csv"), list(
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
        model_id = spec$model_id,
        step = "dharma",
        status = "reused",
        cache_path = cache_path
      ))
      return(cached[c("standard_tests", "temporal_tests", "residuals", "acf", "figures")])
    }
  }
  dharma <- DHARMa::simulateResiduals(
    fittedModel = fit,
    n = cfg$nsim,
    refit = FALSE,
    plot = FALSE,
    seed = seed
  )
  object_path <- file.path(objects_dir, paste0("rq2_speed_sensitivity_dharma_raw_", RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION, "_", suffix, ".rds"))
  rq2_atomic_save_rds(dharma, object_path)

  dharma_plot_path <- file.path(figures_dir, paste0("rq2_speed_sensitivity_dharma_", suffix, ".png"))
  grDevices::png(dharma_plot_path, width = 1800, height = 1400, res = 180)
  plot(dharma)
  grDevices::dev.off()

  standard_tests <- rbind(
    test_row_or_error(DHARMa::testUniformity(dharma, plot = FALSE), spec, "uniformity", dharma$nObs, cfg$nsim),
    test_row_or_error(DHARMa::testDispersion(dharma, alternative = "two.sided", plot = FALSE, type = "DHARMa"), spec, "dispersion", dharma$nObs, cfg$nsim),
    test_row_or_error(DHARMa::testZeroInflation(dharma, plot = FALSE), spec, "zero_inflation", dharma$nObs, cfg$nsim),
    test_row_or_error(DHARMa::testOutliers(dharma, plot = FALSE, type = "bootstrap", nBoot = min(100L, cfg$nsim)), spec, "outliers_bootstrap", dharma$nObs, cfg$nsim)
  )

  residuals <- data.frame(
    threshold_id = spec$threshold_id,
    speed_interval_s = spec$speed_interval_s,
    speed_metric_id = spec$speed_metric_id,
    model_id = spec$model_id,
    family = spec$family,
    fixed_surface = spec$fixed_surface,
    rq2_window_id = data$rq2_window_id,
    day_session_id = data$day_session_id,
    session_id = as.character(data$session_id),
    window_start_seconds_local = data$window_start_seconds_local,
    window_number = data$window_number,
    observed_hum_count = data$hum_count,
    fitted_value = as.numeric(stats::predict(fit, type = "response")),
    scaled_residual = as.numeric(dharma$scaledResiduals),
    stringsAsFactors = FALSE
  )

  temporal_rows <- list()
  acf_rows <- list()
  for (session in sort(unique(residuals$session_id))) {
    idx <- residuals$session_id == session
    session_time <- residuals$window_start_seconds_local[idx]
    if (sum(idx) < 3L || length(unique(session_time)) < 3L) {
      temporal_rows[[length(temporal_rows) + 1L]] <- data.frame(
        threshold_id = spec$threshold_id,
        speed_interval_s = spec$speed_interval_s,
        speed_metric_id = spec$speed_metric_id,
        model_id = spec$model_id,
        family = spec$family,
        fixed_surface = spec$fixed_surface,
        session_id = session,
        n_windows = sum(idx),
        test = "temporal_autocorrelation",
        method = "not run: fewer than three ordered windows",
        statistic_name = "",
        statistic = NA_real_,
        estimate = NA_real_,
        alternative = "",
        p_value = NA_real_,
        status = "skipped",
        stringsAsFactors = FALSE
      )
      next
    }
    recalc <- DHARMa::recalculateResiduals(dharma, sel = idx)
    temporal_rows[[length(temporal_rows) + 1L]] <- cbind(
      session_id = session,
      n_windows = sum(idx),
      test_row_or_error(
        DHARMa::testTemporalAutocorrelation(recalc, time = session_time, plot = FALSE),
        spec,
        "temporal_autocorrelation",
        sum(idx),
        cfg$nsim
      )
    )
    acf_obj <- stats::acf(residuals$scaled_residual[idx] - 0.5, lag.max = min(30L, sum(idx) - 1L), plot = FALSE, na.action = stats::na.pass)
    acf_rows[[length(acf_rows) + 1L]] <- data.frame(
      threshold_id = spec$threshold_id,
      speed_interval_s = spec$speed_interval_s,
      speed_metric_id = spec$speed_metric_id,
      model_id = spec$model_id,
      family = spec$family,
      fixed_surface = spec$fixed_surface,
      session_id = session,
      lag = as.integer(acf_obj$lag[, 1, 1]),
      acf = as.numeric(acf_obj$acf[, 1, 1]),
      stringsAsFactors = FALSE
    )
  }
  temporal_tests <- bind_rows_fill(temporal_rows)
  acf_table <- bind_rows_fill(acf_rows)

  residual_plot_path <- file.path(figures_dir, paste0("rq2_speed_sensitivity_residuals_vs_window_", suffix, ".png"))
  residual_plot <- ggplot2::ggplot(residuals, ggplot2::aes(x = window_number, y = scaled_residual)) +
    ggplot2::geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.3, color = "grey45") +
    ggplot2::geom_point(alpha = 0.45, size = 0.45, color = "#17202a") +
    ggplot2::coord_cartesian(ylim = c(0, 1)) +
    ggplot2::facet_wrap(~day_session_id, ncol = 2, scales = "free_x") +
    ggplot2::labs(
      title = paste0("RQ2 speed sensitivity ", spec$speed_interval_s, " s: DHARMa residuals"),
      x = "Window number within session",
      y = "DHARMa scaled residual"
    ) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(), strip.text = ggplot2::element_text(size = 7, face = "bold"))
  ggplot2::ggsave(residual_plot_path, residual_plot, width = 13, height = 12, dpi = 180, bg = "white")

  acf_plot_path <- file.path(figures_dir, paste0("rq2_speed_sensitivity_residual_acf_", suffix, ".png"))
  acf_plot <- ggplot2::ggplot(acf_table[acf_table$lag > 0, , drop = FALSE], ggplot2::aes(x = lag, y = acf)) +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.3, color = "grey45") +
    ggplot2::geom_col(width = 0.85, fill = "#3f5f5b") +
    ggplot2::facet_wrap(~session_id, ncol = 2) +
    ggplot2::labs(
      title = paste0("RQ2 speed sensitivity ", spec$speed_interval_s, " s: residual autocorrelation"),
      x = "Lag in 60 s windows",
      y = "Autocorrelation"
    ) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(), strip.text = ggplot2::element_text(size = 7, face = "bold"))
  ggplot2::ggsave(acf_plot_path, acf_plot, width = 13, height = 12, dpi = 180, bg = "white")

  figures <- data.frame(
    threshold_id = spec$threshold_id,
    speed_interval_s = spec$speed_interval_s,
    speed_metric_id = spec$speed_metric_id,
    model_id = spec$model_id,
    figure_type = c("DHARMa standard residual plot", "DHARMa residuals vs within-session window number", "Within-session residual ACF"),
    path = c(dharma_plot_path, residual_plot_path, acf_plot_path),
    stringsAsFactors = FALSE
  )
  outputs <- list(
    model_id = spec$model_id,
    family = spec$family,
    fixed_surface = spec$fixed_surface,
    formula_text = spec$formula,
    random_effects = spec$random_effects,
    random_effects_structure_version = RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION,
    nsim = cfg$nsim,
    seed = seed,
    standard_tests = standard_tests,
    temporal_tests = temporal_tests,
    residuals = residuals,
    acf = acf_table,
    figures = figures
  )
  rq2_atomic_save_rds(outputs, cache_path)
  rq2_write_progress_row(file.path(cache_dir, "rq2_speed_sensitivity_model_fit_progress.csv"), list(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    model_id = spec$model_id,
    step = "dharma",
    status = "computed",
    cache_path = cache_path
  ))
  outputs[c("standard_tests", "temporal_tests", "residuals", "acf", "figures")]
}

write_fixed_effect_plot <- function(fixed_effects) {
  plot_data <- fixed_effects[fixed_effects$term != "(Intercept)", , drop = FALSE]
  if (!nrow(plot_data)) return(data.frame())
  term_order <- c("z_area", "z_spread_change", "z_speed", "z_centroid_change", "z_area:z_spread_change")
  plot_data$term_label <- factor(plot_data$term, levels = rev(term_order))
  plot_data$interval_label <- factor(paste0(plot_data$speed_interval_s, " s"), levels = paste0(SPEED_INTERVALS, " s"))
  path <- file.path(figures_dir, "rq2_speed_sensitivity_fixed_effects_incidence_rate_ratios.png")
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = incidence_rate_ratio, y = term_label, color = interval_label)) +
    ggplot2::geom_vline(xintercept = 1, linetype = "dashed", linewidth = 0.3, color = "grey45") +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = irr_95_low, xmax = irr_95_high), orientation = "y", width = 0.18, position = ggplot2::position_dodge(width = 0.5)) +
    ggplot2::geom_point(position = ggplot2::position_dodge(width = 0.5), size = 1.8) +
    ggplot2::scale_x_log10() +
    ggplot2::labs(
      title = "RQ2 speed sensitivity fixed effects",
      subtitle = "NB1 interaction model; incidence-rate ratios per one SD increase.",
      x = "Incidence-rate ratio, log scale",
      y = NULL,
      color = "GPS interval"
    ) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  ggplot2::ggsave(path, p, width = 10, height = 5.5, dpi = 180, bg = "white")
  data.frame(model_id = "all_speed_intervals", figure_type = "Fixed-effect incidence-rate ratio plot", path = path, stringsAsFactors = FALSE)
}

write_speed_effect_plot <- function(fixed_effects) {
  speed <- fixed_effects[fixed_effects$term == "z_speed", , drop = FALSE]
  if (!nrow(speed)) return(data.frame())
  path <- file.path(figures_dir, "rq2_speed_sensitivity_speed_irr_by_interval.png")
  p <- ggplot2::ggplot(speed, ggplot2::aes(x = speed_interval_s, y = incidence_rate_ratio)) +
    ggplot2::geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.3, color = "grey45") +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = irr_95_low, ymax = irr_95_high), width = 0.6, color = "#2f5d62") +
    ggplot2::geom_line(color = "#2f5d62") +
    ggplot2::geom_point(size = 2.2, color = "#2f5d62") +
    ggplot2::scale_x_continuous(breaks = SPEED_INTERVALS) +
    ggplot2::labs(
      title = "Mean group member speed effect by GPS aggregation interval",
      x = "GPS aggregation interval (s)",
      y = "Speed incidence-rate ratio"
    ) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  ggplot2::ggsave(path, p, width = 7, height = 4.5, dpi = 180, bg = "white")
  data.frame(model_id = "speed_effect", figure_type = "Speed-effect IRR by interval", path = path, stringsAsFactors = FALSE)
}

environment_table <- function() {
  packages <- c("glmmTMB", "DHARMa", "ggplot2")
  data.frame(
    run_context = rq2_run_context(),
    machine_description = rq2_machine_description(),
    R_version = R.version.string,
    platform = R.version$platform,
    package = packages,
    package_version = vapply(packages, function(pkg) as.character(utils::packageVersion(pkg)), character(1)),
    nsim_dharma = cfg$nsim,
    seed = cfg$seed,
    mode = cfg$mode,
    random_effects_structure_version = RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION,
    stringsAsFactors = FALSE
  )
}

if (!file.exists(input_path)) stop(paste("missing model input:", input_path))
input <- utils::read.csv(input_path, check.names = FALSE)
missing <- setdiff(REQUIRED, names(input))
if (length(missing)) stop(paste("model input missing required columns:", paste(missing, collapse = ", ")))

input$speed_interval_s <- as.integer(input$speed_interval_s)
specs <- model_specs(input)
specs$random_effects_structure_version <- RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION
utils::write.csv(specs, file.path(tables_dir, "rq2_speed_sensitivity_model_formulas.csv"), row.names = FALSE, na = "")

manifest_rows <- list()
fixed_rows <- list()
standardization_rows <- list()
random_slope_rows <- list()
dharma_standard_rows <- list()
dharma_temporal_rows <- list()
residual_rows <- list()
acf_rows <- list()
figure_rows <- list()

for (i in seq_len(nrow(specs))) {
  spec <- specs[i, , drop = FALSE]
  prepared <- prepare_model_data(input, spec)
  data <- prepared$data
  if (nrow(data) == 0L) stop(paste("no complete model rows for speed interval", spec$speed_interval_s))
  if (length(unique(data$session_id)) < 2L) stop(paste("fewer than two session_id levels for speed interval", spec$speed_interval_s))
  message("Fitting RQ2 speed sensitivity model: ", spec$model_id)
  fit_result <- fit_one_model(spec, data)
  fit <- fit_result$fit

  manifest_rows[[length(manifest_rows) + 1L]] <- manifest_row(spec, fit_result, data)
  fixed_rows[[length(fixed_rows) + 1L]] <- fixed_effect_table(spec, fit)
  standardization_rows[[length(standardization_rows) + 1L]] <- prepared$standardization
  random_slope_rows[[length(random_slope_rows) + 1L]] <- cbind(
    threshold_id = spec$threshold_id,
    speed_interval_s = spec$speed_interval_s,
    speed_metric_id = spec$speed_metric_id,
    rq2_random_slope_decisions(data, spec)
  )
  dharma <- dharma_outputs(spec, fit, data, cfg$seed + i)
  dharma_standard_rows[[length(dharma_standard_rows) + 1L]] <- dharma$standard_tests
  dharma_temporal_rows[[length(dharma_temporal_rows) + 1L]] <- dharma$temporal_tests
  residual_rows[[length(residual_rows) + 1L]] <- dharma$residuals
  acf_rows[[length(acf_rows) + 1L]] <- dharma$acf
  figure_rows[[length(figure_rows) + 1L]] <- dharma$figures
}

run_manifest <- bind_rows_fill(manifest_rows)
fixed_effects <- bind_rows_fill(fixed_rows)
standardization <- bind_rows_fill(standardization_rows)
random_slope_decisions <- bind_rows_fill(random_slope_rows)
dharma_tests <- bind_rows_fill(dharma_standard_rows)
temporal_tests <- bind_rows_fill(dharma_temporal_rows)
scaled_residuals <- bind_rows_fill(residual_rows)
acf_table <- bind_rows_fill(acf_rows)
figure_index <- bind_rows_fill(c(figure_rows, list(write_fixed_effect_plot(fixed_effects), write_speed_effect_plot(fixed_effects))))
environment <- environment_table()

utils::write.csv(run_manifest, file.path(tables_dir, "rq2_speed_sensitivity_model_run_manifest.csv"), row.names = FALSE, na = "")
utils::write.csv(fixed_effects, file.path(tables_dir, "rq2_speed_sensitivity_fixed_effects.csv"), row.names = FALSE, na = "")
utils::write.csv(standardization, file.path(tables_dir, "rq2_speed_sensitivity_predictor_standardization.csv"), row.names = FALSE, na = "")
utils::write.csv(random_slope_decisions, file.path(tables_dir, "rq2_speed_sensitivity_random_slope_decisions.csv"), row.names = FALSE, na = "")
utils::write.csv(dharma_tests, file.path(tables_dir, "rq2_speed_sensitivity_dharma_standard_tests.csv"), row.names = FALSE, na = "")
utils::write.csv(temporal_tests, file.path(tables_dir, "rq2_speed_sensitivity_dharma_temporal_autocorrelation.csv"), row.names = FALSE, na = "")
utils::write.csv(scaled_residuals, gzfile(file.path(tables_dir, "rq2_speed_sensitivity_scaled_residuals.csv.gz")), row.names = FALSE, na = "")
utils::write.csv(acf_table, file.path(tables_dir, "rq2_speed_sensitivity_residual_acf.csv"), row.names = FALSE, na = "")
utils::write.csv(figure_index, file.path(tables_dir, "rq2_speed_sensitivity_figure_index.csv"), row.names = FALSE, na = "")
utils::write.csv(environment, file.path(tables_dir, "rq2_speed_sensitivity_environment.csv"), row.names = FALSE, na = "")
writeLines(capture.output(utils::sessionInfo()), con = file.path(reports_dir, "rq2_speed_sensitivity_session_info.txt"))

message("Wrote RQ2 speed sensitivity outputs to ", output_dir)
