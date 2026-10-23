# Shared RQ2 amplitude-threshold GLMM workflow.
#
# Fits 60 s attenuation-calibrated thresholded responses only:
#   - fixed dBFS thresholds from the attenuation experiment
#   - Poisson and NB1 families
#   - interaction and additive fixed surfaces
#   - DHARMa diagnostics for every fit

options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  out <- list(
    input = "",
    output_dir = "",
    nsim = 1000L,
    seed = 20260804L,
    thresholds = "attenuation_lenient,attenuation_primary,attenuation_conservative",
    mode = "full",
    max_rows = NA_integer_,
    output_prefix = "rq2_attenuation",
    model_id_tag = "attenuation_amplitude",
    response_label = "attenuation_thresholded_hum_count",
    run_label = "attenuation-threshold",
    plot_title_prefix = "Hum-rate fixed effects after attenuation-calibrated amplitude filtering"
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    value <- if (i < length(args)) args[[i + 1L]] else NA_character_
    if (key == "--input") {
      out$input <- value; i <- i + 2L
    } else if (key == "--output-dir") {
      out$output_dir <- value; i <- i + 2L
    } else if (key == "--nsim") {
      out$nsim <- as.integer(value); i <- i + 2L
    } else if (key == "--seed") {
      out$seed <- as.integer(value); i <- i + 2L
    } else if (key == "--thresholds") {
      out$thresholds <- value; i <- i + 2L
    } else if (key == "--mode") {
      out$mode <- value; i <- i + 2L
    } else if (key == "--max-rows") {
      out$max_rows <- suppressWarnings(as.integer(value)); i <- i + 2L
    } else if (key == "--output-prefix") {
      out$output_prefix <- value; i <- i + 2L
    } else if (key == "--model-id-tag") {
      out$model_id_tag <- value; i <- i + 2L
    } else if (key == "--response-label") {
      out$response_label <- value; i <- i + 2L
    } else if (key == "--run-label") {
      out$run_label <- value; i <- i + 2L
    } else if (key == "--plot-title-prefix") {
      out$plot_title_prefix <- value; i <- i + 2L
    } else {
      stop(paste("unknown argument", key))
    }
  }
  if (!nzchar(out$input)) stop("--input is required")
  if (!nzchar(out$output_dir)) stop("--output-dir is required")
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
objects_dir <- file.path(output_dir, "cache", "objects")

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
THRESHOLDS <- if (nzchar(cfg$thresholds)) {
  strsplit(cfg$thresholds, ",", fixed = TRUE)[[1]]
} else {
  c("attenuation_lenient", "attenuation_primary", "attenuation_conservative")
}
OUTPUT_PREFIX <- cfg$output_prefix
MODEL_ID_TAG <- cfg$model_id_tag
RESPONSE_LABEL <- cfg$response_label
RUN_LABEL <- cfg$run_label
PLOT_TITLE_PREFIX <- cfg$plot_title_prefix
FAMILIES <- c("Poisson", "NB1")
SURFACES <- c("interaction", "additive")
RAW_PREDICTORS <- c("area_m2", "spread_change_m2_per_min", "speed_mps", "centroid_change_in_position_m")
Z_COLUMNS <- c("z_area", "z_spread_change", "z_speed", "z_centroid_change")
REQUIRED <- c(
  "threshold_id", "rq2_window_id", "day_session_id", "session_id",
  "window_length_s", "window_start_seconds_local", "window_length_minutes",
  "hum_count", "primary_hum_count", RAW_PREDICTORS
)
FORMULA_INTERACTION <- paste(
  rq2_formula_for_surface("interaction")
)
FORMULA_ADDITIVE <- paste(
  rq2_formula_for_surface("additive")
)

safe <- function(expr) tryCatch(expr, error = function(e) e)

safe_name <- function(value) gsub("[^A-Za-z0-9_]+", "_", value)

PREDICTOR_DISPLAY_LABELS <- c(
  z_area = "Current spread",
  z_spread_change = "Spread change",
  z_speed = "Mean group member speed",
  z_centroid_change = "Centroid displacement",
  "z_area:z_spread_change" = "Current spread \u00d7 spread change"
)

THRESHOLD_DISPLAY_LABELS <- c(
  attenuation_lenient = "External-fence Q90",
  attenuation_primary = "External-fence Q95",
  attenuation_conservative = "External-fence Q97.5"
)

display_label <- function(values, labels) {
  out <- as.character(values)
  matched <- out %in% names(labels)
  out[matched] <- unname(labels[out[matched]])
  out
}

model_display_label <- function(family, fixed_surface) {
  out <- paste(family, fixed_surface, sep = " / ")
  out[family == "Poisson" & fixed_surface == "interaction"] <- "Poisson, with interaction"
  out[family == "Poisson" & fixed_surface == "additive"] <- "Poisson, without interaction"
  out[family == "NB1" & fixed_surface == "interaction"] <- "Negative binomial, with interaction"
  out[family == "NB1" & fixed_surface == "additive"] <- "Negative binomial, without interaction"
  out
}

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

model_specs <- function() {
  rows <- list()
  for (threshold_id in THRESHOLDS) {
    for (family in FAMILIES) {
      for (surface in SURFACES) {
        formula_text <- if (surface == "interaction") FORMULA_INTERACTION else FORMULA_ADDITIVE
        family_id <- if (family == "Poisson") "poisson" else "nb1"
        rows[[length(rows) + 1L]] <- data.frame(
          threshold_id = threshold_id,
          family = family,
          fixed_surface = surface,
          model_id = paste(family_id, surface, MODEL_ID_TAG, threshold_id, sep = "_"),
          formula = formula_text,
          random_effects = rq2_random_effects_for_surface(surface),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  do.call(rbind, rows)
}

family_for_label <- function(label) {
  if (label == "Poisson") return(stats::poisson(link = "log"))
  if (label == "NB1") return(glmmTMB::nbinom1(link = "log"))
  stop(paste("unsupported family label:", label))
}

prepare_model_data <- function(input, spec) {
  data <- input[input$threshold_id == spec$threshold_id & input$window_length_s == WINDOW_LENGTH_S, , drop = FALSE]
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
      model_id = spec$model_id,
      family = spec$family,
      fixed_surface = spec$fixed_surface,
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

fit_one_uncached <- function(spec, data) {
  started <- proc.time()[[3]]
  captured <- with_warnings(
    glmmTMB::glmmTMB(
      formula = stats::as.formula(spec$formula),
      data = data,
      family = family_for_label(spec$family),
      control = glmmTMB::glmmTMBControl(optCtrl = list(iter.max = 10000, eval.max = 10000))
    )
  )
  list(
    threshold_id = spec$threshold_id,
    model_id = spec$model_id,
    family = spec$family,
    fixed_surface = spec$fixed_surface,
    formula = spec$formula,
    formula_text = spec$formula,
    random_effects = spec$random_effects,
    random_effects_structure_version = RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION,
    fit = captured$value,
    runtime_seconds = proc.time()[[3]] - started,
    warnings = captured$warnings,
    cache_status = "computed"
  )
}

fit_one <- function(spec, data) {
  cache_path <- rq2_fit_cache_path(objects_dir, OUTPUT_PREFIX, spec$model_id)
  progress_path <- file.path(output_dir, "cache", paste0(OUTPUT_PREFIX, "_model_fit_progress.csv"))
  cached <- if (file.exists(cache_path)) safe(readRDS(cache_path)) else NULL
  if (!inherits(cached, "error") && rq2_cache_valid(cached, spec)) {
    cached$cache_status <- "reused"
    cached$model_cache_path <- cache_path
    message("Reusing cached ", RUN_LABEL, " RQ2 model fit: ", spec$model_id)
    rq2_write_progress_row(progress_path, list(
      timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
      model_id = spec$model_id,
      step = "fit",
      status = "reused",
      cache_path = cache_path
    ))
    return(cached)
  }
  fit_result <- fit_one_uncached(spec, data)
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

manifest_row <- function(fit_result, data) {
  fit <- fit_result$fit
  status <- fit_status(fit)
  row <- data.frame(
    threshold_id = fit_result$threshold_id,
    model_id = fit_result$model_id,
    window_length_s = WINDOW_LENGTH_S,
    family = fit_result$family,
    fixed_surface = fit_result$fixed_surface,
    response = RESPONSE_LABEL,
    offset = "log(window_length_minutes)",
    random_effects = fit_result$random_effects,
    random_effects_structure_version = RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION,
    formula = fit_result$formula,
    n_rows = nrow(data),
    n_sessions = length(unique(data$session_id)),
    hum_count_total = sum(data$hum_count),
    primary_hum_count_total = sum(data$primary_hum_count),
    removed_hum_count_total = sum(data$primary_hum_count) - sum(data$hum_count),
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

fixed_effect_table <- function(fit_result) {
  fit <- fit_result$fit
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
    threshold_id = fit_result$threshold_id,
    model_id = fit_result$model_id,
    family = fit_result$family,
    fixed_surface = fit_result$fixed_surface,
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

extract_htest <- function(test, fit_result, test_name, n_obs, n_sim) {
  statistic <- if (!is.null(test$statistic) && length(test$statistic)) unname(test$statistic[[1]]) else NA_real_
  statistic_name <- if (!is.null(test$statistic) && length(test$statistic)) names(test$statistic)[[1]] else ""
  estimate <- if (!is.null(test$estimate) && length(test$estimate)) unname(test$estimate[[1]]) else NA_real_
  data.frame(
    threshold_id = fit_result$threshold_id,
    model_id = fit_result$model_id,
    family = fit_result$family,
    fixed_surface = fit_result$fixed_surface,
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

test_row_or_error <- function(expr, fit_result, test_name, n_obs, n_sim) {
  result <- safe(expr)
  if (inherits(result, "error")) {
    return(data.frame(
      threshold_id = fit_result$threshold_id,
      model_id = fit_result$model_id,
      family = fit_result$family,
      fixed_surface = fit_result$fixed_surface,
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
  extract_htest(result, fit_result, test_name, n_obs, n_sim)
}

dharma_outputs <- function(fit_result, data, seed) {
  fit <- fit_result$fit
  if (inherits(fit, "error")) {
    return(list(standard_tests = data.frame(), temporal_tests = data.frame(), residuals = data.frame(), acf = data.frame(), figures = data.frame()))
  }
  cache_path <- rq2_dharma_cache_path(objects_dir, OUTPUT_PREFIX, fit_result$model_id)
  cached <- if (file.exists(cache_path)) safe(readRDS(cache_path)) else NULL
  if (!inherits(cached, "error") && rq2_cache_valid(cached, fit_result, nsim = cfg$nsim, seed = seed)) {
    figure_paths <- if ("figures" %in% names(cached) && nrow(cached$figures) && "path" %in% names(cached$figures)) cached$figures$path else character()
    if (!length(figure_paths) || all(file.exists(figure_paths))) {
      message("Reusing cached ", RUN_LABEL, " RQ2 DHARMa outputs: ", fit_result$model_id)
      rq2_write_progress_row(file.path(output_dir, "cache", paste0(OUTPUT_PREFIX, "_model_fit_progress.csv")), list(
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
        model_id = fit_result$model_id,
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
  object_path <- file.path(objects_dir, paste0(OUTPUT_PREFIX, "_dharma_raw_", RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION, "_", safe_name(fit_result$model_id), ".rds"))
  rq2_atomic_save_rds(dharma, object_path)

  dharma_plot_path <- file.path(figures_dir, paste0(OUTPUT_PREFIX, "_dharma_", safe_name(fit_result$model_id), ".png"))
  grDevices::png(dharma_plot_path, width = 1800, height = 1400, res = 180)
  plot(dharma)
  grDevices::dev.off()

  standard_tests <- rbind(
    test_row_or_error(DHARMa::testUniformity(dharma, plot = FALSE), fit_result, "uniformity", dharma$nObs, cfg$nsim),
    test_row_or_error(DHARMa::testDispersion(dharma, alternative = "two.sided", plot = FALSE, type = "DHARMa"), fit_result, "dispersion", dharma$nObs, cfg$nsim),
    test_row_or_error(DHARMa::testZeroInflation(dharma, plot = FALSE), fit_result, "zero_inflation", dharma$nObs, cfg$nsim),
    test_row_or_error(DHARMa::testOutliers(dharma, plot = FALSE, type = "bootstrap", nBoot = min(100L, cfg$nsim)), fit_result, "outliers_bootstrap", dharma$nObs, cfg$nsim)
  )

  residuals <- data.frame(
    threshold_id = fit_result$threshold_id,
    model_id = fit_result$model_id,
    family = fit_result$family,
    fixed_surface = fit_result$fixed_surface,
    rq2_window_id = data$rq2_window_id,
    day_session_id = data$day_session_id,
    session_id = as.character(data$session_id),
    window_start_seconds_local = data$window_start_seconds_local,
    window_number = data$window_number,
    observed_hum_count = data$hum_count,
    primary_hum_count = data$primary_hum_count,
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
        threshold_id = fit_result$threshold_id,
        model_id = fit_result$model_id,
        family = fit_result$family,
        fixed_surface = fit_result$fixed_surface,
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
        fit_result,
        "temporal_autocorrelation",
        sum(idx),
        cfg$nsim
      )
    )
    acf_obj <- stats::acf(residuals$scaled_residual[idx] - 0.5, lag.max = min(30L, sum(idx) - 1L), plot = FALSE, na.action = stats::na.pass)
    acf_rows[[length(acf_rows) + 1L]] <- data.frame(
      threshold_id = fit_result$threshold_id,
      model_id = fit_result$model_id,
      family = fit_result$family,
      fixed_surface = fit_result$fixed_surface,
      session_id = session,
      lag = as.integer(acf_obj$lag[, 1, 1]),
      acf = as.numeric(acf_obj$acf[, 1, 1]),
      stringsAsFactors = FALSE
    )
  }
  temporal_tests <- bind_rows_fill(temporal_rows)
  acf_table <- bind_rows_fill(acf_rows)

  residual_plot_path <- file.path(figures_dir, paste0(OUTPUT_PREFIX, "_residuals_vs_window_", safe_name(fit_result$model_id), ".png"))
  residual_plot <- ggplot2::ggplot(residuals, ggplot2::aes(x = window_number, y = scaled_residual)) +
    ggplot2::geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.3, color = "grey45") +
    ggplot2::geom_point(alpha = 0.45, size = 0.45, color = "#17202a") +
    ggplot2::coord_cartesian(ylim = c(0, 1)) +
    ggplot2::facet_wrap(~day_session_id, ncol = 2, scales = "free_x") +
    ggplot2::labs(
      title = paste0(PLOT_TITLE_PREFIX, " ", fit_result$model_id, ": DHARMa residuals vs window number"),
      subtitle = "Panels are sessions; dashed line marks the expected scaled-residual median of 0.5.",
      x = "Window number within session",
      y = "DHARMa scaled residual"
    ) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(), strip.text = ggplot2::element_text(size = 7, face = "bold"))
  ggplot2::ggsave(residual_plot_path, residual_plot, width = 13, height = 12, dpi = 180, bg = "white")

  acf_plot_path <- file.path(figures_dir, paste0(OUTPUT_PREFIX, "_residual_acf_", safe_name(fit_result$model_id), ".png"))
  acf_plot <- ggplot2::ggplot(acf_table[acf_table$lag > 0, , drop = FALSE], ggplot2::aes(x = lag, y = acf)) +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.3, color = "grey45") +
    ggplot2::geom_col(width = 0.85, fill = "#3f5f5b") +
    ggplot2::facet_wrap(~session_id, ncol = 2) +
    ggplot2::labs(
      title = paste0(PLOT_TITLE_PREFIX, " ", fit_result$model_id, ": residual autocorrelation by session"),
      subtitle = "ACF of DHARMa scaled residuals centered at 0.5.",
      x = "Lag in 60 s windows",
      y = "Autocorrelation"
    ) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(), strip.text = ggplot2::element_text(size = 7, face = "bold"))
  ggplot2::ggsave(acf_plot_path, acf_plot, width = 13, height = 12, dpi = 180, bg = "white")

  figures <- data.frame(
    threshold_id = fit_result$threshold_id,
    model_id = fit_result$model_id,
    family = fit_result$family,
    fixed_surface = fit_result$fixed_surface,
    figure_type = c("DHARMa standard residual plot", "DHARMa residuals vs within-session window number", "Within-session residual ACF"),
    relative_path = file.path("figures", basename(c(dharma_plot_path, residual_plot_path, acf_plot_path))),
    description = c(
      "DHARMa standard diagnostic panel.",
      "Scaled residuals across ordered 60 s windows within each session.",
      "Within-session ACF of centered DHARMa scaled residuals."
    ),
    path = c(dharma_plot_path, residual_plot_path, acf_plot_path),
    stringsAsFactors = FALSE
  )
  outputs <- list(
    model_id = fit_result$model_id,
    family = fit_result$family,
    fixed_surface = fit_result$fixed_surface,
    formula_text = fit_result$formula,
    random_effects = fit_result$random_effects,
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
  rq2_write_progress_row(file.path(output_dir, "cache", paste0(OUTPUT_PREFIX, "_model_fit_progress.csv")), list(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    model_id = fit_result$model_id,
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
  term_labels <- unname(PREDICTOR_DISPLAY_LABELS[term_order])
  plot_data$threshold_label <- display_label(plot_data$threshold_id, THRESHOLD_DISPLAY_LABELS)
  plot_data$threshold_label <- factor(plot_data$threshold_label, levels = display_label(THRESHOLDS, THRESHOLD_DISPLAY_LABELS))
  plot_data$model_label <- model_display_label(plot_data$family, plot_data$fixed_surface)
  plot_data$model_label <- factor(
    plot_data$model_label,
    levels = c(
      "Poisson, with interaction",
      "Poisson, without interaction",
      "Negative binomial, with interaction",
      "Negative binomial, without interaction"
    )
  )
  plot_data$term_label <- display_label(plot_data$term, PREDICTOR_DISPLAY_LABELS)
  plot_data$term_label <- factor(plot_data$term_label, levels = rev(term_labels))
  path <- file.path(figures_dir, paste0(OUTPUT_PREFIX, "_fixed_effects_incidence_rate_ratios.png"))
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = incidence_rate_ratio, y = term_label, color = threshold_label)) +
    ggplot2::geom_vline(xintercept = 1, linetype = "dashed", linewidth = 0.3, color = "grey45") +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = irr_95_low, xmax = irr_95_high), orientation = "y", width = 0.18, position = ggplot2::position_dodge(width = 0.55)) +
    ggplot2::geom_point(position = ggplot2::position_dodge(width = 0.55), size = 1.6) +
    ggplot2::scale_x_log10() +
    ggplot2::facet_wrap(~model_label, ncol = 2) +
    ggplot2::labs(
      title = PLOT_TITLE_PREFIX,
      subtitle = "Incidence-rate ratios per one SD increase; coefficient-level Wald intervals.",
      x = "Incidence-rate ratio, log scale",
      y = NULL,
      color = "Threshold"
    ) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(), legend.position = "bottom")
  ggplot2::ggsave(path, p, width = 11, height = 7.0, dpi = 180, bg = "white")
  data.frame(
    threshold_id = "all",
    model_id = "all_models",
    family = "all",
    fixed_surface = "all",
    figure_type = "Fixed-effect incidence-rate ratio plot",
    relative_path = file.path("figures", basename(path)),
    description = "Incidence-rate ratios across thresholds, families, and fixed surfaces.",
    path = path,
    stringsAsFactors = FALSE
  )
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

if (!file.exists(input_path)) stop(paste("missing", RUN_LABEL, "model input:", input_path))
input <- utils::read.csv(input_path, check.names = FALSE)
missing <- setdiff(REQUIRED, names(input))
if (length(missing)) stop(paste(RUN_LABEL, "model input missing required columns:", paste(missing, collapse = ", ")))

specs <- model_specs()
specs$random_effects_structure_version <- RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION
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
  message("Fitting ", RUN_LABEL, " RQ2 model: ", spec$model_id)
  prepared <- prepare_model_data(input, spec)
  data <- prepared$data
  if (nrow(data) == 0L) stop(paste("no complete 60 s rows for", RUN_LABEL, "model", spec$model_id))
  if (length(unique(data$session_id)) < 2L) stop(paste("fewer than two session_id levels for", RUN_LABEL, "model", spec$model_id))

  fit_result <- fit_one(spec, data)

  manifest_rows[[length(manifest_rows) + 1L]] <- manifest_row(fit_result, data)
  fixed_rows[[length(fixed_rows) + 1L]] <- fixed_effect_table(fit_result)
  standardization_rows[[length(standardization_rows) + 1L]] <- prepared$standardization
  random_slope_rows[[length(random_slope_rows) + 1L]] <- cbind(
    threshold_id = spec$threshold_id,
    rq2_random_slope_decisions(data, spec)
  )
  dharma <- dharma_outputs(fit_result, data, cfg$seed + i)
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
figure_index <- bind_rows_fill(c(figure_rows, list(write_fixed_effect_plot(fixed_effects))))
environment <- environment_table()

utils::write.csv(
  data.frame(
    model_id = specs$model_id,
    threshold_id = specs$threshold_id,
    family = specs$family,
    fixed_surface = specs$fixed_surface,
    formula_text = specs$formula,
    random_effects = specs$random_effects,
    random_effects_structure_version = RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION,
    stringsAsFactors = FALSE
  ),
  file.path(tables_dir, paste0(OUTPUT_PREFIX, "_model_formula.csv")),
  row.names = FALSE,
  na = ""
)
utils::write.csv(run_manifest, file.path(tables_dir, paste0(OUTPUT_PREFIX, "_model_run_manifest.csv")), row.names = FALSE, na = "")
utils::write.csv(fixed_effects, file.path(tables_dir, paste0(OUTPUT_PREFIX, "_fixed_effects.csv")), row.names = FALSE, na = "")
utils::write.csv(standardization, file.path(tables_dir, paste0(OUTPUT_PREFIX, "_predictor_standardization.csv")), row.names = FALSE, na = "")
utils::write.csv(random_slope_decisions, file.path(tables_dir, paste0(OUTPUT_PREFIX, "_random_slope_decisions.csv")), row.names = FALSE, na = "")
utils::write.csv(dharma_tests, file.path(tables_dir, paste0(OUTPUT_PREFIX, "_dharma_standard_tests.csv")), row.names = FALSE, na = "")
utils::write.csv(temporal_tests, file.path(tables_dir, paste0(OUTPUT_PREFIX, "_dharma_temporal_autocorrelation.csv")), row.names = FALSE, na = "")
utils::write.csv(scaled_residuals, gzfile(file.path(tables_dir, paste0(OUTPUT_PREFIX, "_scaled_residuals.csv.gz"))), row.names = FALSE, na = "")
utils::write.csv(acf_table, file.path(tables_dir, paste0(OUTPUT_PREFIX, "_residual_acf.csv")), row.names = FALSE, na = "")
utils::write.csv(figure_index, file.path(tables_dir, paste0(OUTPUT_PREFIX, "_raw_figure_index.csv")), row.names = FALSE, na = "")
utils::write.csv(environment, file.path(tables_dir, paste0(OUTPUT_PREFIX, "_environment.csv")), row.names = FALSE, na = "")
writeLines(capture.output(utils::sessionInfo()), con = file.path(reports_dir, paste0(OUTPUT_PREFIX, "_session_info.txt")))
message("Wrote ", RUN_LABEL, " RQ2 outputs to ", output_dir)
