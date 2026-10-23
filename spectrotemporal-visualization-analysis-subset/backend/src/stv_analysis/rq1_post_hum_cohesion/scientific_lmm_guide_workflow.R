# RQ1 scientific-guide LMM workflow.
#
# This script implements the RQ1 mixed-model revamp against the materialized
# matched-control views created by scientific_lmm_materialize.py. The modeling
# decisions inspect data first, transform only when useful, choose random
# slopes from the design/within-group variation, fit full/null ML tests, fit
# REML lmerTest coefficient tests, and report convergence, random effects,
# residuals, random effects, collinearity, and stability.

options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  out <- list(
    data_dir = "backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/data",
    output_dir = "backend/data/processed/analysis/rq1_post_hum_cohesion/scientific_lmm_revamp/h1_model_outputs",
    mode = "full",
    max_rows_per_run = NA_integer_,
    stability = "design",
    only_run = "",
    aggregate_only = FALSE,
    skip_aggregate = FALSE,
    run_bootstrap = FALSE,
    optimizer_checkpoint_every = 25L,
    optimizer = "nloptwrap",
    profile_ci = FALSE,
    profile_parallel = "no",
    profile_ncpus = 1L,
    profile_maxpts = 20L,
    profile_devmatchtol = 0.01,
    run_config_csv = "",
    primary_model_id = "",
    predictor_set = "rq1_predictor_revamp",
    fixed_effect_variant = "additive",
    run_aic_predictor_scores = FALSE
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    value <- if (i < length(args)) args[[i + 1]] else NA_character_
    if (key == "--data-dir") {
      out$data_dir <- value; i <- i + 2
    } else if (key == "--output-dir") {
      out$output_dir <- value; i <- i + 2
    } else if (key == "--mode") {
      out$mode <- value; i <- i + 2
    } else if (key == "--max-rows-per-run") {
      out$max_rows_per_run <- as.integer(value); i <- i + 2
    } else if (key == "--stability") {
      out$stability <- value; i <- i + 2
    } else if (key == "--only-run") {
      out$only_run <- value; i <- i + 2
    } else if (key == "--aggregate-only") {
      out$aggregate_only <- TRUE; i <- i + 1
    } else if (key == "--skip-aggregate") {
      out$skip_aggregate <- TRUE; i <- i + 1
    } else if (key == "--run-bootstrap") {
      out$run_bootstrap <- TRUE; i <- i + 1
    } else if (key == "--optimizer-checkpoint-every") {
      out$optimizer_checkpoint_every <- as.integer(value); i <- i + 2
    } else if (key == "--predictor-set") {
      out$predictor_set <- value; i <- i + 2
    } else if (key == "--optimizer") {
      out$optimizer <- value; i <- i + 2
    } else if (key == "--profile-ci") {
      out$profile_ci <- TRUE; i <- i + 1
    } else if (key == "--profile-parallel") {
      out$profile_parallel <- value; i <- i + 2
    } else if (key == "--profile-ncpus") {
      out$profile_ncpus <- as.integer(value); i <- i + 2
    } else if (key == "--profile-maxpts") {
      out$profile_maxpts <- as.integer(value); i <- i + 2
    } else if (key == "--profile-devmatchtol") {
      out$profile_devmatchtol <- as.numeric(value); i <- i + 2
    } else if (key == "--run-config-csv") {
      out$run_config_csv <- value; i <- i + 2
    } else if (key == "--primary-model-id") {
      out$primary_model_id <- value; i <- i + 2
    } else if (key == "--fixed-effect-variant") {
      out$fixed_effect_variant <- value; i <- i + 2
    } else if (key == "--run-aic-predictor-scores") {
      out$run_aic_predictor_scores <- TRUE; i <- i + 1
    } else {
      stop(paste("unknown argument", key))
    }
  }
  if (!out$predictor_set %in% c("rq1_predictor_revamp")) {
    stop(paste("unknown --predictor-set", out$predictor_set))
  }
  if (!out$fixed_effect_variant %in% c("additive")) {
    stop(paste("unknown --fixed-effect-variant", out$fixed_effect_variant))
  }
  if (!out$optimizer %in% c("bobyqa", "nloptwrap")) {
    stop(paste("unknown --optimizer", out$optimizer))
  }
  if (!out$profile_parallel %in% c("no", "multicore", "snow")) {
    stop(paste("unknown --profile-parallel", out$profile_parallel))
  }
  out
}

cfg <- parse_args(args)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
script_dir <- dirname(normalizePath(script_path))
repo_root <- normalizePath(file.path(script_dir, "../../../.."))
resolve_workflow_path <- function(path_value) {
  if (grepl("^/", path_value)) {
    return(normalizePath(path_value, mustWork = FALSE))
  }
  normalizePath(file.path(repo_root, path_value), mustWork = FALSE)
}
data_dir <- resolve_workflow_path(cfg$data_dir)
output_dir <- resolve_workflow_path(cfg$output_dir)
tables_dir <- file.path(output_dir, "tables")
figures_dir <- file.path(output_dir, "figures")
reports_dir <- file.path(output_dir, "reports")
cache_dir <- file.path(output_dir, "cache")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(reports_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    repos <- getOption("repos")
    if (is.null(repos) || identical(unname(repos["CRAN"]), "@CRAN@")) {
      repos <- c(CRAN = "https://cloud.r-project.org")
    }
    install.packages(pkg, repos = repos)
  }
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("R package is required but unavailable:", pkg))
  }
}

ensure_package("lme4")
ensure_package("lmerTest")
ensure_package("ggplot2")
library(lme4)

guide_code <- file.path(script_dir, "scientific_lmm_helpers.R")
if (!file.exists(guide_code)) {
  stop(paste("missing guide helper code:", guide_code))
}
source(guide_code)

set.seed(20260704)

all_raw_predictor_specs <- data.frame(
  raw = c(
    "starting_centroid_distance_m",
    "mean_speed_pre_m_s",
    "pre_delta_centroid_distance_m",
    "observed_polygon_area_pre_m2",
    "recent_change_in_polygon_area_m2"
  ),
  label = c(
    "starting_distance",
    "pre_window_speed",
    "recent_pre_event_distance_change",
    "observed_polygon_area_pre",
    "recent_change_in_polygon_area"
  ),
  nonnegative = c(TRUE, TRUE, FALSE, TRUE, FALSE),
  transform_policy = c(
    "log1p_then_z",
    "log1p_then_z",
    "none_then_z",
    "none_then_z",
    "none_then_z"
  ),
  stringsAsFactors = FALSE
)

select_predictor_specs <- function(predictor_set) {
  if (predictor_set == "rq1_predictor_revamp") {
    return(all_raw_predictor_specs)
  }
  stop(paste("unknown predictor set", predictor_set))
}

raw_predictor_specs <- select_predictor_specs(cfg$predictor_set)

FIXED_EFFECT_DISPLAY_LABELS <- c(
  "(Intercept)" = "Intercept",
  condition_hum = "hum presence",
  z_log1p_starting_distance = "starting focal-to-centroid distance",
  z_log1p_pre_window_speed = "pre-occurrence mean focal speed",
  z_recent_pre_event_distance_change = "recent focal-to-centroid distance change",
  z_observed_polygon_area_pre = "pre-occurrence spread",
  z_recent_change_in_polygon_area = "recent spread change"
)

fixed_effect_display_label <- function(values) {
  out <- as.character(values)
  matched <- out %in% names(FIXED_EFFECT_DISPLAY_LABELS)
  out[matched] <- unname(FIXED_EFFECT_DISPLAY_LABELS[out[matched]])
  out
}

read_view <- function(policy) {
  input_files <- c(standard_matching = "h1_contact_reinforcement_model_input.csv")
  if (!policy %in% names(input_files)) {
    stop(paste("unknown materialized view policy:", policy))
  }
  path <- file.path(data_dir, input_files[[policy]])
  if (!file.exists(path)) {
    stop(paste("missing materialized view:", path))
  }
  read.csv(path, check.names = FALSE)
}

resolve_repo_path <- function(path_value) {
  if (is.null(path_value) || is.na(path_value) || !nzchar(path_value)) {
    return("")
  }
  if (grepl("^/", path_value)) {
    return(path_value)
  }
  file.path(repo_root, path_value)
}

as_bool <- function(value) {
  if (is.logical(value)) return(isTRUE(value))
  text <- tolower(trimws(as.character(value)))
  text %in% c("true", "t", "1", "yes", "y")
}

read_configured_data <- local({
  cache <- new.env(parent = emptyenv())
  function(path_value) {
    path <- resolve_repo_path(path_value)
    if (!file.exists(path)) {
      stop(paste("missing configured model input:", path))
    }
    key <- normalizePath(path, mustWork = TRUE)
    if (!exists(key, envir = cache, inherits = FALSE)) {
      assign(key, read.csv(path, check.names = FALSE), envir = cache)
    }
    get(key, envir = cache, inherits = FALSE)
  }
})

load_run_configs <- function() {
  if (!nzchar(cfg$run_config_csv)) {
    configs <- data.frame(
      matching_policy = "standard_matching",
      window_s = c(30, 60, 120),
      include_match_set = FALSE,
      data_path = "",
      threshold_m = NA_real_,
      threshold_role = "",
      stringsAsFactors = FALSE
    )
    configs$model_id <- paste0("h1_", configs$window_s, "s")
    configs$predictor_set <- cfg$predictor_set
    configs$fixed_effect_variant <- cfg$fixed_effect_variant
    return(configs)
  }
  path <- resolve_repo_path(cfg$run_config_csv)
  if (!file.exists(path)) {
    stop(paste("missing --run-config-csv:", path))
  }
  configs <- read.csv(path, check.names = FALSE)
  required <- c("model_id", "window_s", "matching_policy", "include_match_set", "data_path")
  missing <- setdiff(required, names(configs))
  if (length(missing)) {
    stop(paste("run config is missing required columns:", paste(missing, collapse = ", ")))
  }
  configs$window_s <- as.integer(configs$window_s)
  configs$include_match_set <- vapply(configs$include_match_set, as_bool, logical(1))
  if (any(configs$matching_policy != "standard_matching")) {
    stop("RQ1 scientific LMM runs only support matching_policy == standard_matching")
  }
  if (any(configs$include_match_set)) {
    stop("RQ1 scientific LMM runs do not support match_set_id as an extra fixed or mixed effect")
  }
  if (!"threshold_m" %in% names(configs)) configs$threshold_m <- NA_real_
  if (!"threshold_role" %in% names(configs)) configs$threshold_role <- ""
  configs$predictor_set <- cfg$predictor_set
  configs$fixed_effect_variant <- cfg$fixed_effect_variant
  configs
}

run_context_from_config <- function(row) {
  context <- list()
  for (name in c("threshold_m", "threshold_role", "data_path")) {
    if (name %in% names(row)) {
      context[[name]] <- row[[name]]
    }
  }
  context
}

add_run_context <- function(result, context) {
  if (length(context) == 0) return(result)
  for (name in names(result)) {
    if (!is.data.frame(result[[name]]) || nrow(result[[name]]) == 0) next
    for (context_name in names(context)) {
      if (!context_name %in% names(result[[name]])) {
        result[[name]][[context_name]] <- context[[context_name]]
      }
    }
  }
  result
}

skewness <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  s <- stats::sd(x)
  if (!is.finite(s) || s == 0) return(0)
  mean(((x - mean(x)) / s)^3)
}

zscore <- function(x) {
  m <- mean(x, na.rm = TRUE)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) {
    return(list(value = rep(0, length(x)), mean = m, sd = s, note = "constant predictor; z-score set to 0"))
  }
  list(value = as.numeric((x - m) / s), mean = m, sd = s, note = "")
}

choose_transform <- function(values, label, raw_column, nonnegative, transform_policy) {
  values <- as.numeric(values)
  raw_skew <- skewness(values)
  use_log <- FALSE
  transformed <- values
  transform_name <- "none_then_z"
  reason <- "raw scale retained; z-scored"
  finite_values <- values[is.finite(values)]
  if (transform_policy == "log1p_then_z") {
    if (!isTRUE(nonnegative) || any(finite_values < 0)) {
      stop(paste("log1p transform requested for non-nonnegative predictor", label))
    }
    transformed <- log1p(values)
    use_log <- TRUE
    transform_name <- "log1p_then_z"
    reason <- "fixed revamp policy: log1p then z-score in every window"
  } else if (transform_policy == "none_then_z") {
    transformed <- values
    use_log <- FALSE
    transform_name <- "none_then_z"
    reason <- "fixed revamp policy: raw scale retained then z-scored in every window"
  } else if (transform_policy == "auto_log1p_then_z" && isTRUE(nonnegative) && all(finite_values >= 0)) {
    logged <- log1p(values)
    log_skew <- skewness(logged)
    if (is.finite(raw_skew) && is.finite(log_skew) && abs(raw_skew) > 1 && abs(log_skew) <= 0.8 * abs(raw_skew)) {
      use_log <- TRUE
      transformed <- logged
      transform_name <- "log1p_then_z"
      reason <- "raw distribution was strongly right-skewed and log1p reduced skew"
    }
  } else if (!transform_policy %in% c("auto_log1p_then_z", "none_then_z", "log1p_then_z")) {
    stop(paste("unknown transform policy", transform_policy, "for predictor", label))
  }
  z <- zscore(transformed)
  safe_label <- gsub("[^A-Za-z0-9]+", "_", label)
  z_name <- if (use_log) paste0("z_log1p_", safe_label) else paste0("z_", safe_label)
  list(
    z_name = z_name,
    values = z$value,
    table = data.frame(
      predictor = label,
      raw_column = raw_column,
      transform = transform_name,
      raw_mean = mean(values, na.rm = TRUE),
      raw_sd = stats::sd(values, na.rm = TRUE),
      raw_min = min(values, na.rm = TRUE),
      raw_max = max(values, na.rm = TRUE),
      raw_skewness = raw_skew,
      transform_policy = transform_policy,
      transformed_mean_before_z = z$mean,
      transformed_sd_before_z = z$sd,
      z_column = z_name,
      decision_reason = paste(c(reason, z$note), collapse = "; "),
      stringsAsFactors = FALSE
    )
  )
}

rank_smoke_rows <- function(z) {
  z$._stratum <- interaction(z$session_id, z$focal_id, drop = TRUE, sep = ":")
  z <- z[order(z$._stratum, z$match_set_id, z$analysis_window_id), , drop = FALSE]
  z$._rank_in_stratum <- ave(seq_len(nrow(z)), z$._stratum, FUN = seq_along)
  z <- z[order(z$._rank_in_stratum, z$._stratum, z$match_set_id, z$analysis_window_id), , drop = FALSE]
  z$._stratum <- NULL
  z$._rank_in_stratum <- NULL
  z
}

smoke_sample <- function(x, max_rows) {
  max_rows <- min(nrow(x), as.integer(max_rows))
  hum <- rank_smoke_rows(x[x$condition_hum == 1, , drop = FALSE])
  control <- rank_smoke_rows(x[x$condition_hum == 0, , drop = FALSE])

  n_hum <- if (nrow(hum) > 0) min(nrow(hum), max(1L, floor(max_rows / 2))) else 0L
  n_control <- if (nrow(control) > 0) min(nrow(control), max(1L, max_rows - n_hum)) else 0L

  while (n_hum + n_control < max_rows) {
    if (nrow(hum) > n_hum) {
      n_hum <- n_hum + 1L
    } else if (nrow(control) > n_control) {
      n_control <- n_control + 1L
    } else {
      break
    }
  }

  parts <- list()
  if (n_hum > 0) parts[[length(parts) + 1]] <- hum[seq_len(n_hum), , drop = FALSE]
  if (n_control > 0) parts[[length(parts) + 1]] <- control[seq_len(n_control), , drop = FALSE]
  out <- do.call(rbind, parts)
  out[order(out$session_id, out$focal_id, out$match_set_id, out$condition_hum), , drop = FALSE]
}

prepare_run_data <- function(df, window_s, policy, smoke = FALSE, max_rows = NA_integer_) {
  x <- df[df$window_s == window_s, , drop = FALSE]
  x$condition <- as.integer(x$condition)
  x$condition_hum <- as.numeric(x$condition == 1)
  x$condition_hum_c <- x$condition_hum - mean(x$condition_hum, na.rm = TRUE)
  x$focal_id <- factor(x$focal_id)
  x$session_id <- factor(x$session_id)
  x$match_set_id <- factor(x$match_set_id)
  x$Delta_D_cm <- if ("Delta_D_cm" %in% names(x)) as.numeric(x$Delta_D_cm) else as.numeric(x$Delta_D_w) * 100

  required <- c("Delta_D_cm", "condition_hum", "condition_hum_c", "focal_id", "session_id", "match_set_id", raw_predictor_specs$raw)
  before <- nrow(x)
  x <- x[stats::complete.cases(x[, required]), , drop = FALSE]
  after_complete <- nrow(x)

  if (is.finite(max_rows) && !is.na(max_rows) && nrow(x) > max_rows) {
    x <- smoke_sample(x, max_rows)
    x$focal_id <- droplevels(factor(x$focal_id))
    x$session_id <- droplevels(factor(x$session_id))
    x$match_set_id <- droplevels(factor(x$match_set_id))
  }
  x$condition_hum_c <- x$condition_hum - mean(x$condition_hum, na.rm = TRUE)

  transform_rows <- list()
  transformed_predictors <- character()
  transformed_by_label <- list()
  for (i in seq_len(nrow(raw_predictor_specs))) {
    raw <- raw_predictor_specs$raw[[i]]
    label <- raw_predictor_specs$label[[i]]
    values <- as.numeric(x[[raw]])
    names(values) <- raw
    decision <- choose_transform(values, label, raw, raw_predictor_specs$nonnegative[[i]], raw_predictor_specs$transform_policy[[i]])
    x[[decision$z_name]] <- decision$values
    transformed_predictors <- c(transformed_predictors, decision$z_name)
    transformed_by_label[[label]] <- decision$z_name
    transform_rows[[length(transform_rows) + 1]] <- cbind(
      matching_policy = policy,
      window_s = window_s,
      decision$table
    )
  }
  transform_table <- do.call(rbind, transform_rows)
  list(
    data = x,
    transformed_predictors = transformed_predictors,
    transformed_by_label = transformed_by_label,
    transform_table = transform_table,
    dropped_incomplete_rows = before - after_complete
  )
}

write_distribution_plots <- function(data, transformed_predictors, run_id) {
  path <- file.path(figures_dir, paste0(run_id, "_model_distributions.pdf"))
  grDevices::pdf(path, width = 10, height = 8)
  old <- par(no.readonly = TRUE)
  on.exit({par(old); grDevices::dev.off()}, add = TRUE)
  par(mfrow = c(3, 3), mar = c(3, 3, 2, 1))
  hist(data$Delta_D_w, main = "Delta_D_w raw metres", xlab = "m", col = "white")
  hist(data$Delta_D_cm, main = "Delta_D_cm response", xlab = "cm", col = "white")
  hist(data$condition_hum, main = "condition_hum", xlab = "0 control, 1 hum", col = "white")
  for (raw in raw_predictor_specs$raw) {
    hist(as.numeric(data[[raw]]), main = raw, xlab = "", col = "white")
  }
  for (pred in transformed_predictors) {
    hist(as.numeric(data[[pred]]), main = pred, xlab = "z", col = "white")
  }
  invisible(path)
}

build_model_surface <- function(data, transformed_predictors, transformed_by_label) {
  starting_z <- transformed_by_label[["starting_distance"]]
  if (is.null(starting_z) || !nzchar(starting_z)) {
    stop("starting_distance transformed predictor is unavailable")
  }
  full_rhs <- c("condition_hum", transformed_predictors)
  null_rhs <- transformed_predictors
  vif_terms <- full_rhs
  slope_specs <- data.frame(
    fixed_term = c("condition_hum", transformed_predictors),
    random_term = c("condition_hum_c", transformed_predictors),
    data_column = c("condition_hum", transformed_predictors),
    predictor_type = c("binary_factor_rule", rep("covariate_rule", length(transformed_predictors))),
    stringsAsFactors = FALSE
  )
  list(
    data = data,
    full_rhs = full_rhs,
    null_rhs = null_rhs,
    vif_terms = vif_terms,
    slope_specs = slope_specs,
    starting_z = starting_z,
    transformed_by_label = transformed_by_label
  )
}

random_slope_decisions <- function(data, slope_specs) {
  rows <- list()
  for (group in c("focal_id", "session_id")) {
    for (i in seq_len(nrow(slope_specs))) {
      fixed_term <- slope_specs$fixed_term[[i]]
      random_term <- slope_specs$random_term[[i]]
      data_column <- slope_specs$data_column[[i]]
      predictor_type <- slope_specs$predictor_type[[i]]
      if (predictor_type == "binary_factor_rule") {
        tab <- table(data[[group]], data[[data_column]])
        ok_by_level <- rowSums(tab >= 2) >= 2
        rule_1_levels_ok <- NA_integer_
        rule_2_levels_ok <- sum(ok_by_level)
        include <- mean(ok_by_level) >= 0.5
        rule <- "factor: at least two levels with at least two observations in at least half of grouping levels"
      } else {
        unique_counts <- tapply(data[[data_column]], data[[group]], function(z) length(unique(z[is.finite(z)])))
        ok_rule_1 <- unique_counts >= 3
        rule_2_counts <- tapply(
          seq_len(nrow(data)),
          data[[group]],
          function(idx) {
            z <- data[[data_column]][idx]
            z <- z[is.finite(z)]
            if (!length(z)) return(0L)
            sum(table(z) >= 2)
          }
        )
        ok_rule_2 <- rule_2_counts >= 2
        rule_1_levels_ok <- sum(ok_rule_1, na.rm = TRUE)
        rule_2_levels_ok <- sum(ok_rule_2, na.rm = TRUE)
        include <- mean(ok_rule_1, na.rm = TRUE) >= 0.5 || mean(ok_rule_2, na.rm = TRUE) >= 0.5
        rule <- "covariate: at least three unique values, or factor-style identifiability, in at least half of grouping levels"
      }
      rows[[length(rows) + 1]] <- data.frame(
        grouping_factor = group,
        fixed_term = fixed_term,
        random_slope_term = random_term,
        random_slope_data_column = data_column,
        predictor_type = predictor_type,
        n_group_levels = length(unique(data[[group]])),
        rule_1_levels_ok = rule_1_levels_ok,
        rule_2_levels_ok = rule_2_levels_ok,
        include_random_slope = include,
        decision_rule = rule,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

random_formula_from_decisions <- function(decisions, include_match_set, omit_random_terms = character()) {
  terms <- character()
  for (group in c("focal_id", "session_id")) {
    slopes <- decisions$random_slope_term[decisions$grouping_factor == group & decisions$include_random_slope]
    slopes <- unique(slopes)
    slopes <- setdiff(slopes, omit_random_terms)
    if (length(slopes) == 0) {
      terms <- c(terms, paste0("(1 | ", group, ")"))
    } else {
      terms <- c(terms, paste0("(1 + ", paste(slopes, collapse = " + "), " | ", group, ")"))
    }
  }
  if (include_match_set) {
    terms <- c(terms, "(1 | match_set_id)")
  }
  paste(terms, collapse = " + ")
}

safe_fe_re_summary <- function(data, full_fixed_formula, run_id) {
  path <- file.path(tables_dir, paste0(run_id, "_fixed_random_effects_summary.txt"))
  result <- tryCatch(
    {
      tab <- fe.re.tab(
        fe.model = full_fixed_formula,
        re = "(1|focal_id)+(1|session_id)",
        data = data
      )
      write.fe.re.summary2file(tab$summary, file = path)
      "written"
    },
    error = function(e) {
      writeLines(paste("fe.re.tab failed:", conditionMessage(e)), con = path)
      "failed"
    }
  )
  result
}

atomic_save_rds <- function(object, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  saveRDS(object, tmp)
  if (file.exists(path)) {
    unlink(path)
  }
  file.rename(tmp, path)
}

optimizer_progress_path <- function(trace_path) {
  sub("\\.rds$", "_progress.csv", trace_path)
}

write_optimizer_checkpoint <- function(
  trace_path,
  label,
  formula_text,
  reml,
  use_lmer_test,
  optimizer_name,
  eval_count,
  current_theta,
  current_fval,
  best_theta,
  best_fval,
  status
) {
  if (is.null(trace_path) || !nzchar(trace_path)) return(invisible(NULL))
  checkpoint <- list(
    label = label,
    formula_text = formula_text,
    reml = reml,
    use_lmer_test = use_lmer_test,
    optimizer_name = optimizer_name,
    eval_count = eval_count,
    current_theta = current_theta,
    current_fval = current_fval,
    best_theta = best_theta,
    best_fval = best_fval,
    status = status,
    updated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  )
  atomic_save_rds(checkpoint, trace_path)
  progress_path <- optimizer_progress_path(trace_path)
  row <- data.frame(
    timestamp = checkpoint$updated_at,
    label = label,
    eval_count = eval_count,
    current_fval = current_fval,
    best_fval = best_fval,
    status = status,
    stringsAsFactors = FALSE
  )
  utils::write.table(
    row,
    file = progress_path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(progress_path),
    append = file.exists(progress_path)
  )
  invisible(NULL)
}

load_optimizer_start <- function(trace_path, formula_text, reml, use_lmer_test, optimizer_name) {
  if (is.null(trace_path) || !file.exists(trace_path)) return(NULL)
  checkpoint <- tryCatch(readRDS(trace_path), error = function(e) NULL)
  if (is.null(checkpoint)) return(NULL)
  if (!identical(checkpoint$formula_text, formula_text)) return(NULL)
  if (!identical(isTRUE(checkpoint$reml), isTRUE(reml))) return(NULL)
  if (!identical(isTRUE(checkpoint$use_lmer_test), isTRUE(use_lmer_test))) return(NULL)
  if (!is.null(checkpoint$optimizer_name) && !identical(checkpoint$optimizer_name, optimizer_name)) return(NULL)
  theta <- checkpoint$best_theta
  if (is.null(theta) || !length(theta) || any(!is.finite(theta))) return(NULL)
  list(theta = theta)
}

checkpointed_deviance <- function(
  fn,
  par,
  trace_path,
  label,
  formula_text,
  reml,
  use_lmer_test,
  optimizer_name,
  checkpoint_every
) {
  eval_count <- 0L
  best_theta <- par
  best_fval <- Inf
  current_theta <- par
  current_fval <- Inf
  checkpoint_every <- max(1L, as.integer(checkpoint_every))
  write_optimizer_checkpoint(
    trace_path, label, formula_text, reml, use_lmer_test, optimizer_name,
    eval_count, current_theta, current_fval, best_theta, best_fval,
    status = "optimizer_started"
  )
  wrapped_fn <- function(theta) {
    value <- fn(theta)
    eval_count <<- eval_count + 1L
    current_theta <<- theta
    current_fval <<- value
    if (is.finite(value) && value < best_fval) {
      best_theta <<- theta
      best_fval <<- value
    }
    if (eval_count %% checkpoint_every == 0L) {
      write_optimizer_checkpoint(
        trace_path, label, formula_text, reml, use_lmer_test, optimizer_name,
        eval_count, current_theta, current_fval, best_theta, best_fval,
        status = "optimizer_running"
      )
    }
    value
  }
  list(
    fn = wrapped_fn,
    finalize = function(status = "optimizer_exit") {
      write_optimizer_checkpoint(
        trace_path, label, formula_text, reml, use_lmer_test, optimizer_name,
        eval_count, current_theta, current_fval, best_theta, best_fval,
        status = status
      )
    }
  )
}

make_checkpointing_bobyqa <- function(trace_path, label, formula_text, reml, use_lmer_test, checkpoint_every) {
  force(trace_path)
  force(label)
  force(formula_text)
  force(reml)
  force(use_lmer_test)
  force(checkpoint_every)
  function(fn, par, lower, upper, control = list(), ...) {
    if (!requireNamespace("minqa", quietly = TRUE)) {
      stop("minqa is required for checkpointing bobyqa optimizer")
    }
    if (all(par == 0)) par[] <- 0.001
    if (!is.numeric(control$maxfun)) control$maxfun <- 200000
    if (!is.numeric(control$rhobeg)) control$rhobeg <- 2e-04
    if (!is.numeric(control$rhoend)) control$rhoend <- 2e-07
    if (!is.numeric(control$iprint)) control$iprint <- 0L
    checkpoint <- checkpointed_deviance(
      fn, par, trace_path, label, formula_text, reml, use_lmer_test,
      optimizer_name = "bobyqa", checkpoint_every = checkpoint_every
    )
    on.exit(checkpoint$finalize(), add = TRUE)
    opt <- minqa::bobyqa(par, checkpoint$fn, lower = lower, upper = upper, control = control, ...)
    list(
      par = opt$par,
      fval = opt$fval,
      feval = opt$feval,
      conv = opt$ierr,
      message = opt$msg
    )
  }
}

make_checkpointing_nloptwrap <- function(trace_path, label, formula_text, reml, use_lmer_test, checkpoint_every) {
  force(trace_path)
  force(label)
  force(formula_text)
  force(reml)
  force(use_lmer_test)
  force(checkpoint_every)
  function(par, fn, lower, upper, control = list(), ...) {
    if (!requireNamespace("nloptr", quietly = TRUE)) {
      stop("nloptr is required for checkpointing nloptwrap optimizer")
    }
    defaults <- environment(lme4::nloptwrap)$defaultControl
    for (name in names(defaults)) {
      if (is.null(control[[name]])) control[[name]] <- defaults[[name]]
    }
    if (!is.numeric(control$maxeval)) control$maxeval <- 200000
    checkpoint <- checkpointed_deviance(
      fn, par, trace_path, label, formula_text, reml, use_lmer_test,
      optimizer_name = "nloptwrap", checkpoint_every = checkpoint_every
    )
    on.exit(checkpoint$finalize(), add = TRUE)
    opt <- nloptr::nloptr(
      x0 = par,
      eval_f = checkpoint$fn,
      lb = lower,
      ub = upper,
      opts = control,
      ...
    )
    list(
      par = opt$solution,
      fval = opt$objective,
      feval = opt$iterations,
      conv = if (opt$status < 0 || opt$status == 5) opt$status else 0,
      message = opt$message
    )
  }
}

make_checkpointing_optimizer <- function(optimizer_name, trace_path, label, formula_text, reml, use_lmer_test, checkpoint_every) {
  if (optimizer_name == "bobyqa") {
    return(make_checkpointing_bobyqa(trace_path, label, formula_text, reml, use_lmer_test, checkpoint_every))
  }
  if (optimizer_name == "nloptwrap") {
    return(make_checkpointing_nloptwrap(trace_path, label, formula_text, reml, use_lmer_test, checkpoint_every))
  }
  stop(paste("unsupported checkpointing optimizer", optimizer_name))
}

optimizer_control <- function(optimizer_name) {
  if (optimizer_name == "nloptwrap") {
    return(list(maxeval = 200000))
  }
  list(maxfun = 200000)
}

fit_lmer_capture <- function(
  formula_text,
  data,
  reml,
  use_lmer_test = FALSE,
  start = NULL,
  trace_path = NULL,
  label = "lmer fit",
  checkpoint_every = 25L,
  optimizer_name = cfg$optimizer
) {
  warnings_seen <- character()
  started <- proc.time()[[3]]
  optimizer <- if (!is.null(trace_path) && is.finite(checkpoint_every) && checkpoint_every > 0L) {
    make_checkpointing_optimizer(optimizer_name, trace_path, label, formula_text, reml, use_lmer_test, checkpoint_every)
  } else {
    optimizer_name
  }
  fit_once <- function(start_arg = NULL) {
    call_args <- list(
      formula = stats::as.formula(formula_text),
      data = data,
      REML = reml,
      control = lme4::lmerControl(
        optimizer = optimizer,
        optCtrl = optimizer_control(optimizer_name),
        calc.derivs = TRUE
      )
    )
    if (!is.null(start_arg)) {
      call_args$start <- start_arg
    }
    if (use_lmer_test) {
      do.call(lmerTest::lmer, call_args)
    } else {
      do.call(lme4::lmer, call_args)
    }
  }
  fit <- withCallingHandlers(
    tryCatch(
      {
        fit_once(start)
      },
      error = function(e) {
        if (!is.null(start)) {
          warnings_seen <<- c(warnings_seen, paste("optimizer checkpoint start rejected; retrying from default start:", conditionMessage(e)))
          return(fit_once(NULL))
        }
        e
      }
    ),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(fit = fit, warnings = unique(warnings_seen), runtime_seconds = proc.time()[[3]] - started)
}

fit_lmer_cached <- function(cache_path, label, formula_text, data, reml, use_lmer_test = FALSE) {
  if (file.exists(cache_path)) {
    message("Reusing cached ", label, ": ", cache_path)
    return(readRDS(cache_path))
  }
  message("Fitting ", label, " started at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  trace_path <- sub("\\.rds$", "_optimizer_trace.rds", cache_path)
  start <- load_optimizer_start(trace_path, formula_text, reml = reml, use_lmer_test = use_lmer_test, optimizer_name = cfg$optimizer)
  if (!is.null(start)) {
    message("Reusing optimizer checkpoint start for ", label, ": ", trace_path)
  }
  fit <- fit_lmer_capture(
    formula_text,
    data,
    reml = reml,
    use_lmer_test = use_lmer_test,
    start = start,
    trace_path = trace_path,
    label = label,
    checkpoint_every = cfg$optimizer_checkpoint_every,
    optimizer_name = cfg$optimizer
  )
  saveRDS(fit, cache_path)
  message(
    "Fitting ", label, " finished at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    "; runtime_seconds=", round(fit$runtime_seconds, 2),
    "; status=", fit_status(fit$fit)
  )
  fit
}

validate_lmer_formula <- function(formula_text, data) {
  warnings_seen <- character()
  started <- proc.time()[[3]]
  parsed <- withCallingHandlers(
    tryCatch(
      lme4::lFormula(stats::as.formula(formula_text), data = data, REML = FALSE),
      error = function(e) e
    ),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(
    ok = !inherits(parsed, "error"),
    error = if (inherits(parsed, "error")) conditionMessage(parsed) else "",
    warnings = unique(warnings_seen),
    runtime_seconds = proc.time()[[3]] - started
  )
}

max_gradient <- function(fit) {
  gradient <- fit@optinfo$derivs$gradient
  if (is.null(gradient)) return(NA_real_)
  max(abs(gradient), na.rm = TRUE)
}

fit_status <- function(fit) {
  if (inherits(fit, "error")) return("failed")
  messages <- fit@optinfo$conv$lme4$messages
  opt_code <- fit@optinfo$conv$opt
  non_boundary <- messages[!grepl("^boundary \\(singular\\) fit", messages)]
  converged <- isTRUE(as.integer(opt_code) == 0L) && length(non_boundary) == 0
  singular <- lme4::isSingular(fit, tol = 1e-4)
  if (!converged) return("unstable")
  if (singular) return("acceptable_boundary_simplified")
  "acceptable"
}

varcorr_table <- function(fit, run_id) {
  if (inherits(fit, "error")) {
    return(data.frame())
  }
  vc <- as.data.frame(lme4::VarCorr(fit))
  vc$model_id <- run_id
  vc$boundary_flag <- ifelse(
    is.na(vc$var2),
    is.finite(vc$sdcor) & abs(vc$sdcor) <= 1e-6,
    is.finite(vc$sdcor) & abs(vc$sdcor) >= 0.99
  )
  vc
}

fixed_effect_table <- function(fit_reml, run_id, window_s, policy, match_policy, formula_text, n_rows, n_hum, n_control) {
  if (inherits(fit_reml, "error")) {
    return(data.frame())
  }
  coefs <- coef(summary(fit_reml, ddf = "Satterthwaite"))
  out <- as.data.frame(coefs)
  out$term <- rownames(out)
  rownames(out) <- NULL
  names(out) <- sub("Std\\. Error", "std_error", names(out))
  names(out) <- sub("t value", "test_statistic", names(out), fixed = TRUE)
  names(out) <- sub("Pr\\(>\\|t\\|\\)", "p_value", names(out))
  out$ci95_low_cm <- out$Estimate - stats::qt(0.975, out$df) * out$std_error
  out$ci95_high_cm <- out$Estimate + stats::qt(0.975, out$df) * out$std_error
  out$model_id <- run_id
  out$window_s <- window_s
  out$matching_policy <- policy
  out$match_set_random_effect <- match_policy
  out$n_rows <- n_rows
  out$n_hum_rows <- n_hum
  out$n_control_rows <- n_control
  out$model_family <- "linear mixed-effects model"
  out$model_link <- "identity"
  out$response_unit <- "centimeters"
  out$p_value_method <- "lmerTest Satterthwaite"
  out$ci_method <- "lmerTest Satterthwaite Wald t 95"
  out$implemented_formula <- formula_text
  out[, c(
    "model_id", "window_s", "matching_policy", "match_set_random_effect",
    "term", "Estimate", "std_error", "df", "test_statistic", "p_value",
    "ci95_low_cm", "ci95_high_cm", "response_unit", "p_value_method",
    "ci_method", "model_family", "model_link", "n_rows", "n_hum_rows",
    "n_control_rows", "implemented_formula"
  )]
}

compute_vif <- function(data, terms) {
  design <- stats::model.matrix(stats::as.formula(paste("~", paste(terms, collapse = " + "))), data = data)
  design <- design[, colnames(design) != "(Intercept)", drop = FALSE]
  rows <- list()
  for (term in colnames(design)) {
    others <- setdiff(colnames(design), term)
    if (length(others) == 0) {
      vif <- 1
    } else {
      frame <- as.data.frame(design[, c(term, others), drop = FALSE])
      fit <- stats::lm(stats::as.formula(paste(term, "~", paste(others, collapse = " + "))), data = frame)
      r2 <- summary(fit)$r.squared
      vif <- if (is.finite(r2) && r2 < 1) 1 / (1 - r2) else Inf
    }
    rows[[length(rows) + 1]] <- data.frame(term = term, VIF = vif)
  }
  do.call(rbind, rows)
}

residual_outputs <- function(fit, data, run_id) {
  if (inherits(fit, "error")) {
    return(list(summary = data.frame(), residuals = data.frame()))
  }
  residual <- stats::residuals(fit)
  fitted <- stats::fitted(fit)
  sigma <- stats::sigma(fit)
  standardized <- residual / sigma
  row_level <- data.frame(
    model_id = run_id,
    analysis_window_id = data$analysis_window_id,
    match_set_id = data$match_set_id,
    condition = ifelse(data$condition_hum == 1, "hum", "control"),
    window_s = data$window_s,
    focal_id = data$focal_id,
    session_id = data$session_id,
    observed_delta_D_cm = data$Delta_D_cm,
    fitted_delta_D_cm = fitted,
    residual_cm = residual,
    standardized_residual = standardized,
    stringsAsFactors = FALSE
  )
  summary <- data.frame(
    model_id = run_id,
    n_residuals = length(residual),
    residual_mean_cm = mean(residual),
    residual_sd_cm = stats::sd(residual),
    residual_median_cm = stats::median(residual),
    residual_iqr_cm = stats::IQR(residual),
    residual_q05_cm = unname(stats::quantile(residual, 0.05)),
    residual_q95_cm = unname(stats::quantile(residual, 0.95)),
    standardized_max_abs = max(abs(standardized), na.rm = TRUE),
    residual_spread_hum_cm = stats::sd(residual[data$condition_hum == 1]),
    residual_spread_control_cm = stats::sd(residual[data$condition_hum == 0]),
    stringsAsFactors = FALSE
  )
  list(summary = summary, residuals = row_level)
}

write_model_diagnostic_plots <- function(fit, run_id) {
  if (inherits(fit, "error")) return(character())
  paths <- c(
    residual = file.path(figures_dir, paste0(run_id, "_residual_diagnostics.pdf")),
    random_effect_hist = file.path(figures_dir, paste0(run_id, "_random_effect_histograms.pdf")),
    random_effect_qq = file.path(figures_dir, paste0(run_id, "_random_effect_qq.pdf"))
  )
  grDevices::pdf(paths[["residual"]], width = 8, height = 8)
  diagnostics.plot(fit)
  grDevices::dev.off()
  grDevices::pdf(paths[["random_effect_hist"]], width = 10, height = 8)
  ranef.diagn.plot(fit, QQ = FALSE)
  grDevices::dev.off()
  grDevices::pdf(paths[["random_effect_qq"]], width = 10, height = 8)
  ranef.diagn.plot(fit, QQ = TRUE)
  grDevices::dev.off()
  paths
}

write_random_effects <- function(fit, data, run_id, run_dir) {
  if (inherits(fit, "error")) return(data.frame())
  values <- lme4::ranef(fit)
  rows <- list()
  idx <- 1
  for (group in names(values)) {
    current <- values[[group]]
    counts <- if (group %in% names(data)) table(data[[group]]) else integer()
    for (level in rownames(current)) {
      for (term in colnames(current)) {
        rows[[idx]] <- data.frame(
          model_id = run_id,
          grouping_factor = group,
          group_id = level,
          random_effect_term = term,
          random_effect_estimate_cm = current[level, term],
          n_rows = if (length(counts) && level %in% names(counts)) as.integer(counts[[level]]) else NA_integer_,
          stringsAsFactors = FALSE
        )
        idx <- idx + 1
      }
    }
  }
  out <- do.call(rbind, rows)
  utils::write.csv(out, gzfile(file.path(run_dir, paste0(run_id, "_random_effects.csv.gz"))), row.names = FALSE, na = "")
  aggregate(
    random_effect_estimate_cm ~ model_id + grouping_factor + random_effect_term,
    data = out,
    FUN = function(z) paste(
      c(
        n = length(z),
        mean = round(mean(z), 6),
        sd = round(stats::sd(z), 6),
        q05 = round(stats::quantile(z, 0.05), 6),
        q95 = round(stats::quantile(z, 0.95), 6)
      ),
      collapse = ";"
    )
  )
}

model_stability <- function(formula_text, data, run_id, stability_mode) {
  if (stability_mode == "none") return(list(detailed = data.frame(), summary = data.frame()))
  groups <- c("focal_id", "session_id")
  tasks <- data.frame(grouping_factor = character(), level = character(), stringsAsFactors = FALSE)
  for (group in groups) {
    levels <- sort(unique(as.character(data[[group]])))
    if (stability_mode == "smoke") levels <- head(levels, 2)
    tasks <- rbind(tasks, data.frame(grouping_factor = group, level = levels, stringsAsFactors = FALSE))
  }
  stability_cache_dir <- file.path(cache_dir, run_id, "stability_refits")
  dir.create(stability_cache_dir, recursive = TRUE, showWarnings = FALSE)
  safe_name <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)
  rows <- list()
  for (i in seq_len(nrow(tasks))) {
    group <- tasks$grouping_factor[[i]]
    level <- tasks$level[[i]]
    task_cache_path <- file.path(stability_cache_dir, paste0(safe_name(group), "__", safe_name(level), ".rds"))
    if (file.exists(task_cache_path)) {
      message("Reusing cached stability refit for ", run_id, ": drop ", group, "=", level)
      rows[[length(rows) + 1]] <- readRDS(task_cache_path)
      next
    }
    message("Stability refit started for ", run_id, ": drop ", group, "=", level, " at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
    subset <- data[as.character(data[[group]]) != level, , drop = FALSE]
    subset$focal_id <- droplevels(factor(subset$focal_id))
    subset$session_id <- droplevels(factor(subset$session_id))
    subset$match_set_id <- droplevels(factor(subset$match_set_id))
    fit <- fit_lmer_capture(formula_text, subset, reml = FALSE, use_lmer_test = FALSE)
    task_rows <- list()
    if (inherits(fit$fit, "error")) {
      task_rows[[length(task_rows) + 1]] <- data.frame(
        model_id = run_id,
        grouping_factor = group,
        omitted_level = level,
        term = NA_character_,
        estimate = NA_real_,
        fit_status = "failed",
        warning_notes = conditionMessage(fit$fit),
        stringsAsFactors = FALSE
      )
    } else {
      fixed <- lme4::fixef(fit$fit)
      for (term in names(fixed)) {
        task_rows[[length(task_rows) + 1]] <- data.frame(
          model_id = run_id,
          grouping_factor = group,
          omitted_level = level,
          term = term,
          estimate = unname(fixed[[term]]),
          fit_status = fit_status(fit$fit),
          warning_notes = paste(fit$warnings, collapse = " | "),
          stringsAsFactors = FALSE
        )
      }
    }
    task_result <- do.call(rbind, task_rows)
    saveRDS(task_result, task_cache_path)
    message("Stability refit finished for ", run_id, ": drop ", group, "=", level, " at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
    rows[[length(rows) + 1]] <- task_result
  }
  detailed <- if (length(rows)) do.call(rbind, rows) else data.frame()
  if (nrow(detailed) == 0 || all(is.na(detailed$term))) {
    summary <- data.frame()
  } else {
    ok <- detailed[!is.na(detailed$term), , drop = FALSE]
    summary <- aggregate(
      estimate ~ model_id + term,
      data = ok,
      FUN = function(z) paste(c(min = min(z), max = max(z)), collapse = ";")
    )
    split_values <- do.call(rbind, strsplit(summary$estimate, ";", fixed = TRUE))
    summary$stability_min <- as.numeric(sub("^min=", "", split_values[, 1]))
    summary$stability_max <- as.numeric(sub("^max=", "", split_values[, 2]))
    summary$estimate <- NULL
  }
  list(detailed = detailed, summary = summary)
}

write_stability_plot <- function(stability_summary, fixed_effects, run_id) {
  if (nrow(stability_summary) == 0 || nrow(fixed_effects) == 0) return(NA_character_)
  coefs <- fixed_effects[, c("term", "Estimate")]
  merged <- merge(coefs, stability_summary, by = "term", all.x = FALSE)
  if (nrow(merged) == 0) return(NA_character_)
  path <- file.path(figures_dir, paste0(run_id, "_fixed_effect_stability.pdf"))
  display_order <- c(
    "hum presence",
    "starting focal-to-centroid distance",
    "pre-occurrence mean focal speed",
    "recent focal-to-centroid distance change",
    "pre-occurrence spread",
    "recent spread change",
    "Intercept"
  )
  merged$term_label <- fixed_effect_display_label(merged$term)
  merged$term_label <- factor(merged$term_label, levels = rev(display_order))
  p <- ggplot2::ggplot(merged, ggplot2::aes(x = Estimate, y = term_label)) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dotted", linewidth = 0.4, color = "grey35") +
    ggplot2::geom_errorbar(
      ggplot2::aes(xmin = stability_min, xmax = stability_max),
      orientation = "y",
      width = 0.18,
      linewidth = 0.45
    ) +
    ggplot2::geom_point(shape = 18, size = 2.4) +
    ggplot2::labs(
      title = "H1 fixed-effect stability",
      subtitle = "Response: focal-to-centroid distance change",
      x = "Estimate (cm change in focal-to-centroid distance)",
      y = NULL
    ) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
  ggplot2::ggsave(path, p, width = 8, height = 6, bg = "white")
  path
}

profile_ci_cached <- function(fit, run_id, run_dir) {
  profile_dir <- file.path(run_dir, "profile_ci")
  dir.create(profile_dir, recursive = TRUE, showWarnings = FALSE)
  profile_path <- file.path(profile_dir, "profile.rds")
  ci_path <- file.path(profile_dir, "profile_ci.csv")
  status_path <- file.path(profile_dir, "profile_status.rds")
  if (!isTRUE(cfg$profile_ci)) {
    return(list(table = data.frame(), status = data.frame()))
  }
  if (inherits(fit, "error")) {
    return(list(table = data.frame(), status = data.frame(
      model_id = run_id,
      profile_status = "skipped_fit_failed",
      profile_runtime_seconds = NA_real_,
      profile_parallel = cfg$profile_parallel,
      profile_ncpus = cfg$profile_ncpus,
      profile_maxpts = cfg$profile_maxpts,
      profile_devmatchtol = cfg$profile_devmatchtol,
      profile_path = profile_path,
      ci_path = ci_path,
      stringsAsFactors = FALSE
    )))
  }
  if (file.exists(status_path) && file.exists(ci_path)) {
    return(readRDS(status_path))
  }
  started <- proc.time()[[3]]
  message("Profile CI started for ", run_id, " at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  prof <- tryCatch(
    stats::profile(
      fit,
      parallel = cfg$profile_parallel,
      ncpus = cfg$profile_ncpus,
      maxpts = cfg$profile_maxpts,
      devmatchtol = cfg$profile_devmatchtol
    ),
    error = function(e) e
  )
  runtime <- proc.time()[[3]] - started
  if (inherits(prof, "error")) {
    out <- list(
      table = data.frame(),
      status = data.frame(
        model_id = run_id,
        profile_status = "failed",
        profile_runtime_seconds = runtime,
        profile_parallel = cfg$profile_parallel,
        profile_ncpus = cfg$profile_ncpus,
        profile_maxpts = cfg$profile_maxpts,
        profile_devmatchtol = cfg$profile_devmatchtol,
        profile_path = profile_path,
        ci_path = ci_path,
        profile_error = conditionMessage(prof),
        stringsAsFactors = FALSE
      )
    )
    saveRDS(out, status_path)
    message("Profile CI failed for ", run_id, ": ", conditionMessage(prof))
    return(out)
  }
  saveRDS(prof, profile_path)
  ci <- tryCatch(stats::confint(prof), error = function(e) e)
  if (inherits(ci, "error")) {
    out <- list(
      table = data.frame(),
      status = data.frame(
        model_id = run_id,
        profile_status = "profile_ok_ci_failed",
        profile_runtime_seconds = runtime,
        profile_parallel = cfg$profile_parallel,
        profile_ncpus = cfg$profile_ncpus,
        profile_maxpts = cfg$profile_maxpts,
        profile_devmatchtol = cfg$profile_devmatchtol,
        profile_path = profile_path,
        ci_path = ci_path,
        profile_error = conditionMessage(ci),
        stringsAsFactors = FALSE
      )
    )
    saveRDS(out, status_path)
    message("Profile CI confint failed for ", run_id, ": ", conditionMessage(ci))
    return(out)
  }
  ci_table <- as.data.frame(ci)
  ci_table$term <- rownames(ci_table)
  rownames(ci_table) <- NULL
  names(ci_table)[seq_len(2)] <- c("profile_ci_low", "profile_ci_high")
  ci_table$model_id <- run_id
  ci_table$profile_parallel <- cfg$profile_parallel
  ci_table$profile_ncpus <- cfg$profile_ncpus
  ci_table$profile_maxpts <- cfg$profile_maxpts
  ci_table$profile_devmatchtol <- cfg$profile_devmatchtol
  ci_table <- ci_table[, c("model_id", "term", "profile_ci_low", "profile_ci_high", "profile_parallel", "profile_ncpus", "profile_maxpts", "profile_devmatchtol")]
  utils::write.csv(ci_table, ci_path, row.names = FALSE, na = "")
  out <- list(
    table = ci_table,
    status = data.frame(
      model_id = run_id,
      profile_status = "complete",
      profile_runtime_seconds = runtime,
      profile_parallel = cfg$profile_parallel,
      profile_ncpus = cfg$profile_ncpus,
      profile_maxpts = cfg$profile_maxpts,
      profile_devmatchtol = cfg$profile_devmatchtol,
      profile_path = profile_path,
      ci_path = ci_path,
      stringsAsFactors = FALSE
    )
  )
  saveRDS(out, status_path)
  message("Profile CI finished for ", run_id, " at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  out
}

fit_checkpoint_audit <- function(run_id, run_dir) {
  files <- data.frame(
    model_id = run_id,
    checkpoint_step = c(
      "full_ml_fit", "full_ml_optimizer_trace", "full_ml_optimizer_progress",
      "null_ml_fit", "null_ml_optimizer_trace", "null_ml_optimizer_progress",
      "full_reml_fit", "full_reml_optimizer_trace", "full_reml_optimizer_progress",
      "profile_ci_status", "profile_ci_profile", "profile_ci_table"
    ),
    path = file.path(
      run_dir,
      c(
        "full_ml_fit.rds", "full_ml_fit_optimizer_trace.rds", "full_ml_fit_optimizer_trace_progress.csv",
        "null_ml_fit.rds", "null_ml_fit_optimizer_trace.rds", "null_ml_fit_optimizer_trace_progress.csv",
        "full_reml_fit.rds", "full_reml_fit_optimizer_trace.rds", "full_reml_fit_optimizer_trace_progress.csv",
        "profile_ci/profile_status.rds", "profile_ci/profile.rds", "profile_ci/profile_ci.csv"
      )
    ),
    stringsAsFactors = FALSE
  )
  files$exists <- file.exists(files$path)
  files$size_bytes <- ifelse(files$exists, file.info(files$path)$size, NA_real_)
  files$modified_time <- ifelse(files$exists, as.character(file.info(files$path)$mtime), "")
  files
}

aic_candidate_specs <- function(model_surface, transformed_predictors) {
  rows <- list()
  rows[[length(rows) + 1]] <- list(
    candidate_effect = "intended_full_model",
    candidate_role = "full",
    fixed_rhs = model_surface$full_rhs,
    omit_random_terms = character()
  )
  for (term in c("condition_hum", transformed_predictors)) {
    if (term == "condition_hum") {
      effect <- "condition_hum"
      random_omit <- "condition_hum_c"
    } else {
      label_match <- names(model_surface$transformed_by_label)[match(term, unlist(model_surface$transformed_by_label))]
      raw_match <- raw_predictor_specs$raw[match(label_match, raw_predictor_specs$label)]
      effect <- if (!is.na(raw_match)) raw_match else term
      random_omit <- term
    }
    rows[[length(rows) + 1]] <- list(
      candidate_effect = effect,
      candidate_role = "drop_main_effect",
      fixed_rhs = setdiff(c("condition_hum", transformed_predictors), term),
      omit_random_terms = random_omit
    )
  }
  rows
}

fit_predictor_aic_scores <- function(
  data,
  transformed_predictors,
  model_surface,
  decisions,
  include_match_set,
  run_id,
  run_dir,
  window_s,
  policy,
  match_policy,
  full_ml
) {
  if (!isTRUE(cfg$run_aic_predictor_scores)) return(data.frame())
  if ("threshold_m" %in% names(data) && any(is.finite(suppressWarnings(as.numeric(data$threshold_m))))) return(data.frame())
  aic_dir <- file.path(run_dir, "predictor_aic_scores")
  dir.create(aic_dir, recursive = TRUE, showWarnings = FALSE)
  rows <- list()
  specs <- aic_candidate_specs(model_surface, transformed_predictors)
  for (spec in specs) {
    random_formula <- random_formula_from_decisions(
      decisions,
      include_match_set,
      omit_random_terms = spec$omit_random_terms
    )
    fixed_rhs <- spec$fixed_rhs
    fixed_formula <- if (length(fixed_rhs)) {
      paste("Delta_D_cm ~", paste(fixed_rhs, collapse = " + "))
    } else {
      "Delta_D_cm ~ 1"
    }
    formula_text <- paste(fixed_formula, random_formula, sep = " + ")
    safe_effect <- gsub("[^A-Za-z0-9_.-]+", "_", spec$candidate_effect)
    cache_path <- file.path(aic_dir, paste0(safe_effect, ".rds"))
    fit <- if (spec$candidate_role == "full" && !inherits(full_ml$fit, "error")) {
      full_ml
    } else {
      fit_lmer_cached(
        cache_path,
        paste0(run_id, " predictor AIC ", spec$candidate_effect),
        formula_text,
        data,
        reml = FALSE,
        use_lmer_test = FALSE
      )
    }
    if (inherits(fit$fit, "error")) {
      rows[[length(rows) + 1]] <- data.frame(
        model_id = run_id,
        fixed_effect_variant = cfg$fixed_effect_variant,
        window_s = window_s,
        matching_policy = policy,
        match_set_random_effect = match_policy,
        candidate_effect = spec$candidate_effect,
        candidate_role = spec$candidate_role,
        n_rows = nrow(data),
        fit_status = "failed",
        log_likelihood = NA_real_,
        n_parameters_AIC = NA_real_,
        AIC = NA_real_,
        BIC = NA_real_,
        delta_AIC = NA_real_,
        akaike_weight = NA_real_,
        fixed_formula = fixed_formula,
        random_formula = random_formula,
        full_formula = formula_text,
        omitted_random_terms = paste(spec$omit_random_terms, collapse = " + "),
        warning_notes = conditionMessage(fit$fit),
        stringsAsFactors = FALSE
      )
    } else {
      rows[[length(rows) + 1]] <- data.frame(
        model_id = run_id,
        fixed_effect_variant = cfg$fixed_effect_variant,
        window_s = window_s,
        matching_policy = policy,
        match_set_random_effect = match_policy,
        candidate_effect = spec$candidate_effect,
        candidate_role = spec$candidate_role,
        n_rows = nrow(data),
        fit_status = fit_status(fit$fit),
        log_likelihood = as.numeric(stats::logLik(fit$fit)),
        n_parameters_AIC = attr(stats::logLik(fit$fit), "df"),
        AIC = stats::AIC(fit$fit),
        BIC = stats::BIC(fit$fit),
        delta_AIC = NA_real_,
        akaike_weight = NA_real_,
        fixed_formula = fixed_formula,
        random_formula = random_formula,
        full_formula = formula_text,
        omitted_random_terms = paste(spec$omit_random_terms, collapse = " + "),
        warning_notes = paste(fit$warnings, collapse = " | "),
        stringsAsFactors = FALSE
      )
    }
  }
  out <- do.call(rbind, rows)
  ok <- is.finite(out$AIC)
  if (any(ok)) {
    out$delta_AIC[ok] <- out$AIC[ok] - min(out$AIC[ok])
    weights <- exp(-0.5 * out$delta_AIC[ok])
    out$akaike_weight[ok] <- weights / sum(weights)
  }
  out[order(out$window_s, out$AIC, out$candidate_effect, na.last = TRUE), , drop = FALSE]
}

run_one_model <- function(
  view_data,
  window_s,
  policy,
  include_match_set,
  mode,
  max_rows,
  stability_mode,
  run_id_override = "",
  run_context = list()
) {
  match_policy <- if (include_match_set) "match_set_model" else "standard_model"
  run_id <- if (nzchar(run_id_override)) run_id_override else paste0("h1_", window_s, "s")
  run_dir <- file.path(cache_dir, run_id)
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  complete_flag <- file.path(run_dir, "COMPLETE")
  result_path <- file.path(run_dir, "result.rds")
  if (file.exists(complete_flag) && file.exists(result_path)) {
    result <- add_run_context(readRDS(result_path), run_context)
    saveRDS(result, result_path)
    return(result)
  }

  prepared <- prepare_run_data(
    view_data,
    window_s = window_s,
    policy = policy,
    smoke = mode == "smoke",
    max_rows = max_rows
  )
  data <- prepared$data
  if (nrow(prepared$transform_table)) {
    prepared$transform_table$fixed_effect_variant <- cfg$fixed_effect_variant
  }
  transformed_predictors <- prepared$transformed_predictors
  model_surface <- build_model_surface(data, transformed_predictors, prepared$transformed_by_label)
  data <- model_surface$data
  full_fixed_formula <- paste("Delta_D_cm ~", paste(model_surface$full_rhs, collapse = " + "))
  null_fixed_formula <- paste("Delta_D_cm ~", paste(model_surface$null_rhs, collapse = " + "))
  distribution_plot <- write_distribution_plots(data, transformed_predictors, run_id)
  fe_re_status <- safe_fe_re_summary(data, full_fixed_formula, run_id)
  decisions <- random_slope_decisions(data, model_surface$slope_specs)
  random_formula <- random_formula_from_decisions(decisions, include_match_set)
  full_formula <- paste(full_fixed_formula, random_formula, sep = " + ")
  null_formula <- paste(null_fixed_formula, random_formula, sep = " + ")

  if (mode == "smoke") {
    full_validation <- validate_lmer_formula(full_formula, data)
    null_validation <- validate_lmer_formula(null_formula, data)
    n_hum <- sum(data$condition_hum == 1)
    n_control <- sum(data$condition_hum == 0)
    validation_ok <- isTRUE(full_validation$ok) && isTRUE(null_validation$ok)

    vif <- compute_vif(data, model_surface$vif_terms)
    vif$model_id <- run_id
    vif$fixed_effect_variant <- cfg$fixed_effect_variant
    vif$window_s <- window_s
    vif$matching_policy <- policy
    vif$match_set_random_effect <- match_policy

    run_manifest <- data.frame(
      model_id = run_id,
      predictor_set = cfg$predictor_set,
      fixed_effect_variant = cfg$fixed_effect_variant,
      optimizer = cfg$optimizer,
      window_s = window_s,
      matching_policy = policy,
      match_set_random_effect = match_policy,
      n_rows = nrow(data),
      n_hum_rows = n_hum,
      n_control_rows = n_control,
      n_focal_ids = length(unique(data$focal_id)),
      n_sessions = length(unique(data$session_id)),
      n_match_sets = length(unique(data$match_set_id)),
      dropped_incomplete_rows = prepared$dropped_incomplete_rows,
      response = "Delta_D_cm",
      response_unit = "centimeters",
      model_family = "linear mixed-effects model",
      model_link = "identity",
      full_formula = full_formula,
      null_formula = null_formula,
      random_formula = random_formula,
      fe_re_tab_status = fe_re_status,
      fit_status = if (validation_ok) "smoke_formula_validated" else "smoke_formula_failed",
      converged = NA,
      singular = NA,
      max_gradient_or_equivalent = NA_real_,
      ML_runtime_seconds = full_validation$runtime_seconds + null_validation$runtime_seconds,
      REML_runtime_seconds = NA_real_,
      warning_notes = paste(unique(c(full_validation$warnings, null_validation$warnings)), collapse = " | "),
      diagnostic_plot_residual = "",
      diagnostic_plot_random_effect_hist = "",
      diagnostic_plot_random_effect_qq = "",
      distribution_plot = distribution_plot,
      stability_plot = NA_character_,
      formula_validation_error = paste(c(full_validation$error, null_validation$error), collapse = " | "),
      stringsAsFactors = FALSE
    )

    result <- add_run_context(list(
      run_manifest = run_manifest,
      transform_table = prepared$transform_table,
      random_slope_decisions = cbind(model_id = run_id, fixed_effect_variant = cfg$fixed_effect_variant, window_s = window_s, matching_policy = policy, match_set_random_effect = match_policy, decisions),
      full_null_test = data.frame(),
      fixed_effects = data.frame(),
      predictor_aic_scores = data.frame(),
      vif = vif,
      varcorr = data.frame(),
      residual_summary = data.frame(),
      random_effect_summary = data.frame(),
      stability_detailed = data.frame(),
      stability_summary = data.frame(),
      profile_ci = data.frame(),
      profile_status = data.frame(),
      checkpoint_audit = fit_checkpoint_audit(run_id, run_dir)
    ), run_context)
    saveRDS(result, result_path)
    file.create(complete_flag)
    return(result)
  }

  full_ml <- fit_lmer_cached(
    file.path(run_dir, "full_ml_fit.rds"),
    paste0(run_id, " full ML"),
    full_formula,
    data,
    reml = FALSE,
    use_lmer_test = FALSE
  )
  null_ml <- fit_lmer_cached(
    file.path(run_dir, "null_ml_fit.rds"),
    paste0(run_id, " null ML"),
    null_formula,
    data,
    reml = FALSE,
    use_lmer_test = FALSE
  )
  full_reml <- fit_lmer_cached(
    file.path(run_dir, "full_reml_fit.rds"),
    paste0(run_id, " full REML lmerTest"),
    full_formula,
    data,
    reml = TRUE,
    use_lmer_test = TRUE
  )

  n_hum <- sum(data$condition_hum == 1)
  n_control <- sum(data$condition_hum == 0)
  model_status <- fit_status(full_ml$fit)
  lrt <- data.frame()
  if (!inherits(full_ml$fit, "error") && !inherits(null_ml$fit, "error")) {
    lrt_raw <- as.data.frame(stats::anova(null_ml$fit, full_ml$fit, test = "Chisq"))
    lrt <- data.frame(
      model_id = run_id,
      fixed_effect_variant = cfg$fixed_effect_variant,
      window_s = window_s,
      matching_policy = policy,
      match_set_random_effect = match_policy,
      null_df = lrt_raw$Df[[2]],
      chisq = lrt_raw$Chisq[[2]],
      p_value = lrt_raw$`Pr(>Chisq)`[[2]],
      test = "full-null likelihood-ratio test for condition_hum; ML fits",
      stringsAsFactors = FALSE
    )
  }

  fixed <- fixed_effect_table(
    full_reml$fit,
    run_id,
    window_s,
    policy,
    match_policy,
    full_formula,
    nrow(data),
    n_hum,
    n_control
  )
  if (nrow(fixed)) fixed$fixed_effect_variant <- cfg$fixed_effect_variant
  vif <- compute_vif(data, model_surface$vif_terms)
  vif$model_id <- run_id
  vif$fixed_effect_variant <- cfg$fixed_effect_variant
  vif$window_s <- window_s
  vif$matching_policy <- policy
  vif$match_set_random_effect <- match_policy
  vc <- varcorr_table(full_ml$fit, run_id)
  residuals <- residual_outputs(full_ml$fit, data, run_id)
  diagnostic_plots <- if (mode == "smoke") character() else write_model_diagnostic_plots(full_ml$fit, run_id)
  diagnostic_residual_plot <- if ("residual" %in% names(diagnostic_plots)) unname(diagnostic_plots[["residual"]]) else ""
  diagnostic_random_effect_hist_plot <- if ("random_effect_hist" %in% names(diagnostic_plots)) unname(diagnostic_plots[["random_effect_hist"]]) else ""
  diagnostic_random_effect_qq_plot <- if ("random_effect_qq" %in% names(diagnostic_plots)) unname(diagnostic_plots[["random_effect_qq"]]) else ""
  random_effect_summary <- if (mode == "smoke") data.frame() else write_random_effects(full_ml$fit, data, run_id, run_dir)
  stability <- model_stability(full_formula, data, run_id, stability_mode)
  stability_plot <- write_stability_plot(stability$summary, fixed, run_id)
  profile_ci <- profile_ci_cached(full_reml$fit, run_id, run_dir)
  predictor_aic_scores <- fit_predictor_aic_scores(
    data,
    transformed_predictors,
    model_surface,
    decisions,
    include_match_set,
    run_id,
    run_dir,
    window_s,
    policy,
    match_policy,
    full_ml
  )

  if (nrow(residuals$residuals)) {
    utils::write.csv(
      residuals$residuals,
      gzfile(file.path(run_dir, paste0(run_id, "_residuals.csv.gz"))),
      row.names = FALSE,
      na = ""
    )
  }

  run_manifest <- data.frame(
    model_id = run_id,
    predictor_set = cfg$predictor_set,
    fixed_effect_variant = cfg$fixed_effect_variant,
    optimizer = cfg$optimizer,
    window_s = window_s,
    matching_policy = policy,
    match_set_random_effect = match_policy,
    n_rows = nrow(data),
    n_hum_rows = n_hum,
    n_control_rows = n_control,
    n_focal_ids = length(unique(data$focal_id)),
    n_sessions = length(unique(data$session_id)),
    n_match_sets = length(unique(data$match_set_id)),
    dropped_incomplete_rows = prepared$dropped_incomplete_rows,
    response = "Delta_D_cm",
    response_unit = "centimeters",
    model_family = "linear mixed-effects model",
    model_link = "identity",
    full_formula = full_formula,
    null_formula = null_formula,
    random_formula = random_formula,
    fe_re_tab_status = fe_re_status,
    fit_status = model_status,
    converged = if (!inherits(full_ml$fit, "error")) model_status %in% c("acceptable", "acceptable_boundary_simplified") else FALSE,
    singular = if (!inherits(full_ml$fit, "error")) lme4::isSingular(full_ml$fit, tol = 1e-4) else NA,
    max_gradient_or_equivalent = if (!inherits(full_ml$fit, "error")) max_gradient(full_ml$fit) else NA_real_,
    ML_runtime_seconds = full_ml$runtime_seconds,
    REML_runtime_seconds = full_reml$runtime_seconds,
    warning_notes = paste(unique(c(full_ml$warnings, null_ml$warnings, full_reml$warnings)), collapse = " | "),
    diagnostic_plot_residual = diagnostic_residual_plot,
    diagnostic_plot_random_effect_hist = diagnostic_random_effect_hist_plot,
    diagnostic_plot_random_effect_qq = diagnostic_random_effect_qq_plot,
    distribution_plot = distribution_plot,
    stability_plot = stability_plot,
    profile_ci_status = if (nrow(profile_ci$status)) profile_ci$status$profile_status[[1]] else "not_requested",
    stringsAsFactors = FALSE
  )
  if (!inherits(full_ml$fit, "error")) {
    run_manifest$log_likelihood <- as.numeric(stats::logLik(full_ml$fit))
    run_manifest$n_parameters_AIC <- attr(stats::logLik(full_ml$fit), "df")
    run_manifest$AIC <- stats::AIC(full_ml$fit)
    run_manifest$BIC <- stats::BIC(full_ml$fit)
    run_manifest$residual_sd_cm <- stats::sigma(full_ml$fit)
    run_manifest$observations_per_estimated_parameter <- nrow(data) / (length(lme4::fixef(full_ml$fit)) + nrow(as.data.frame(lme4::VarCorr(full_ml$fit))))
  } else {
    run_manifest$failure_reason <- conditionMessage(full_ml$fit)
  }

  result <- add_run_context(list(
    run_manifest = run_manifest,
    transform_table = prepared$transform_table,
    random_slope_decisions = cbind(model_id = run_id, fixed_effect_variant = cfg$fixed_effect_variant, window_s = window_s, matching_policy = policy, match_set_random_effect = match_policy, decisions),
    full_null_test = lrt,
    fixed_effects = fixed,
    predictor_aic_scores = predictor_aic_scores,
    vif = vif,
    varcorr = vc,
    residual_summary = residuals$summary,
    random_effect_summary = random_effect_summary,
    stability_detailed = stability$detailed,
    stability_summary = stability$summary,
    profile_ci = profile_ci$table,
    profile_status = profile_ci$status,
    checkpoint_audit = fit_checkpoint_audit(run_id, run_dir)
  ), run_context)
  saveRDS(result, result_path)
  file.create(complete_flag)
  result
}

mode <- cfg$mode
stability_mode <- cfg$stability
if (mode == "smoke" && stability_mode == "design") {
  stability_mode <- "none"
}
max_rows <- cfg$max_rows_per_run
if (mode == "smoke" && (!is.finite(max_rows) || is.na(max_rows))) {
  max_rows <- 4000L
}

all_run_configs <- load_run_configs()

selected_configs <- all_run_configs
if (nzchar(cfg$only_run)) {
  requested <- strsplit(cfg$only_run, ",", fixed = TRUE)[[1]]
  selected_configs <- selected_configs[selected_configs$model_id %in% requested, , drop = FALSE]
  missing <- setdiff(requested, selected_configs$model_id)
  if (length(missing)) {
    stop(paste("unknown --only-run model_id(s):", paste(missing, collapse = ", ")))
  }
}

read_cached_results <- function() {
  paths <- list.files(cache_dir, pattern = "^result\\.rds$", recursive = TRUE, full.names = TRUE)
  if (!length(paths)) return(list())
  lapply(sort(paths), readRDS)
}

results <- list()
if (!cfg$aggregate_only) {
  views <- list()
  if (!nzchar(cfg$run_config_csv)) {
    views <- list(standard_matching = read_view("standard_matching"))
  }
  for (i in seq_len(nrow(selected_configs))) {
    row <- selected_configs[i, ]
    policy <- row$matching_policy
    window_s <- row$window_s
    include_match_set <- as_bool(row$include_match_set)
    run_id <- row$model_id
    run_context <- run_context_from_config(row)
    view_data <- if (nzchar(cfg$run_config_csv)) read_configured_data(row$data_path) else views[[policy]]
    message("Running ", run_id, " policy=", policy, " W", window_s, " matchSetModel=", include_match_set)
    results[[length(results) + 1]] <- run_one_model(
      view_data,
      window_s = window_s,
      policy = policy,
      include_match_set = include_match_set,
      mode = mode,
      max_rows = max_rows,
      stability_mode = stability_mode,
      run_id_override = run_id,
      run_context = run_context
    )
  }
}

if (cfg$skip_aggregate) {
  message("Skipping aggregate tables/report by request.")
  quit(save = "no", status = 0)
}

if (cfg$aggregate_only) {
  results <- read_cached_results()
}

bind <- function(name) {
  frames <- lapply(results, `[[`, name)
  frames <- frames[vapply(frames, function(x) is.data.frame(x) && nrow(x) > 0, logical(1))]
  if (length(frames)) do.call(rbind, frames) else data.frame()
}

run_manifest <- bind("run_manifest")
transform_table <- bind("transform_table")
random_slope_decisions_table <- bind("random_slope_decisions")
full_null_tests <- bind("full_null_test")
fixed_effects <- bind("fixed_effects")
predictor_aic_scores <- bind("predictor_aic_scores")
vif_table <- bind("vif")
varcorr <- bind("varcorr")
residual_summary <- bind("residual_summary")
random_effect_summary <- bind("random_effect_summary")
stability_detailed <- bind("stability_detailed")
stability_summary <- bind("stability_summary")
profile_ci_table <- bind("profile_ci")
profile_status_table <- bind("profile_status")
checkpoint_audit_table <- bind("checkpoint_audit")

write.csv(run_manifest, file.path(tables_dir, "rq1_lmm_run_manifest.csv"), row.names = FALSE, na = "")
write.csv(transform_table, file.path(tables_dir, "rq1_lmm_transform_decisions.csv"), row.names = FALSE, na = "")
write.csv(random_slope_decisions_table, file.path(tables_dir, "rq1_lmm_random_slope_decisions.csv"), row.names = FALSE, na = "")
write.csv(full_null_tests, file.path(tables_dir, "rq1_lmm_full_null_tests.csv"), row.names = FALSE, na = "")
write.csv(fixed_effects, file.path(tables_dir, "rq1_lmm_fixed_effects_lmerTest_REML.csv"), row.names = FALSE, na = "")
write.csv(predictor_aic_scores, file.path(tables_dir, "rq1_lmm_predictor_aic_scores.csv"), row.names = FALSE, na = "")
write.csv(vif_table, file.path(tables_dir, "rq1_lmm_vif.csv"), row.names = FALSE, na = "")
write.csv(varcorr, file.path(tables_dir, "rq1_lmm_random_effect_variance.csv"), row.names = FALSE, na = "")
write.csv(residual_summary, file.path(tables_dir, "rq1_lmm_residual_summary.csv"), row.names = FALSE, na = "")
write.csv(random_effect_summary, file.path(tables_dir, "rq1_lmm_random_effect_summary.csv"), row.names = FALSE, na = "")
write.csv(stability_detailed, file.path(tables_dir, "rq1_lmm_model_stability_detailed.csv"), row.names = FALSE, na = "")
write.csv(stability_summary, file.path(tables_dir, "rq1_lmm_model_stability_summary.csv"), row.names = FALSE, na = "")
write.csv(profile_ci_table, file.path(tables_dir, "rq1_lmm_profile_ci.csv"), row.names = FALSE, na = "")
write.csv(profile_status_table, file.path(tables_dir, "rq1_lmm_profile_status.csv"), row.names = FALSE, na = "")
write.csv(checkpoint_audit_table, file.path(tables_dir, "rq1_lmm_fit_checkpoint_audit.csv"), row.names = FALSE, na = "")

primary_id <- cfg$primary_model_id
if (!nzchar(primary_id)) {
  if ("threshold_role" %in% names(run_manifest) && "window_s" %in% names(run_manifest)) {
    primary_candidates <- run_manifest$model_id[
      is.finite(suppressWarnings(as.numeric(run_manifest$threshold_m))) & run_manifest$window_s == 60
    ]
    primary_id <- if (length(primary_candidates)) primary_candidates[[1]] else "h1_60s"
  } else {
    primary_id <- "h1_60s"
  }
}
primary_fixed <- fixed_effects[fixed_effects$model_id == primary_id & fixed_effects$term == "condition_hum", , drop = FALSE]
primary_lrt <- full_null_tests[full_null_tests$model_id == primary_id, , drop = FALSE]
threshold_summary_line <- if ("threshold_m" %in% names(run_manifest)) {
  threshold_values <- suppressWarnings(as.numeric(run_manifest$threshold_m))
  thresholds <- sort(unique(threshold_values[is.finite(threshold_values)]))
  if (length(thresholds)) {
    paste0("- Threshold supplementary run: one frozen global threshold, `", paste(thresholds, collapse = ", "), "` m.")
  } else {
    "- Threshold supplementary run: `FALSE`."
  }
} else {
  "- Threshold supplementary run: `FALSE`."
}

report_path <- file.path(reports_dir, "rq1_scientific_lmm_guide_report.md")
report <- c(
  "# RQ1 Scientific-Guide LMM Revamp",
  "",
  paste0("- Run mode: `", mode, "`"),
  paste0("- Predictor set: `", cfg$predictor_set, "`"),
  paste0("- Fixed-effect variant: `", cfg$fixed_effect_variant, "`"),
  paste0("- Optimizer: `", cfg$optimizer, "`"),
  paste0("- Optimizer checkpoint interval: every `", cfg$optimizer_checkpoint_every, "` deviance evaluations"),
  paste0("- Profile CI requested: `", cfg$profile_ci, "`"),
  paste0("- Data directory: `", data_dir, "`"),
  paste0("- Output directory: `", output_dir, "`"),
  paste0("- Run config CSV: `", ifelse(nzchar(cfg$run_config_csv), resolve_repo_path(cfg$run_config_csv), "built-in canonical run set"), "`"),
  paste0("- Primary model ID: `", primary_id, "`"),
  threshold_summary_line,
  "- Biological question: whether hum windows are followed by smaller post-event increases, or larger decreases, in focal distance to the centroid of the other observed collared alpacas than matched no-hum controls.",
  "- Response used for fitting: `Delta_D_cm = Delta_D_w * 100`; the model is still an identity-link Gaussian LMM, now expressed in centimeters for numerical scaling.",
  "- Canonical event policy: `None` and `alpaca_hum` event labels are retained as alpaca hums; `rumble` is excluded upstream.",
  "- Random-effect base: `focal_id` and `session_id`; `match_set_id` is not fitted. No focal-session combined random-slope grouping is used.",
  "",
  "## Model Surface",
  "",
  "The canonical scientific launch fits three pre-specified H1 additive LMMs: 30 s, 60 s, and 120 s. The expected primary biological timescale is 60 s; 30 s and 120 s are window-length checks.",
  "",
  "## Dataset Sizes",
  "",
  if (nrow(run_manifest)) paste(capture.output(print(run_manifest[, c("model_id", "predictor_set", "optimizer", "n_rows", "n_hum_rows", "n_control_rows", "n_focal_ids", "n_sessions", "n_match_sets", "fit_status", "singular")], row.names = FALSE)), collapse = "\n") else "No run manifest rows.",
  "",
  "## Transformations",
  "",
  "Predictor histograms are written as per-run distribution PDFs before transformation. The revamp uses the same deterministic transform policy in every window: starting distance and pre-window speed are log1p-transformed then z-scored; recent distance change, pre-event polygon area, and recent polygon-area change are raw-scale z-scored. Pre-event polygon area is not log transformed under any window.",
  "",
  if (nrow(transform_table)) paste(capture.output(print(unique(transform_table[, c("predictor", "transform", "decision_reason")]), row.names = FALSE)), collapse = "\n") else "No transform rows.",
  "",
  "## Random Slopes",
  "",
  "Random slopes were determined using the guide's within-group variation rules and the companion `fe.re.tab` helper output. Binary condition uses the factor rule; quantitative covariates use the three-unique-values rule and the factor-style fallback rule. A slope is included when the rule is met for at least half of the levels of the grouping factor.",
  "",
  if (nrow(random_slope_decisions_table)) paste(capture.output(print(random_slope_decisions_table[, c("model_id", "grouping_factor", "fixed_term", "include_random_slope", "decision_rule")], row.names = FALSE)), collapse = "\n") else "No random-slope decision rows.",
  "",
  "## Full-Null Tests",
  "",
  "The full-null tests compare the full fixed-effect model with a null model that omits the hum-condition effect while keeping the same random-effects structure, using ML fits and a likelihood-ratio test.",
  "",
  if (nrow(full_null_tests)) paste(capture.output(print(full_null_tests, row.names = FALSE)), collapse = "\n") else "No full-null test rows.",
  "",
  "## Predictor AIC Scores",
  "",
  "Primary non-threshold runs optionally fit the intended full ML model and fixed-effect drop-one variants on the same rows. These AIC scores are diagnostics only and do not select the reported model.",
  "",
  if (nrow(predictor_aic_scores)) paste(capture.output(print(predictor_aic_scores[, c("model_id", "candidate_effect", "candidate_role", "AIC", "delta_AIC", "akaike_weight", "fit_status")], row.names = FALSE)), collapse = "\n") else "Predictor AIC scores were not requested for this run.",
  "",
  "## Primary Configuration",
  "",
  if (nrow(primary_fixed)) paste0(
    "- Primary condition estimate: ", round(primary_fixed$Estimate[[1]], 4), " cm (",
    round(primary_fixed$Estimate[[1]] / 100, 4), " m), 95% CI [",
    round(primary_fixed$ci95_low_cm[[1]], 4), ", ", round(primary_fixed$ci95_high_cm[[1]], 4), "] cm; p = ",
    signif(primary_fixed$p_value[[1]], 4), " by lmerTest/Satterthwaite REML coefficient test."
  ) else "- Primary condition coefficient unavailable.",
  if (nrow(primary_lrt)) paste0(
    "- Primary full-null LRT: Chisq = ", round(primary_lrt$chisq[[1]], 4),
    ", df = ", primary_lrt$null_df[[1]], ", p = ", signif(primary_lrt$p_value[[1]], 4), "."
  ) else "- Primary full-null LRT unavailable.",
  "",
  "## Diagnostics",
  "",
  "- Convergence, singularity, gradient, AIC/BIC, and formulas are in `tables/rq1_lmm_run_manifest.csv`.",
  "- Random-effect standard deviations/correlations and boundary flags are in `tables/rq1_lmm_random_effect_variance.csv`.",
  "- VIF values are in `tables/rq1_lmm_vif.csv`.",
  "- Residual summaries are in `tables/rq1_lmm_residual_summary.csv`; full row-level residuals are stored per run under `cache/<model_id>/*_residuals.csv.gz`.",
  "- Random-effect summaries are in `tables/rq1_lmm_random_effect_summary.csv`; full random-effect tables are stored per run under `cache/<model_id>/*_random_effects.csv.gz`.",
  "- Model stability excludes levels of `focal_id` and `session_id` one at a time. `match_set_id` is not used for leave-one-level stability because it has tens of thousands of matched sets and is only an optional nuisance blocking intercept in this revamp.",
  "- Step-level fit caches and optimizer traces are audited in `tables/rq1_lmm_fit_checkpoint_audit.csv`.",
  "- Optional profile CI outputs are in `tables/rq1_lmm_profile_ci.csv` and `tables/rq1_lmm_profile_status.csv`; profile CI is off unless explicitly requested.",
  "- Per-run residual, random effect, distribution, and stability plots are in `figures/`.",
  "",
  "## Interpretation Boundary",
  "",
  "A negative `condition_hum` coefficient means hum windows had lower post-event change in centroid distance than matched no-hum controls. This supports only an observed-collared-subnetwork association; it does not identify a caller and does not make a full-herd or causal claim."
)
writeLines(report, con = report_path)

message("Wrote report: ", report_path)
