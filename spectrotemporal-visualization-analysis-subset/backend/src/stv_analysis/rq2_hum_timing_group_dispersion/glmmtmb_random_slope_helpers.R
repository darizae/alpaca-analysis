RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION <- "rq2_session_maximal_random_slopes_v1"
RQ2_INTERACTION_DATA_COLUMN <- "z_area_x_z_spread_change"

rq2_random_effects_for_surface <- function(surface) {
  if (surface == "interaction") {
    return("(1 + z_area * z_spread_change + z_speed + z_centroid_change | session_id)")
  }
  if (surface == "additive") {
    return("(1 + z_area + z_spread_change + z_speed + z_centroid_change | session_id)")
  }
  stop(paste("unsupported fixed surface:", surface))
}

rq2_fixed_rhs_for_surface <- function(surface) {
  if (surface == "interaction") {
    return("z_area * z_spread_change + z_speed + z_centroid_change")
  }
  if (surface == "additive") {
    return("z_area + z_spread_change + z_speed + z_centroid_change")
  }
  stop(paste("unsupported fixed surface:", surface))
}

rq2_formula_for_surface <- function(surface, response = "hum_count") {
  paste(
    paste(response, "~", rq2_fixed_rhs_for_surface(surface)),
    "+ offset(log(window_length_minutes)) +",
    rq2_random_effects_for_surface(surface)
  )
}

rq2_safe_name <- function(value) gsub("[^A-Za-z0-9_]+", "_", value)

rq2_atomic_save_rds <- function(object, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  saveRDS(object, tmp)
  if (file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) {
    stop(paste("failed to atomically move temporary RDS into place:", path))
  }
  invisible(path)
}

rq2_cache_valid <- function(cached, spec, nsim = NULL, seed = NULL) {
  if (!is.list(cached)) return(FALSE)
  expected_formula <- if ("formula_text" %in% names(spec)) spec$formula_text[[1]] else spec$formula[[1]]
  expected_family <- if ("family_label" %in% names(spec)) spec$family_label[[1]] else spec$family[[1]]
  expected_random <- spec$random_effects[[1]]
  if (!identical(cached$model_id, spec$model_id[[1]])) return(FALSE)
  if (!identical(cached$formula_text, expected_formula)) return(FALSE)
  if (!identical(cached$family, expected_family)) return(FALSE)
  if (!identical(cached$random_effects, expected_random)) return(FALSE)
  if (!identical(cached$random_effects_structure_version, RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION)) return(FALSE)
  if (!is.null(nsim) && !identical(as.integer(cached$nsim), as.integer(nsim))) return(FALSE)
  if (!is.null(seed) && !identical(as.integer(cached$seed), as.integer(seed))) return(FALSE)
  TRUE
}

rq2_fit_cache_path <- function(objects_dir, prefix, model_id) {
  file.path(
    objects_dir,
    paste0(prefix, "_fit_", RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION, "_", rq2_safe_name(model_id), ".rds")
  )
}

rq2_dharma_cache_path <- function(objects_dir, prefix, model_id) {
  file.path(
    objects_dir,
    paste0(prefix, "_dharma_outputs_", RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION, "_", rq2_safe_name(model_id), ".rds")
  )
}

rq2_write_progress_row <- function(path, row) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  row <- as.data.frame(row, stringsAsFactors = FALSE)
  utils::write.table(
    row,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(path),
    append = file.exists(path)
  )
  invisible(path)
}

rq2_deterministic_smoke_sample <- function(data, max_rows) {
  if (!is.finite(max_rows) || is.na(max_rows) || nrow(data) <= max_rows) return(data)
  data$rank_in_session <- ave(seq_len(nrow(data)), data$session_id, FUN = seq_along)
  data <- data[
    order(data$rank_in_session, data$session_id, data$window_start_seconds_local, data$rq2_window_id),
    ,
    drop = FALSE
  ]
  data <- data[seq_len(max_rows), , drop = FALSE]
  data$rank_in_session <- NULL
  data[order(data$session_id, data$window_start_seconds_local, data$rq2_window_id), , drop = FALSE]
}

rq2_add_interaction_columns <- function(data) {
  if (all(c("z_area", "z_spread_change") %in% names(data))) {
    data[[RQ2_INTERACTION_DATA_COLUMN]] <- data$z_area * data$z_spread_change
  }
  data
}

rq2_random_slope_specs <- function(surface) {
  specs <- data.frame(
    fixed_term = c("z_area", "z_spread_change", "z_speed", "z_centroid_change"),
    random_slope_term = c("z_area", "z_spread_change", "z_speed", "z_centroid_change"),
    data_column = c("z_area", "z_spread_change", "z_speed", "z_centroid_change"),
    predictor_type = rep("covariate_rule", 4L),
    component_terms = rep("", 4L),
    stringsAsFactors = FALSE
  )
  if (surface == "interaction") {
    specs <- rbind(
      specs,
      data.frame(
        fixed_term = "z_area:z_spread_change",
        random_slope_term = "z_area:z_spread_change",
        data_column = RQ2_INTERACTION_DATA_COLUMN,
        predictor_type = "covariate_interaction_rule",
        component_terms = "z_area,z_spread_change",
        stringsAsFactors = FALSE
      )
    )
  }
  specs
}

rq2_random_slope_decisions <- function(data, spec, grouping_factor = "session_id") {
  slope_specs <- rq2_random_slope_specs(spec$fixed_surface[[1]])
  rows <- list()
  for (i in seq_len(nrow(slope_specs))) {
    data_column <- slope_specs$data_column[[i]]
    component_terms <- slope_specs$component_terms[[i]]
    if (!data_column %in% names(data)) {
      stop(paste("missing random-slope data column:", data_column))
    }
    if (nzchar(component_terms)) {
      components <- strsplit(component_terms, ",", fixed = TRUE)[[1]]
      ok_rule_1 <- tapply(
        seq_len(nrow(data)),
        data[[grouping_factor]],
        function(idx) {
          all(vapply(
            components,
            function(component) {
              values <- data[[component]][idx]
              values <- values[is.finite(values)]
              length(unique(values)) >= 3L
            },
            logical(1)
          ))
        }
      )
      rule_2_counts <- tapply(
        seq_len(nrow(data)),
        data[[grouping_factor]],
        function(idx) {
          values <- data[[data_column]][idx]
          values <- values[is.finite(values)]
          if (!length(values)) return(0L)
          sum(table(values) >= 2L)
        }
      )
      rule <- "interaction: all component covariates have at least three unique values, or product has factor-style identifiability, in at least half of sessions"
    } else {
      ok_rule_1 <- tapply(
        data[[data_column]],
        data[[grouping_factor]],
        function(values) {
          values <- values[is.finite(values)]
          length(unique(values)) >= 3L
        }
      )
      rule_2_counts <- tapply(
        data[[data_column]],
        data[[grouping_factor]],
        function(values) {
          values <- values[is.finite(values)]
          if (!length(values)) return(0L)
          sum(table(values) >= 2L)
        }
      )
      rule <- "covariate: at least three unique values, or factor-style identifiability, in at least half of sessions"
    }
    ok_rule_2 <- rule_2_counts >= 2L
    include <- mean(ok_rule_1, na.rm = TRUE) >= 0.5 || mean(ok_rule_2, na.rm = TRUE) >= 0.5
    rows[[length(rows) + 1L]] <- data.frame(
      model_id = spec$model_id[[1]],
      family = if ("family_label" %in% names(spec)) spec$family_label[[1]] else spec$family[[1]],
      fixed_surface = spec$fixed_surface[[1]],
      grouping_factor = grouping_factor,
      fixed_term = slope_specs$fixed_term[[i]],
      random_slope_term = slope_specs$random_slope_term[[i]],
      random_slope_data_column = data_column,
      predictor_type = slope_specs$predictor_type[[i]],
      n_group_levels = length(unique(data[[grouping_factor]])),
      rule_1_levels_ok = sum(ok_rule_1, na.rm = TRUE),
      rule_2_levels_ok = sum(ok_rule_2, na.rm = TRUE),
      include_random_slope = include,
      implemented_in_formula = include,
      decision_rule = rule,
      random_effects = spec$random_effects[[1]],
      random_effects_structure_version = RQ2_RANDOM_EFFECTS_STRUCTURE_VERSION,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

rq2_run_context <- function() {
  explicit <- Sys.getenv("STV_ANALYSIS_RUN_CONTEXT", unset = "")
  if (nzchar(explicit)) return(explicit)
  if (nzchar(Sys.getenv("SLURM_JOB_ID", unset = ""))) return("hpc_slurm")
  "local"
}

rq2_machine_description <- function() {
  node <- Sys.info()[["nodename"]]
  if (is.na(node) || !nzchar(node)) node <- Sys.info()[["sysname"]]
  paste("R execution host", node)
}
