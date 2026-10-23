#!/usr/bin/env Rscript

source(file.path("research", "hum_acoustic_profile", "R", "hum_acoustic_profile.R"))

args <- commandArgs(trailingOnly = TRUE)
config <- hum_profile_default_config()

for (i in seq_along(args)) {
  arg <- args[[i]]
  if (arg == "--bundle-dir") {
    config$bundle_dir <- args[[i + 1]]
  }
  if (arg == "--output-dir") {
    config$output_dir <- args[[i + 1]]
  }
  if (arg == "--sample-size") {
    config$sample_size <- args[[i + 1]]
  }
}

run_hum_acoustic_profile(config)

