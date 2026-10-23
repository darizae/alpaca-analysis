repos <- "https://cloud.r-project.org"
packages <- c("tuneR", "jsonlite", "digest", "knitr")
missing <- packages[!packages %in% rownames(installed.packages())]

if (length(missing) > 0) {
  install.packages(missing, repos = repos)
}

installed <- packages %in% rownames(installed.packages())
if (!all(installed)) {
  stop(
    "Missing required R packages after installation attempt: ",
    paste(packages[!installed], collapse = ", "),
    call. = FALSE
  )
}

cat("Installed required packages: ", paste(packages, collapse = ", "), "\n", sep = "")
