# Syntax-check every R script, the Shiny app, and all knitr chunks in the Rmd
# reports. This does not execute the pipeline — the dataset is not in the repo —
# so it verifies that the committed source is parseable, nothing more.
root <- normalizePath(".", winslash = "/")
r_files <- list.files(root, pattern = "[.][Rr]$", recursive = TRUE, full.names = TRUE)
r_files <- r_files[!grepl("/(renv|packrat|[.]Rproj[.]user|[.]claude)/", r_files)]
rmd_files <- list.files(root, pattern = "[.][Rr]md$", recursive = TRUE, full.names = TRUE)
rmd_files <- rmd_files[!grepl("/(renv|packrat|[.]Rproj[.]user|[.]claude)/", rmd_files)]

fail <- character()
for (f in r_files) {
  msg <- tryCatch({ parse(f); NULL }, error = function(e) conditionMessage(e))
  rel <- sub(paste0(root, "/"), "", f, fixed = TRUE)
  if (is.null(msg)) cat("ok    ", rel, "\n") else { cat("FAIL  ", rel, "\n       ", msg, "\n"); fail <- c(fail, rel) }
}
for (f in rmd_files) {
  rel <- sub(paste0(root, "/"), "", f, fixed = TRUE)
  msg <- tryCatch({ parse(knitr::purl(f, output = tempfile(), quiet = TRUE)); NULL },
                  error = function(e) conditionMessage(e))
  if (is.null(msg)) cat("ok    ", rel, "(chunks)\n") else { cat("FAIL  ", rel, "\n       ", msg, "\n"); fail <- c(fail, rel) }
}
cat("\nChecked", length(r_files), "R files and", length(rmd_files), "Rmd files.\n")
if (length(fail)) { cat("Failed:", paste(fail, collapse = ", "), "\n"); quit(status = 1) }
cat("All parsed cleanly.\n")
