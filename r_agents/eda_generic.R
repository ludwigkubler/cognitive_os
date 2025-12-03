#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  suppressWarnings({
    if (requireNamespace("RSQLite", quietly = TRUE)) {
      library(RSQLite)
    }
  })
})

# -------------------------------------------------------------------
# Lettura input: JSON passato come unico argomento da riga di comando
# -------------------------------------------------------------------
read_job <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) {
    stop("Nessun JSON passato come argomento. Usa: Rscript eda_generic.R '<json>'")
  }
  txt <- args[1]
  fromJSON(txt, simplifyVector = FALSE)
}

# -------------------------------------------------------------------
# Caricamento dataset
# -------------------------------------------------------------------
load_dataset <- function(dataset_ref) {
  if (is.null(dataset_ref$type)) stop("dataset_ref$type mancante")
  type <- dataset_ref$type

  if (type == "csv") {
    path <- dataset_ref$path
    if (is.null(path)) stop("dataset_ref$path mancante per type='csv'")
    if (!file.exists(path)) stop(paste("File CSV inesistente:", path))
    df <- read.csv(path, stringsAsFactors = FALSE)
    return(df)
  }

  if (type == "sqlite_table") {
    if (!requireNamespace("RSQLite", quietly = TRUE)) {
      stop("Pacchetto RSQLite non disponibile per type='sqlite_table'")
    }
    path <- dataset_ref$path
    table <- dataset_ref$table
    if (is.null(path) || is.null(table)) {
      stop("dataset_ref$path o dataset_ref$table mancante per sqlite_table")
    }
    if (!file.exists(path)) stop(paste("DB inesistente:", path))
    con <- RSQLite::dbConnect(RSQLite::SQLite(), path)
    on.exit(RSQLite::dbDisconnect(con), add = TRUE)
    df <- RSQLite::dbReadTable(con, table)
    return(df)
  }

  stop(paste("dataset_ref$type non supportato:", type))
}

# -------------------------------------------------------------------
# EDA
# -------------------------------------------------------------------
compute_eda <- function(df) {
  n_rows <- nrow(df)
  n_cols <- ncol(df)
  col_types <- vapply(df, function(x) class(x)[1], character(1))

  if (n_rows > 0L) {
    miss_counts <- vapply(df, function(x) sum(is.na(x)), numeric(1))
    miss_perc <- round(miss_counts / n_rows * 100, 2)
  } else {
    miss_perc <- rep(NA_real_, n_cols)
  }
  names(miss_perc) <- names(df)

  num_cols <- vapply(df, is.numeric, logical(1))
  numeric_summary <- NULL
  corr_small <- NULL

  if (any(num_cols)) {
    num_df <- df[, num_cols, drop = FALSE]
    numeric_summary <- lapply(names(num_df), function(col) {
      x <- num_df[[col]]
      x <- x[!is.na(x)]
      if (!length(x)) {
        return(list())
      }
      list(
        mean   = mean(x),
        sd     = sd(x),
        min    = min(x),
        q25    = as.numeric(quantile(x, 0.25)),
        median = median(x),
        q75    = as.numeric(quantile(x, 0.75)),
        max    = max(x)
      )
    })
    names(numeric_summary) <- names(num_df)

    if (ncol(num_df) >= 2L) {
      cm <- tryCatch(
        cor(num_df, use = "pairwise.complete.obs"),
        error = function(e) NULL
      )
      if (!is.null(cm)) {
        k <- min(5L, nrow(cm))
        corr_small <- cm[seq_len(k), seq_len(k), drop = FALSE]
      }
    }
  }

  list(
    n_rows = n_rows,
    n_cols = n_cols,
    column_types = as.list(col_types),
    missing_perc = as.list(miss_perc),
    numeric_summary = numeric_summary,
    sample_head = head(df, 5),
    numeric_corr_head = corr_small
  )
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
main <- function() {
  result <- list(ok = FALSE, error = NULL, eda = NULL)

  job <- tryCatch(
    read_job(),
    error = function(e) {
      result$error <<- paste("input_error:", e$message)
      NULL
    }
  )
  if (is.null(job)) {
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }

  if (is.null(job$params$dataset_ref)) {
    result$error <- "campo params$dataset_ref mancante nel job"
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }

  df <- tryCatch(
    load_dataset(job$params$dataset_ref),
    error = function(e) {
      result$error <<- paste("load_error:", e$message)
      NULL
    }
  )
  if (is.null(df)) {
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }

  eda <- tryCatch(
    compute_eda(df),
    error = function(e) {
      result$error <<- paste("eda_error:", e$message)
      NULL
    }
  )
  if (is.null(eda)) {
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }

  result$ok <- TRUE
  result$eda <- eda
  result$error <- NULL

  cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
}

main()
