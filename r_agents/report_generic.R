#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
})

read_input <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) stop("No JSON argument provided")
  fromJSON(args[[1]], simplifyVector = FALSE)
}

summarize_eda <- function(eda) {
  if (is.null(eda)) {
    return("Non ho informazioni di EDA disponibili.\n")
  }
  n_rows <- eda$n_rows %||% NA
  n_cols <- eda$n_cols %||% NA

  txt <- sprintf(
    "Il dataset contiene %s righe e %s colonne.\n",
    as.character(n_rows),
    as.character(n_cols)
  )

  if (!is.null(eda$missing_perc)) {
    high_na <- Filter(
      function(x) !is.null(x$value) && !is.na(x$value) && x$value > 20,
      lapply(
        names(eda$missing_perc),
        function(name) list(name = name, value = eda$missing_perc[[name]])
      )
    )
    if (length(high_na) > 0) {
      txt <- paste0(
        txt,
        "Le colonne con più di 20% di valori mancanti includono: ",
        paste0(
          vapply(high_na, function(h) sprintf("%s (%.1f%%)", h$name, h$value), ""),
          collapse = ", "
        ),
        ".\n"
      )
    } else {
      txt <- paste0(
        txt,
        "Non risultano colonne con più di 20% di valori mancanti.\n"
      )
    }
  }

  txt
}

summarize_modeling <- function(modeling) {
  if (is.null(modeling)) {
    return("Non ho risultati di modellazione disponibili.\n")
  }
  mt <- modeling$model_type %||% "sconosciuto"
  n_obs <- modeling$n_obs %||% NA
  train_size <- modeling$train_size %||% NA
  test_size <- modeling$test_size %||% NA

  lines <- c()
  lines <- c(
    lines,
    sprintf(
      "Ho addestrato un modello di tipo %s su %s osservazioni (train=%s, test=%s).",
      mt, as.character(n_obs), as.character(train_size), as.character(test_size)
    )
  )

  if (!is.null(modeling$accuracy)) {
    lines <- c(
      lines,
      sprintf("Accuratezza sul test: %.3f.", modeling$accuracy)
    )
  }
  if (!is.null(modeling$rmse)) {
    lines <- c(
      lines,
      sprintf("RMSE sul test: %.3f.", modeling$rmse)
    )
  }
  if (!is.null(modeling$mae)) {
    lines <- c(
      lines,
      sprintf("MAE sul test: %.3f.", modeling$mae)
    )
  }
  if (!is.null(modeling$r2)) {
    lines <- c(
      lines,
      sprintf("R² sul test: %.3f.", modeling$r2)
    )
  }

  paste0(paste(lines, collapse = " "), "\n")
}

main <- function() {
  result <- list(ok = FALSE, error = NULL, report = NULL)

  input <- tryCatch(
    read_input(),
    error = function(e) {
      result$error <<- paste("input_error:", e$message)
      NULL
    }
  )
  if (is.null(input)) {
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }

  eda <- input$eda_result
  modeling <- input$modeling_result

  report_text <- ""
  report_text <- paste0(report_text, "=== Report sintetico analisi dati ===\n\n")
  report_text <- paste0(report_text, "1) Analisi esplorativa (EDA)\n")
  report_text <- paste0(report_text, summarize_eda(eda), "\n")

  report_text <- paste0(report_text, "2) Modellazione\n")
  report_text <- paste0(report_text, summarize_modeling(modeling), "\n")

  report_text <- paste0(
    report_text,
    "Nota: questo è un report sintetico generato automaticamente "
  )

  result$ok <- TRUE
  result$report <- list(
    text = report_text
  )
  result$error <- NULL

  cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
}

`%||%` <- function(x, y) if (is.null(x)) y else x

main()
