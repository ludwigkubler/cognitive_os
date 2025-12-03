#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  suppressWarnings({
    if (requireNamespace("RSQLite", quietly = TRUE)) {
      library(RSQLite)
    }
  })
})

read_input <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) stop("No JSON argument provided")
  fromJSON(args[[1]], simplifyVector = FALSE)
}

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

train_test_split <- function(n, test_ratio = 0.3) {
  set.seed(1234)
  idx <- seq_len(n)
  test_n <- max(1L, round(n * test_ratio))
  test_idx <- sample(idx, size = test_n)
  train_idx <- setdiff(idx, test_idx)
  list(train = train_idx, test = test_idx)
}

model_classification_glm <- function(df, target) {
  df <- df[!is.na(df[[target]]), , drop = FALSE]
  df[[target]] <- as.factor(df[[target]])
  if (nrow(df) < 10L) stop("Troppe poche righe dopo filtraggio NA per classificazione")

  split <- train_test_split(nrow(df))
  train <- df[split$train, , drop = FALSE]
  test  <- df[split$test,  , drop = FALSE]

  form <- as.formula(paste(target, "~ ."))
  model <- glm(form, data = train, family = binomial())

  # se il target ha più di 2 livelli, glm binomial non è ideale, ma teniamo le 2 principali
  probs <- suppressWarnings(predict(model, newdata = test, type = "response"))
  # threshold 0.5 per binario
  levels_target <- levels(train[[target]])
  if (length(levels_target) == 2L) {
    pred_class <- ifelse(probs >= 0.5, levels_target[2], levels_target[1])
  } else {
    pred_class <- factor(levels(train[[target]])[1], levels = levels_target)
  }

  true <- test[[target]]
  acc <- mean(pred_class == true)

  list(
    model_type = "glm_binomial",
    n_obs = nrow(df),
    train_size = nrow(train),
    test_size = nrow(test),
    accuracy = acc,
    target_levels = as.list(levels_target),
    coefficients = as.list(coef(model))
  )
}

model_regression_lm <- function(df, target) {
  df <- df[!is.na(df[[target]]), , drop = FALSE]
  if (nrow(df) < 10L) stop("Troppe poche righe dopo filtraggio NA per regressione")

  split <- train_test_split(nrow(df))
  train <- df[split$train, , drop = FALSE]
  test  <- df[split$test,  , drop = FALSE]

  form <- as.formula(paste(target, "~ ."))
  model <- lm(form, data = train)

  preds <- predict(model, newdata = test)
  y_true <- test[[target]]

  rmse <- sqrt(mean((preds - y_true)^2))
  mae  <- mean(abs(preds - y_true))
  r2   <- summary(model)$r.squared

  list(
    model_type = "lm",
    n_obs = nrow(df),
    train_size = nrow(train),
    test_size = nrow(test),
    rmse = rmse,
    mae = mae,
    r2 = r2,
    coefficients = as.list(coef(model))
  )
}

main <- function() {
  result <- list(ok = FALSE, error = NULL, modeling = NULL)

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

  if (is.null(input$dataset_ref)) {
    result$error <- "campo dataset_ref mancante"
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }

  target <- input$target
  if (is.null(target)) {
    result$error <- "campo target mancante"
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }
  target <- as.character(target)

  problem_type <- tolower(as.character(input$problem_type %||% "classification"))

  df <- tryCatch(
    load_dataset(input$dataset_ref),
    error = function(e) {
      result$error <<- paste("load_error:", e$message)
      NULL
    }
  )
  if (is.null(df)) {
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }

  if (!target %in% names(df)) {
    result$error <- paste("target", target, "non presente nel dataset")
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }

  modeling <- tryCatch(
    {
      if (problem_type == "regression") {
        model_regression_lm(df, target)
      } else {
        model_classification_glm(df, target)
      }
    },
    error = function(e) {
      result$error <<- paste("model_error:", e$message)
      NULL
    }
  )

  if (is.null(modeling)) {
    cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
    quit(status = 0)
  }

  result$ok <- TRUE
  result$modeling <- modeling
  result$error <- NULL

  cat(toJSON(result, auto_unbox = TRUE, null = "null", digits = 4))
}

# operatore %||% per default
`%||%` <- function(x, y) if (is.null(x)) y else x

main()
