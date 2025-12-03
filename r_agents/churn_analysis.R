#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
})

# -------------------------------------------------------------------
# Operatore %||% (come negli altri script)
# -------------------------------------------------------------------
`%||%` <- function(x, y) {
  if (is.null(x)) {
    y
  } else {
    x
  }
}

# -------------------------------------------------------------------
# Lettura job JSON
# -------------------------------------------------------------------
read_job <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) {
    return(list(params = list()))
  }

  tryCatch(
    fromJSON(args[1], simplifyVector = FALSE),
    error = function(e) {
      # fallback: job vuoto
      list(params = list())
    }
  )
}

# -------------------------------------------------------------------
# Simulazione dataset churn
# -------------------------------------------------------------------
simulate_churn_data <- function(n = 1000L, seed = 123L) {
  set.seed(seed)

  recency   <- rexp(n, rate = 1 / 30)
  frequency <- rpois(n, lambda = 3)
  amount    <- rgamma(n, shape = 2, scale = 50)

  eta <- -2 + 0.01 * recency - 0.2 * frequency + 0.001 * amount
  p   <- 1 / (1 + exp(-eta))
  churn <- rbinom(n, size = 1, prob = p)

  data.frame(
    recency   = recency,
    frequency = frequency,
    amount    = amount,
    churn     = factor(churn)
  )
}

# -------------------------------------------------------------------
# Fit modello logistico e summary compatto
# -------------------------------------------------------------------
fit_churn_model <- function(df) {
  fit <- suppressWarnings(
    glm(churn ~ recency + frequency + amount, data = df, family = binomial())
  )
  s <- summary(fit)

  coef_df <- as.data.frame(coef(s))
  coef_df$term <- rownames(coef_df)

  list(
    churn_rate = mean(df$churn == "1"),
    coefficients = lapply(seq_len(nrow(coef_df)), function(i) {
      list(
        term      = coef_df$term[i],
        estimate  = unname(coef_df$Estimate[i]),
        std_error = unname(coef_df$`Std. Error`[i]),
        z_value   = unname(coef_df$`z value`[i]),
        p_value   = unname(coef_df$`Pr(>|z|)`[i])
      )
    })
  )
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
main <- function() {
  result <- list(
    ok           = FALSE,
    error        = NULL,
    n            = NULL,
    churn_rate   = NULL,
    coefficients = NULL
  )

  job <- read_job()

  # parametro opzionale: numero di righe
  n <- job$params$n %||% 1000L
  n <- as.integer(n)
  if (is.na(n) || n <= 0L) {
    n <- 1000L
  }

  # parametro opzionale: seed (per riproducibilità cross-run)
  seed <- job$params$seed %||% 123L
  seed <- as.integer(seed %||% 123L)

  df <- simulate_churn_data(n = n, seed = seed)

  model_out <- tryCatch(
    fit_churn_model(df),
    error = function(e) {
      result$error <<- paste("model_error:", e$message)
      NULL
    }
  )

  if (is.null(model_out)) {
    cat(toJSON(result, auto_unbox = TRUE, null = "null"))
    quit(status = 0L)
  }

  result$ok           <- TRUE
  result$error        <- NULL
  result$n            <- n
  result$churn_rate   <- model_out$churn_rate
  result$coefficients <- model_out$coefficients

  cat(toJSON(result, auto_unbox = TRUE, null = "null"))
}

main()

