#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  # jsonlite è l'unico hard-dependency
  library(jsonlite)
})

# Operatore %||% locale (come in rlang):
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (is.character(x) && identical(x, ""))) {
    y
  } else {
    x
  }
}

args <- commandArgs(trailingOnly = TRUE)

# Di default, job è una lista vuota; se l'argomento non è JSON valido,
# non andiamo in errore, ma logghiamo un warning su stderr.
job <- list()
if (length(args) >= 1) {
  job <- tryCatch(
    fromJSON(args[[1]]),
    error = function(e) {
      message("[conversation_logger.R] Impossibile parsare JSON: ", conditionMessage(e))
      list()
    }
  )
}

conversation_id <- job$conversation_id %||% "unknown_conv"
user_id        <- job$user_id        %||% "unknown_user"
messages       <- job$messages       %||% list()
db_path        <- job$db_path        %||% NA_character_
rdata_dir      <- job$rdata_dir      %||% NA_character_

# ---------------------------
# 1) Log minimale su stdout (debug / piping futuro)
# ---------------------------
summary_obj <- list(
  ok              = TRUE,
  conversation_id = conversation_id,
  user_id         = user_id,
  n_messages      = length(messages)
)

# Stampiamo un JSON minimale su stdout.
cat(toJSON(summary_obj, auto_unbox = TRUE), "\n")

# ---------------------------
# 2) Salvataggio opzionale su SQLite (se DBI + RSQLite presenti)
# ---------------------------
if (!is.na(db_path)) {
  has_dbi      <- requireNamespace("DBI", quietly = TRUE)
  has_rsqlite  <- requireNamespace("RSQLite", quietly = TRUE)

  if (has_dbi && has_rsqlite) {
    con <- NULL
    try({
      con <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)

      # Tabella conversazioni
      DBI::dbExecute(
        con,
        "
        CREATE TABLE IF NOT EXISTS conversations (
          id           TEXT PRIMARY KEY,
          user_id      TEXT,
          created_at   TEXT
        )
        "
      )

      # Tabella messaggi
      DBI::dbExecute(
        con,
        "
        CREATE TABLE IF NOT EXISTS conversation_messages (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          conversation_id TEXT,
          role            TEXT,
          content         TEXT,
          ts              TEXT
        )
        "
      )

      # Inserisci (o ignora) la conversazione
      DBI::dbExecute(
        con,
        "
        INSERT OR IGNORE INTO conversations (id, user_id, created_at)
        VALUES (?, ?, datetime('now'))
        ",
        params = list(conversation_id, user_id)
      )

      # Inserisci i messaggi
      if (length(messages) > 0L) {
        for (m in messages) {
          role    <- m$role    %||% NA_character_
          content <- m$content %||% NA_character_
          ts      <- m$timestamp %||% NA_character_

          DBI::dbExecute(
            con,
            "
            INSERT INTO conversation_messages (conversation_id, role, content, ts)
            VALUES (?, ?, ?, ?)
            ",
            params = list(conversation_id, role, content, ts)
          )
        }
      }
    }, silent = TRUE)

    if (!is.null(con)) {
      try(DBI::dbDisconnect(con), silent = TRUE)
    }
  } else {
    message(
      "[conversation_logger.R] DBI/RSQLite non disponibili, salto logging su SQLite.\n",
      "  - has_dbi     = ", has_dbi, "\n",
      "  - has_rsqlite = ", has_rsqlite, "\n"
    )
  }
}

# ---------------------------
# 3) Salvataggio RDS opzionale in rdata_dir
# ---------------------------
if (!is.na(rdata_dir)) {
  dir.create(rdata_dir, showWarnings = FALSE, recursive = TRUE)
  rds_path <- file.path(rdata_dir, paste0("conversation_", conversation_id, ".rds"))
  try(
    saveRDS(job, file = rds_path),
    silent = TRUE
  )
}

quit(status = 0L)
