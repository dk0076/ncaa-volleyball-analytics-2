library(xgboost)
library(dplyr)
library(pROC)

# Matches seed in 05_model.R — CV round counts are comparable across scripts,
# though validation models are trained on the train subset only (not models.rds).
# Reported AUC characterizes the architecture, not the deployed model specifically.
set.seed(42)

serves_featured  <- readRDS("data/serves_featured.rds")
all_contests_df  <- readRDS("data/big_west_contests.rds")
model_artifacts  <- readRDS("data/models.rds")


# Chronological contest order
contest_order <- all_contests_df %>%
  mutate(date = as.Date(date, "%m/%d/%Y")) %>%
  arrange(date) %>%
  pull(contest) %>%
  unique()
contest_order <- contest_order[contest_order %in% unique(serves_featured$contestid)]

# Last 20 matches as holdout test set
test_contests  <- tail(contest_order, 20)
train_contests <- setdiff(contest_order, test_contests)

train <- serves_featured %>% filter(contestid %in% train_contests)
test  <- serves_featured %>% filter(contestid %in% test_contests)
cat("Train:", nrow(train), "| Test:", nrow(test), "\n")

features_base <- model_artifacts$features_base
features_fbk  <- model_artifacts$features_fbk

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
params <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  nthread          = 1  # single-threaded training is bit-reproducible given the seed
)

logloss <- function(actual, predicted, eps = 1e-15) {
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
}

tune_and_validate <- function(X_train, y_train, X_test, y_test, label) {
  cv <- xgb.cv(
    params                = params,
    data                  = xgb.DMatrix(X_train, label = y_train),
    nrounds               = 500,
    nfold                 = 5,
    early_stopping_rounds = 20,
    verbose               = 0
  )
  best_rounds <- if (length(cv$best_iteration) > 0) {
    cv$best_iteration
  } else {
    which.min(cv$evaluation_log$test_logloss_mean)
  }

  model <- xgb.train(
    params  = params,
    data    = xgb.DMatrix(X_train, label = y_train),
    nrounds = best_rounds,
    verbose = 0
  )

  preds       <- predict(model, xgb.DMatrix(X_test))
  baseline_ll <- logloss(y_test, rep(mean(y_train), length(y_test)))
  model_ll    <- logloss(y_test, preds)
  auc_val     <- round(as.numeric(pROC::auc(pROC::roc(y_test, preds, quiet = TRUE))), 4)

  # SHAP importance: mean(|SHAP|) per feature on test set.
  # More reliable than gain for high-cardinality features like receiver_id.
  shap_matrix <- predict(model, xgb.DMatrix(X_test), predcontrib = TRUE)
  # predcontrib appends a bias column as the last column (name varies by xgboost version)
  stopifnot("SHAP matrix has unexpected column count" =
              ncol(shap_matrix) == ncol(X_test) + 1)
  shap_matrix <- shap_matrix[, -ncol(shap_matrix), drop = FALSE]  # drop bias column
  shap_imp    <- sort(colMeans(abs(shap_matrix)), decreasing = TRUE)

  cat("\n--- Validation:", label, "---\n")
  cat("Baseline logloss:", round(baseline_ll, 4), "\n")
  cat("Model logloss:   ", round(model_ll,    4), "\n")
  cat("Improvement:     ", round(baseline_ll - model_ll, 4), "\n")
  cat("AUC:             ", auc_val, "\n")
  cat("SHAP importance (mean |SHAP| on test set):\n")
  print(round(shap_imp, 5))

  list(baseline_ll = round(baseline_ll, 4),
       model_ll    = round(model_ll, 4),
       improvement = round(baseline_ll - model_ll, 4),
       auc         = auc_val)
}

# ── M1: P(Ace) ────────────────────────────────────────────────────────────────
cat("Tuning M1 (Ace)...\n")
m1_val <- tune_and_validate(
  X_train = as.matrix(train[, features_base]),
  y_train = train$ace,
  X_test  = as.matrix(test[, features_base]),
  y_test  = test$ace,
  label   = "P(Ace) — M1"
)

# ── M2: P(Service Error) ─────────────────────────────────────────────────────
cat("Tuning M2 (Service Error)...\n")
m2_val <- tune_and_validate(
  X_train = as.matrix(train[, features_base]),
  y_train = train$service_error,
  X_test  = as.matrix(test[, features_base]),
  y_test  = test$service_error,
  label   = "P(Service Error) — M2"
)

# ── M3: P(FBK | In Play) ─────────────────────────────────────────────────────
cat("Tuning M3 (FBK)...\n")
train_ip <- train %>% filter(in_play == 1)
test_ip  <- test  %>% filter(in_play == 1)

m3_val <- tune_and_validate(
  X_train = as.matrix(train_ip[, features_fbk]),
  y_train = train_ip$fbk_against,
  X_test  = as.matrix(test_ip[, features_fbk]),
  y_test  = test_ip$fbk_against,
  label   = "P(FBK Against | In Play) — M3"
)

saveRDS(
  list(
    # M3 AUC kept as `auc` for backwards compatibility with scouting report footnote
    auc              = m3_val$auc,
    auc_m1           = m1_val$auc,
    auc_m2           = m2_val$auc,
    auc_m3           = m3_val$auc,
    baseline_logloss = m3_val$baseline_ll,
    model_logloss    = m3_val$model_ll,
    improvement      = m3_val$improvement,
    n_train          = nrow(train),
    n_test           = nrow(test)
  ),
  "data/validation_metrics.rds"
)
cat("Validation metrics saved to data/validation_metrics.rds\n")

# Record the exact R and package versions used — committed to the repo so
# results can be tied to a specific software environment.
writeLines(capture.output(sessionInfo()), "session_info.txt")
cat("Session info written to session_info.txt\n")
