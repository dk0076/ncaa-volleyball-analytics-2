library(xgboost)
library(dplyr)

set.seed(42)  # matches 06_validation.R so CV round counts are comparable

stopifnot("Run 04_prior_features.R before this script" =
            file.exists("data/serves_featured.rds"))

serves       <- readRDS("data/serves_featured.rds")
bwc_contests <- readRDS("data/big_west_contests.rds")

# Features for ace/error models (all serves — receiver may be NA for aces/errors).
# Integer ID columns removed: prior rate features are already target-encoded
# per-player/team rates computed chronologically in 04_prior_features.R.
# opp_prior_ace_rate / opp_prior_error_rate replace opp_team_id for M1/M2.
features_base <- c(
  "set_num", "score_diff", "is_home", "is_late_set",
  "prior_ace_rate", "prior_error_rate",
  "opp_prior_ace_rate", "opp_prior_error_rate",
  "match_ace_rate", "match_error_rate"
)

# Features for FBK model. M3 is TRAINED on in-play serves only (the conditional
# outcome is defined only when the serve is returned) but PREDICTS on every
# serve: receiver_prior_fbk_rate is imputed with the opponent team's prior FBK
# rate in 04_prior_features.R when the receiver is unobserved (aces / errors).
features_fbk <- c(
  features_base,
  "prior_fbk_rate",
  "receiver_prior_fbk_rate", "opp_prior_fbk_rate"
)

# ── Chronological fold assignment (5 contiguous date blocks) ─────────────────
contest_order <- bwc_contests %>%
  mutate(date_parsed = as.Date(date, "%m/%d/%Y")) %>%
  arrange(date_parsed) %>%
  pull(contest)
contest_order <- contest_order[contest_order %in% unique(serves$contestid)]

serves <- serves %>%
  mutate(
    contest_rank = match(as.character(contestid), as.character(contest_order)),
    fold         = ceiling(contest_rank / ceiling(length(contest_order) / 5))
  )

stopifnot("fold NAs — some contestids not in bwc_contests" = !any(is.na(serves$fold)))

serves$row_id <- seq_len(nrow(serves))
in_play <- serves %>% filter(in_play == 1)

X_all <- as.matrix(serves[,  features_base])
X_fbk <- as.matrix(in_play[, features_fbk])

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
# nthread = 1: multithreaded histogram construction is not bit-reproducible;
# single-threaded training makes results identical across runs given the seed.
params <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  nthread          = 1
)

# 5-fold CV with early stopping — round counts tuned once on full data and
# reused across the forward-chained blocks below. This is a deliberate,
# disclosed simplification: it leaks only the number of boosting rounds (a
# single scalar), not model weights, into earlier blocks.
tune_rounds <- function(X, y) {
  cv <- xgb.cv(
    params                = params,
    data                  = xgb.DMatrix(X, label = y),
    nrounds               = 500,
    nfold                 = 5,
    early_stopping_rounds = 20,
    verbose               = 0
  )
  if (length(cv$best_iteration) > 0) cv$best_iteration
  else which.min(cv$evaluation_log$test_logloss_mean)
}

cat("Tuning M1 (Ace)...\n")
nr1 <- tune_rounds(X_all, serves$ace)
cat("Tuning M2 (Error)...\n")
nr2 <- tune_rounds(X_all, serves$service_error)
cat("Tuning M3 (FBK)...\n")
nr3 <- tune_rounds(X_fbk, in_play$fbk_against)
cat("Best rounds — M1:", nr1, "| M2:", nr2, "| M3:", nr3, "\n")

# ── Forward-chained out-of-fold predictions ──────────────────────────────────
# Block k (k = 2..5) is predicted by models trained only on blocks 1..k-1 —
# strictly earlier matches. No model weight ever sees the future. Block 1 has
# no prior training data and therefore receives no OOF prediction; it is
# excluded from the leaderboard but still gets full-data predictions below for
# scouting use.
#
# M3 predicts on ALL serves in each block (not just in-play ones): serve
# quality is a pre-serve expectation, so the conditional FBK probability must
# be applied uniformly regardless of how the serve actually turned out.
# Conditioning the penalty on the realized outcome would leak that outcome
# into a metric that claims to be predictive.
serves$p_ace_oof   <- NA_real_
serves$p_error_oof <- NA_real_
serves$p_fbk_oof   <- NA_real_

cat("Computing forward-chained OOF predictions...\n")
for (k in 2:5) {
  cat("  Block", k, "(trained on blocks 1 to", k - 1, ")\n")
  tr <- serves$fold < k
  te <- serves$fold == k

  m1_k <- xgb.train(params = params, nrounds = nr1, verbose = 0,
                    data = xgb.DMatrix(as.matrix(serves[tr, features_base]),
                                       label = serves$ace[tr]))
  serves$p_ace_oof[te] <- predict(m1_k, xgb.DMatrix(as.matrix(serves[te, features_base])))

  m2_k <- xgb.train(params = params, nrounds = nr2, verbose = 0,
                    data = xgb.DMatrix(as.matrix(serves[tr, features_base]),
                                       label = serves$service_error[tr]))
  serves$p_error_oof[te] <- predict(m2_k, xgb.DMatrix(as.matrix(serves[te, features_base])))

  ip_tr <- serves$in_play == 1 & serves$fold < k
  m3_k  <- xgb.train(params = params, nrounds = nr3, verbose = 0,
                     data = xgb.DMatrix(as.matrix(serves[ip_tr, features_fbk]),
                                        label = serves$fbk_against[ip_tr]))
  serves$p_fbk_oof[te] <- predict(m3_k, xgb.DMatrix(as.matrix(serves[te, features_fbk])))
}

# ── OOF serve quality ────────────────────────────────────────────────────────
serves <- serves %>%
  mutate(
    # M1 and M2 are trained independently (binary:logistic), so their probabilities
    # are not constrained to sum to <= 1. pmax(0, ...) clamps the residual.
    p_in_play_oof = pmax(0, 1 - p_ace_oof - p_error_oof),
    # Serve quality: expected-outcome composite with unit weights, computed
    # identically for every scored serve from pre-serve features only.
    # Implicit assumption: ace (+1), error (-1), and FBK against (-1 conditional on
    # in-play) are treated as equal in magnitude. In practice an ace ends the rally
    # entirely, while FBK raises the opponent's rally-win probability rather than
    # guaranteeing a point — so this slightly overvalues aces relative to FBK
    # avoidance. A calibrated version would weight by empirical points-won probability
    # per outcome; this formula is the unit-weight baseline.
    serve_quality = p_ace_oof - p_error_oof - p_in_play_oof * p_fbk_oof
  )

scored <- !is.na(serves$serve_quality)
cat("Scored serves (blocks 2-5):", sum(scored),
    "| unscored (block 1):", sum(!scored), "\n")

n_clamped <- sum((1 - serves$p_ace_oof[scored] - serves$p_error_oof[scored]) < 0)
if (n_clamped > 0) warning(n_clamped, " serves had p_ace_oof + p_error_oof > 1 before clamping.")

# Global 0-100 quality index for interpretable display.
# Saved to models.rds so scouting report applies identical scaling to new predictions.
quality_min <- min(serves$serve_quality[scored])
quality_max <- max(serves$serve_quality[scored])
serves <- serves %>%
  mutate(quality_index = ifelse(
    is.na(serve_quality), NA_integer_,
    as.integer(round(100 * (serve_quality - quality_min) / (quality_max - quality_min)))
  ))

# ── Train final full-data models (for scouting report matchup predictions) ───
cat("Training final models on full data...\n")
m1 <- xgb.train(params = params, nrounds = nr1, verbose = 0,
                data = xgb.DMatrix(X_all, label = serves$ace))
m2 <- xgb.train(params = params, nrounds = nr2, verbose = 0,
                data = xgb.DMatrix(X_all, label = serves$service_error))
m3 <- xgb.train(params = params, nrounds = nr3, verbose = 0,
                data = xgb.DMatrix(X_fbk, label = in_play$fbk_against))

# Full-data predictions stored separately — used by scouting report for
# per-server probability display. M3 scores every serve here too.
serves$p_ace     <- predict(m1, xgb.DMatrix(X_all))
serves$p_error   <- predict(m2, xgb.DMatrix(X_all))
serves$p_in_play <- pmax(0, 1 - serves$p_ace - serves$p_error)
serves$p_fbk     <- predict(m3, xgb.DMatrix(as.matrix(serves[, features_fbk])))

# ── Leaderboard (forward-chained OOF quality_index — not in-sample) ──────────
# Block 1 serves have no OOF prediction and are excluded; n_serves counts
# scored serves only, so early-season-only players may drop below the cutoff.
leaderboard <- serves %>%
  filter(!is.na(quality_index)) %>%
  group_by(player, serve_team) %>%
  summarise(
    n_serves    = n(),
    avg_quality = round(mean(quality_index)),
    avg_p_ace   = round(mean(p_ace_oof), 3),
    avg_p_error = round(mean(p_error_oof), 3),
    avg_p_fbk   = round(mean(p_fbk_oof), 3),
    .groups = "drop"
  ) %>%
  filter(n_serves >= 10) %>%
  arrange(desc(avg_quality))

print(leaderboard)

# Diagnostic: correlation between the model-based index and raw realized ace
# rate among scored serves. Before the uniform-FBK fix this correlation was
# inflated by realized outcomes leaking into the quality formula.
raw_rates <- serves %>%
  filter(!is.na(quality_index)) %>%
  group_by(player) %>%
  summarise(raw_ace_rate = mean(ace), n = n(), .groups = "drop") %>%
  filter(n >= 10)
diag <- leaderboard %>% inner_join(raw_rates, by = "player")
cat("\nSpearman cor(quality_index, raw ace rate):",
    round(cor(diag$avg_quality, diag$raw_ace_rate, method = "spearman"), 3), "\n")

saveRDS(
  list(
    m1 = m1, m2 = m2, m3 = m3,
    features_base = features_base,
    features_fbk  = features_fbk,
    quality_min   = quality_min,
    quality_max   = quality_max
  ),
  "data/models.rds"
)
saveRDS(serves %>% select(-contest_rank, -fold), "data/serve_quality.rds")
cat("\nDone. Models saved.\n")
