library(dplyr)

serves       <- readRDS("data/serves_clean.rds")
bwc_contests <- readRDS("data/big_west_contests.rds")

contest_order <- bwc_contests %>%
  mutate(date_parsed = as.Date(date, "%m/%d/%Y")) %>%
  arrange(date_parsed) %>%
  pull(contest)

serves <- serves %>%
  mutate(contest_idx = match(as.character(contestid), as.character(contest_order)))

stopifnot("contest_idx has NAs — contestid type mismatch with big_west_contests.rds" =
            !any(is.na(serves$contest_idx)))

# ── Global means ───────────────────────────────────────────────────────────────
# Computed first — used both as shrinkage targets and as fallback for entities
# absent from training data entirely (left_join NAs).
mean_ace   <- mean(serves$ace)
mean_error <- mean(serves$service_error)
mean_fbk   <- mean(serves$fbk_against[serves$in_play == 1], na.rm = TRUE)

# ── Shrinkage parameter estimation (empirical Bayes, Beta-Binomial) ───────────
# Model: each player/receiver/team has a true rate drawn from Beta(alpha, beta).
# The posterior mean given cum_n observations and cum_outcome successes is:
#   smoothed_rate = (cum_outcome + k * global_mean) / (cum_n + k)
# where k = alpha + beta controls shrinkage strength.
#
# k is estimated from the data via method of moments:
#   k = mu*(1-mu)/var_between - 1
#   var_between = var(observed_rates) - mean_sampling_variance
#
# Effect: first-match value = global_mean exactly (same as old imputation).
# As cum_n grows, smoothed_rate converges to the empirical rate. Low-volume
# players are pulled toward the league mean more strongly than high-volume ones.
shrinkage_k <- function(totals_n, totals_outcome, global_mean, min_n = 20) {
  keep     <- totals_n >= min_n
  # Sparse fallback: k = 50 is interpretable as "weight the prior as if you had
  # seen 50 observations" — a round number roughly equal to the median per-player
  # serve count. It is the only non-data-driven value in this function.
  if (sum(keep) < 5) return(50)
  rates    <- totals_outcome[keep] / totals_n[keep]
  n_bar    <- mean(totals_n[keep])
  var_obs  <- var(rates)
  var_samp <- global_mean * (1 - global_mean) / n_bar
  var_bet  <- max(var_obs - var_samp, 1e-6)
  max(1, round(global_mean * (1 - global_mean) / var_bet - 1))
}

player_totals <- serves %>%
  group_by(player) %>%
  summarise(n = n(), n_ace = sum(ace), n_error = sum(service_error), .groups = "drop")

k_ace   <- shrinkage_k(player_totals$n, player_totals$n_ace,   mean_ace)
k_error <- shrinkage_k(player_totals$n, player_totals$n_error, mean_error)

receiver_totals <- serves %>%
  filter(in_play == 1, !is.na(receiver)) %>%
  group_by(receiver) %>%
  summarise(n_ip = n(), n_fbk = sum(fbk_against), .groups = "drop")

# k_fbk: between-receiver variance — applied to receiver_prior_fbk_rate and
# server prior_fbk_rate (servers are also receivers in this dataset).
k_fbk <- shrinkage_k(receiver_totals$n_ip, receiver_totals$n_fbk, mean_fbk)

team_totals <- serves %>%
  group_by(opp_team) %>%
  summarise(n = n(), n_ace = sum(ace), n_error = sum(service_error), .groups = "drop")

team_totals_fbk <- serves %>%
  filter(in_play == 1) %>%
  group_by(opp_team) %>%
  summarise(n_ip = n(), n_fbk = sum(fbk_against), .groups = "drop")

# k_opp_fbk estimated from between-team FBK variance — separate from k_fbk
# (receiver level) because teams aggregate more observations and the
# between-entity variance structure differs from individual receivers.
k_opp_fbk   <- shrinkage_k(team_totals_fbk$n_ip, team_totals_fbk$n_fbk, mean_fbk, min_n = 5)
k_opp_ace   <- shrinkage_k(team_totals$n, team_totals$n_ace,             mean_ace,   min_n = 5)
k_opp_error <- shrinkage_k(team_totals$n, team_totals$n_error,           mean_error, min_n = 5)

cat("Shrinkage k — ace:", k_ace, "| error:", k_error, "| fbk:", k_fbk,
    "| opp_fbk:", k_opp_fbk, "| opp_ace:", k_opp_ace, "| opp_error:", k_opp_error, "\n")

# ── Server prior ace / error rates (all serves) ───────────────────────────────
player_contest_stats <- serves %>%
  group_by(player, contestid, contest_idx) %>%
  summarise(n = n(), n_ace = sum(ace), n_error = sum(service_error), .groups = "drop")

server_prior <- player_contest_stats %>%
  arrange(player, contest_idx) %>%
  group_by(player) %>%
  mutate(
    cum_n     = cumsum(n)       - n,
    cum_ace   = cumsum(n_ace)   - n_ace,
    cum_error = cumsum(n_error) - n_error,
    prior_ace_rate   = (cum_ace   + k_ace   * mean_ace)   / (cum_n + k_ace),
    prior_error_rate = (cum_error + k_error * mean_error) / (cum_n + k_error)
  ) %>%
  ungroup() %>%
  select(player, contestid, prior_ace_rate, prior_error_rate)

# ── Server prior FBK rate (in-play serves only) ───────────────────────────────
server_fbk <- serves %>%
  filter(in_play == 1) %>%
  group_by(player, contestid, contest_idx) %>%
  summarise(n_ip = n(), n_fbk = sum(fbk_against), .groups = "drop") %>%
  arrange(player, contest_idx) %>%
  group_by(player) %>%
  mutate(
    cum_n_ip = cumsum(n_ip) - n_ip,
    cum_fbk  = cumsum(n_fbk) - n_fbk,
    prior_fbk_rate = (cum_fbk + k_fbk * mean_fbk) / (cum_n_ip + k_fbk)
  ) %>%
  ungroup() %>%
  select(player, contestid, prior_fbk_rate)

# ── Receiver prior FBK rate (in-play serves, keyed on receiver) ───────────────
receiver_fbk <- serves %>%
  filter(in_play == 1, !is.na(receiver)) %>%
  group_by(receiver, contestid, contest_idx) %>%
  summarise(n_ip = n(), n_fbk = sum(fbk_against), .groups = "drop") %>%
  arrange(receiver, contest_idx) %>%
  group_by(receiver) %>%
  mutate(
    cum_n_ip = cumsum(n_ip) - n_ip,
    cum_fbk  = cumsum(n_fbk) - n_fbk,
    receiver_prior_fbk_rate = (cum_fbk + k_fbk * mean_fbk) / (cum_n_ip + k_fbk)
  ) %>%
  ungroup() %>%
  select(receiver, contestid, receiver_prior_fbk_rate)

# ── Opponent team prior FBK rate ──────────────────────────────────────────────
opp_team_fbk <- serves %>%
  filter(in_play == 1) %>%
  group_by(opp_team, contestid, contest_idx) %>%
  summarise(n_ip = n(), n_fbk = sum(fbk_against), .groups = "drop") %>%
  arrange(opp_team, contest_idx) %>%
  group_by(opp_team) %>%
  mutate(
    cum_n_ip = cumsum(n_ip) - n_ip,
    cum_fbk  = cumsum(n_fbk) - n_fbk,
    opp_prior_fbk_rate = (cum_fbk + k_opp_fbk * mean_fbk) / (cum_n_ip + k_opp_fbk)
  ) %>%
  ungroup() %>%
  select(opp_team, contestid, opp_prior_fbk_rate)

# ── Opponent team prior ace / error rates ─────────────────────────────────────
# Captures how good this opponent's serve receive is — teams ace more often
# against weaker receivers. Replaces the integer opp_team_id in M1/M2.
opp_team_ace_error <- serves %>%
  group_by(opp_team, contestid, contest_idx) %>%
  summarise(n = n(), n_ace = sum(ace), n_error = sum(service_error), .groups = "drop") %>%
  arrange(opp_team, contest_idx) %>%
  group_by(opp_team) %>%
  mutate(
    cum_n     = cumsum(n)       - n,
    cum_ace   = cumsum(n_ace)   - n_ace,
    cum_error = cumsum(n_error) - n_error,
    opp_prior_ace_rate   = (cum_ace   + k_opp_ace   * mean_ace)   / (cum_n + k_opp_ace),
    opp_prior_error_rate = (cum_error + k_opp_error * mean_error) / (cum_n + k_opp_error)
  ) %>%
  ungroup() %>%
  select(opp_team, contestid, opp_prior_ace_rate, opp_prior_error_rate)

# ── Combine all features ──────────────────────────────────────────────────────
# The shrinkage formula handles first-match values (cum_n == 0 → global_mean)
# so no imputation is needed for that case. The coalesce fallbacks below only
# fire for entities entirely absent from training data (new players/teams).
serves_featured <- serves %>%
  left_join(server_prior,       by = c("player",   "contestid")) %>%
  left_join(server_fbk,         by = c("player",   "contestid")) %>%
  left_join(receiver_fbk,       by = c("receiver", "contestid")) %>%
  left_join(opp_team_fbk,       by = c("opp_team", "contestid")) %>%
  left_join(opp_team_ace_error, by = c("opp_team", "contestid")) %>%
  mutate(
    prior_ace_rate          = coalesce(prior_ace_rate,          mean_ace),
    prior_error_rate        = coalesce(prior_error_rate,        mean_error),
    prior_fbk_rate          = coalesce(prior_fbk_rate,          mean_fbk),
    opp_prior_fbk_rate      = coalesce(opp_prior_fbk_rate,      mean_fbk),
    opp_prior_ace_rate      = coalesce(opp_prior_ace_rate,      mean_ace),
    opp_prior_error_rate    = coalesce(opp_prior_error_rate,    mean_error),
    # Receiver identity is unobserved for serves that end as aces or errors
    # (no Reception event fires). Impute with the opponent TEAM's prior FBK
    # rate — the best available pre-serve estimate of who would have received —
    # so M3 can score every serve, not just the ones that happened to be
    # returned. Falls back to the global mean only for teams entirely absent
    # from training data. Computed after opp_prior_fbk_rate above so the
    # team-level rate is available within this mutate().
    receiver_prior_fbk_rate = coalesce(receiver_prior_fbk_rate,
                                       opp_prior_fbk_rate, mean_fbk),
    # Within-match rates — shrink toward player's own career prior rather than
    # hard-switching at serve 1. When match_serves_prior == 0 this returns
    # prior_ace_rate exactly; as match volume grows it converges to the raw rate.
    # Note: k_ace / k_error were estimated from between-player career variance,
    # not between-match within-player variance — the latter is typically smaller,
    # meaning these k values may over-shrink slightly in the within-match context.
    # In practice the effect is minor given the small within-match sample sizes.
    match_ace_rate   = (match_aces_prior   + k_ace   * prior_ace_rate)   / (match_serves_prior + k_ace),
    match_error_rate = (match_errors_prior + k_error * prior_error_rate) / (match_serves_prior + k_error)
  ) %>%
  select(-contest_idx, -match_serves_prior, -match_aces_prior, -match_errors_prior)

saveRDS(serves_featured, "data/serves_featured.rds")
cat("Saved serves_featured.rds:", nrow(serves_featured), "rows\n")
