library(duckdb)
library(dplyr)

con <- dbConnect(duckdb(), dbdir = "data/volleyball.duckdb", read_only = TRUE)
pbp <- dbGetQuery(con, "SELECT * FROM pbp")
dbDisconnect(con)

# FBK rallies
fbk_rallies <- pbp %>%
  filter(event == "First ball kill") %>%
  select(contestid, set, rally, fbk_team = team)

# Receiver for each in-play rally (first Reception event)
receivers <- pbp %>%
  filter(event == "Reception") %>%
  arrange(contestid, set, rally, rally_event) %>%
  group_by(contestid, set, rally) %>%
  slice(1) %>%
  ungroup() %>%
  select(contestid, set, rally, receiver = player)

# Serve outcome lookup tables.
# "Service error" events are tagged to the RECEIVING team in ncaavolleyballr,
# not the serving team. "Ace" events ARE tagged to the serving team but a
# "Serve" event also fires for the same rally. Strategy: use only "Serve"
# events as the base (one row per rally, always tagged to the server), then
# join ace/error outcomes by rally key.
ace_rallies <- pbp %>%
  filter(event == "Ace") %>%
  select(contestid, set, rally) %>%
  distinct() %>%
  mutate(is_ace = 1L)

error_rallies <- pbp %>%
  filter(event == "Service error") %>%
  select(contestid, set, rally) %>%
  distinct() %>%
  mutate(is_error = 1L)

# Build serve-level dataset
# home_team and away_team are native pbp columns — no extra join needed.
# Exclude rows where score is not a valid "away-home" string (e.g. "Media timeout"
# rows that ncaavolleyballr tags with event == "Serve" but carry no score data).
serves <- pbp %>%
  filter(event == "Serve", grepl("^[0-9]+-[0-9]+$", score)) %>%
  left_join(ace_rallies,   by = c("contestid", "set", "rally")) %>%
  left_join(error_rallies, by = c("contestid", "set", "rally")) %>%
  left_join(fbk_rallies,   by = c("contestid", "set", "rally")) %>%
  left_join(receivers,     by = c("contestid", "set", "rally")) %>%
  mutate(
    ace            = coalesce(is_ace,   0L),
    service_error  = coalesce(is_error, 0L),
    in_play        = as.integer(is.na(is_ace) & is.na(is_error)),
    fbk_against    = as.integer(!is.na(fbk_team) & fbk_team != team),
    set_num        = as.integer(set),
    # Score string is always away-home (e.g. "2-3" = away 2, home 3)
    score_away     = as.integer(trimws(sub("-.*", "", score))),
    score_home     = as.integer(trimws(sub(".*-", "", score))),
    score_diff     = ifelse(team == home_team,
                            score_home - score_away,   # home server: positive = winning
                            score_away - score_home),  # away server: positive = winning
    is_home        = as.integer(team == home_team),
    is_late_set    = as.integer(
      pmax(score_away, score_home, na.rm = TRUE) >=
        ifelse(as.integer(set) == 5, 12, 20)
    ),
    opp_team       = ifelse(team == home_team, away_team, home_team)
  ) %>%
  select(-is_ace, -is_error) %>%
  # Within-match rolling counts (excludes current serve)
  arrange(contestid, set_num, rally) %>%
  group_by(contestid, player) %>%
  mutate(
    match_serves_prior = row_number() - 1L,
    match_aces_prior   = cumsum(ace)           - ace,
    match_errors_prior = cumsum(service_error) - service_error
  ) %>%
  ungroup() %>%
  select(
    contestid, set_num, rally,
    serve_team = team, player, receiver, opp_team,
    ace, service_error, in_play, fbk_against,
    score_diff, is_home, is_late_set,
    match_serves_prior, match_aces_prior, match_errors_prior
  )

# Verify score parsing produced no unexpected NAs.
# as.integer() coercion failures (malformed score strings) surface here rather
# than propagating silently into score_diff and is_late_set.
score_na <- sum(is.na(serves$score_diff))
if (score_na > 0) {
  warning(score_na, " NAs in score_diff — inspect malformed score strings in PBP data.")
} else {
  cat("score_diff: 0 NAs\n")
}

glimpse(serves)
cat("\nTotal serves:", nrow(serves))
cat("\nAce rate:", round(mean(serves$ace), 3))
cat("\nService error rate:", round(mean(serves$service_error), 3))
cat("\nFBK against rate (in-play only):",
    round(mean(serves$fbk_against[serves$in_play == 1], na.rm = TRUE), 3))
cat("\n")

saveRDS(serves, "data/serves_clean.rds")
cat("Saved to data/serves_clean.rds\n")
