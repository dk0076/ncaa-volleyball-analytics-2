library(shiny)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(here)
library(xgboost)

# Eager-loaded at startup — serve_quality.rds contains all feature columns
# (it is a superset of serves_featured.rds) plus OOF predictions and quality_index.
serves      <- readRDS(here("data", "serve_quality.rds"))
# Contest dates for chronological ordering in latest_rates() — contest IDs are
# assigned sequentially by the NCAA but monotonicity isn't guaranteed by the API.
contest_dates <- readRDS(here("data", "big_west_contests.rds")) %>%
  mutate(date_parsed = as.Date(date, "%m/%d/%Y")) %>%
  select(contestid = contest, date_parsed)
val_metrics <- if (file.exists(here("data", "validation_metrics.rds"))) {
  readRDS(here("data", "validation_metrics.rds"))
} else NULL

# Leaderboard summarization — forward-chained OOF columns throughout.
# Block 1 (earliest matches) has no OOF predictions and is excluded here;
# quality_index is the 0-100 integer; serve_quality is the raw signed composite.
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

teams          <- sort(unique(leaderboard$serve_team))
opponent_teams <- sort(unique(serves$opp_team[serves$serve_team == "Cal Poly"]))

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- fluidPage(
  titlePanel("NCAA Volleyball Serve Quality Index — Cal Poly 2025"),

  sidebarLayout(
    sidebarPanel(
      # Leaderboard / chart controls — hidden on matchup tab
      conditionalPanel(
        condition = "input.tabs !== 'scouting'",
        selectInput("team", "Filter by Team:",
                    choices  = c("All Teams", teams),
                    selected = "Cal Poly"),
        sliderInput("min_serves", "Minimum Serves:",
                    min = 10, max = 100, value = 10)
      ),
      # Scouting controls — only visible on matchup tab
      conditionalPanel(
        condition = "input.tabs === 'scouting'",
        selectInput("opponent", "Opponent:", choices = opponent_teams),
        sliderInput("min_receptions", "Min. Receptions:",
                    min = 10, max = 100, value = 20)
      ),
      hr(),
      p("Serve Quality = P(Ace) \u2212 P(Error) \u2212 P(In Play) \u00d7 P(FBK Against | In Play)"),
      p("Higher = better serving performance."),
      if (!is.null(val_metrics)) {
        tagList(
          hr(),
          strong("Chronological Holdout AUC"),
          tags$ul(
            tags$li(paste("M1 P(Ace):",   val_metrics$auc_m1)),
            tags$li(paste("M2 P(Error):", val_metrics$auc_m2)),
            tags$li(paste("M3 P(FBK):",   val_metrics$auc_m3))
          ),
          p(em(paste0("Holdout: last 20 matches (", val_metrics$n_test, " serves)")),
            style = "font-size:11px; color:#666;")
        )
      }
    ),

    mainPanel(
      tabsetPanel(
        id = "tabs",
        tabPanel("Leaderboard",
          br(),
          tableOutput("leaderboard_table")
        ),
        tabPanel("Quality Chart",
          br(),
          plotOutput("quality_plot", height = "500px")
        ),
        tabPanel("FBK vs Ace",
          br(),
          plotOutput("scatter_plot", height = "500px")
        ),
        tabPanel("Scouting Matchup", value = "scouting",
          br(),
          uiOutput("matchup_loading_msg"),
          h4("Opponent Receiver Targets"),
          tableOutput("receiver_table"),
          br(),
          plotOutput("matchup_heatmap", height = "420px"),
          br(),
          h4("Top Recommendations"),
          tableOutput("recommendations_table")
        )
      )
    )
  )
)

# ── Server ────────────────────────────────────────────────────────────────────
server <- function(input, output) {

  # ── Lazy-load models.rds on first matchup tab access ──────────────────────
  # serve_quality.rds (already loaded as `serves`) is a superset of
  # serves_featured.rds, so only the trained model objects need lazy loading.
  artifacts <- reactiveVal(NULL)

  observeEvent(input$tabs, {
    if (input$tabs == "scouting" && is.null(artifacts())) {
      artifacts(readRDS(here("data", "models.rds")))
    }
  })

  output$matchup_loading_msg <- renderUI({
    if (input$tabs == "scouting" && is.null(artifacts())) {
      p("Loading model data...", style = "color:#666; font-style:italic;")
    }
  })

  # ── Leaderboard / chart reactives ─────────────────────────────────────────
  filtered <- reactive({
    df <- leaderboard %>% filter(n_serves >= input$min_serves)
    if (input$team != "All Teams") df <- df %>% filter(serve_team == input$team)
    df
  })

  output$leaderboard_table <- renderTable({
    filtered() %>%
      arrange(desc(avg_quality)) %>%
      mutate(Rank = row_number()) %>%
      select(Rank, Player = player, Team = serve_team,
             Serves = n_serves, Quality = avg_quality,
             `P(Ace)` = avg_p_ace, `P(Error)` = avg_p_error,
             `P(FBK Against)` = avg_p_fbk)
  })

  output$quality_plot <- renderPlot({
    df <- filtered() %>% arrange(desc(avg_quality))
    if (input$team == "All Teams") df <- slice_head(df, n = 30)
    df$player  <- factor(df$player, levels = df$player)
    cal_poly   <- df$serve_team == "Cal Poly"

    ggplot(df, aes(x = player, y = avg_quality,
                   fill = ifelse(cal_poly, "Cal Poly", "Other"))) +
      geom_col() +
      scale_fill_manual(values = c("Cal Poly" = "#154734", "Other" = "#999999"),
                        name = "") +
      coord_flip() +
      labs(title = "Serve Quality by Player", x = NULL, y = "Average Serve Quality") +
      theme_minimal(base_size = 13)
  })

  output$scatter_plot <- renderPlot({
    df       <- filtered()
    cal_poly <- df$serve_team == "Cal Poly"

    ggplot(df, aes(x = avg_p_fbk, y = avg_p_ace,
                   color = ifelse(cal_poly, "Cal Poly", "Other"),
                   size  = n_serves)) +
      geom_point(alpha = 0.7) +
      ggrepel::geom_text_repel(
        data = df %>% filter(serve_team == "Cal Poly"),
        aes(label = player), size = 3, color = "#154734"
      ) +
      scale_color_manual(values = c("Cal Poly" = "#154734", "Other" = "#999999"),
                         name = "") +
      labs(title = "P(Ace) vs P(FBK Against)",
           x = "P(FBK Against) \u2014 lower is better",
           y = "P(Ace) \u2014 higher is better",
           size = "Serves") +
      theme_minimal(base_size = 13)
  })

  # ── Scouting Matchup reactives ─────────────────────────────────────────────

  # Most recent prior rates per player/receiver/team (mirrors scouting_report.Rmd).
  # Derived from `serves` (serve_quality.rds) — no second file load needed.
  latest_rates <- reactive({
    req(artifacts())  # only compute after models.rds is loaded (tab is active)

    server_rates <- serves %>%
      left_join(contest_dates, by = "contestid") %>%
      arrange(date_parsed) %>%
      group_by(player, serve_team) %>%
      slice_tail(n = 1) %>%
      ungroup() %>%
      select(player, serve_team, prior_ace_rate, prior_error_rate,
             prior_fbk_rate, match_ace_rate, match_error_rate)

    receiver_rates <- serves %>%
      filter(in_play == 1, !is.na(receiver)) %>%
      left_join(contest_dates, by = "contestid") %>%
      arrange(date_parsed) %>%
      group_by(receiver, opp_team) %>%
      slice_tail(n = 1) %>%
      ungroup() %>%
      select(receiver, opp_team, receiver_prior_fbk_rate)

    opp_rates <- serves %>%
      left_join(contest_dates, by = "contestid") %>%
      arrange(date_parsed) %>%
      group_by(opp_team) %>%
      slice_tail(n = 1) %>%
      ungroup() %>%
      select(opp_team, opp_prior_fbk_rate, opp_prior_ace_rate, opp_prior_error_rate)

    list(server = server_rates, receiver = receiver_rates, opp = opp_rates)
  })

  receiver_targets <- reactive({
    req(artifacts(), input$opponent, input$min_receptions)
    use_model_rates <- isTRUE(!is.null(val_metrics) && val_metrics$auc_m3 >= 0.60)
    lr              <- latest_rates()

    targets <- serves %>%
      filter(in_play == 1, opp_team == input$opponent, !is.na(receiver)) %>%
      group_by(receiver) %>%
      summarise(
        Receptions   = n(),
        raw_fbk_rate = mean(fbk_against),
        .groups      = "drop"
      ) %>%
      filter(Receptions >= input$min_receptions) %>%
      left_join(
        lr$receiver %>%
          filter(opp_team == input$opponent) %>%
          select(receiver, receiver_prior_fbk_rate),
        by = "receiver"
      ) %>%
      mutate(
        fbk_rate = round(
          ifelse(use_model_rates, coalesce(receiver_prior_fbk_rate, raw_fbk_rate), raw_fbk_rate),
          3
        )
      ) %>%
      arrange(fbk_rate) %>%
      mutate(
        Rank = row_number(),
        Recommendation = case_when(
          fbk_rate <= quantile(fbk_rate, 0.33) ~ "Primary Target",
          fbk_rate <= quantile(fbk_rate, 0.67) ~ "Secondary Target",
          TRUE                                  ~ "Avoid"
        )
      ) %>%
      select(Rank, Player = receiver, Receptions, `FBK Rate` = fbk_rate, Recommendation)

    validate(need(nrow(targets) > 0,
                  paste0("No receivers with \u2265 ", input$min_receptions,
                         " receptions vs. ", input$opponent, ".")))

    # Verify receiver names matched across the join — a mismatch in opp_team
    # string (e.g. "Cal St. Fullerton" vs "CSUF") would silently leave all
    # receiver_prior_fbk_rate values as NA, falling back to raw_fbk_rate.
    if (use_model_rates) {
      known_receivers <- lr$receiver$receiver[lr$receiver$opp_team == input$opponent]
      n_unmatched     <- sum(!targets$Player %in% known_receivers)
      if (n_unmatched > 0)
        warning(n_unmatched, " of ", nrow(targets), " receivers had no prior FBK rate match",
                " for opponent '", input$opponent, "' — check opp_team string consistency.")
    }

    targets
  })

  matchup_results <- reactive({
    req(artifacts(), receiver_targets(), input$opponent)
    art <- artifacts()
    lr  <- latest_rates()

    cp_servers <- serves %>%
      filter(serve_team == "Cal Poly") %>%
      group_by(player) %>%
      summarise(n = n(), .groups = "drop") %>%
      filter(n >= 20) %>%
      pull(player)

    opp_receivers <- receiver_targets()$Player

    validate(
      need(length(cp_servers) > 0, "No Cal Poly servers with \u2265 20 serves."),
      need(length(opp_receivers) > 0, "No opponent receivers meet the minimum threshold.")
    )

    # Global fallback means computed from the full dataset — not from the small
    # matchups frame, where columns can be mostly or entirely NA after joins.
    mean_fbk   <- mean(serves$fbk_against[serves$in_play == 1], na.rm = TRUE)
    mean_ace   <- mean(serves$ace,           na.rm = TRUE)
    mean_error <- mean(serves$service_error, na.rm = TRUE)

    opp_rate_row <- lr$opp %>% filter(opp_team == input$opponent)

    matchups <- expand.grid(player = cp_servers, receiver = opp_receivers,
                            stringsAsFactors = FALSE) %>%
      left_join(lr$server %>% filter(serve_team == "Cal Poly"), by = "player") %>%
      left_join(lr$receiver %>% filter(opp_team == input$opponent), by = "receiver") %>%
      mutate(
        # Neutral game state — set 2, tied score, away
        set_num    = 2L,
        score_diff = 0L,
        is_home    = 0L,
        is_late_set = 0L,
        # Opponent rates: use latest prior from lr$opp; fall back to global mean
        opp_prior_fbk_rate   = if (nrow(opp_rate_row) == 1) opp_rate_row$opp_prior_fbk_rate   else mean_fbk,
        opp_prior_ace_rate   = if (nrow(opp_rate_row) == 1) opp_rate_row$opp_prior_ace_rate   else mean_ace,
        opp_prior_error_rate = if (nrow(opp_rate_row) == 1) opp_rate_row$opp_prior_error_rate else mean_error
      ) %>%
      mutate(
        prior_ace_rate          = coalesce(prior_ace_rate,          mean_ace),
        prior_error_rate        = coalesce(prior_error_rate,        mean_error),
        prior_fbk_rate          = coalesce(prior_fbk_rate,          mean_fbk),
        receiver_prior_fbk_rate = coalesce(receiver_prior_fbk_rate, mean_fbk),
        match_ace_rate          = coalesce(match_ace_rate,          mean_ace),
        match_error_rate        = coalesce(match_error_rate,        mean_error)
      )

    X_base <- as.matrix(matchups[, art$features_base])
    X_fbk  <- as.matrix(matchups[, art$features_fbk])

    matchups$p_ace     <- predict(art$m1, xgb.DMatrix(X_base))
    matchups$p_error   <- predict(art$m2, xgb.DMatrix(X_base))
    matchups$p_fbk     <- predict(art$m3, xgb.DMatrix(X_fbk))
    matchups$p_in_play <- pmax(0, 1 - matchups$p_ace - matchups$p_error)
    quality_raw        <- matchups$p_ace - matchups$p_error - matchups$p_in_play * matchups$p_fbk
    matchups$quality   <- pmax(0L, pmin(100L, as.integer(round(
      100 * (quality_raw - art$quality_min) / (art$quality_max - art$quality_min)
    ))))

    matchups
  })

  output$receiver_table <- renderTable({
    req(receiver_targets())
    receiver_targets()
  })

  output$matchup_heatmap <- renderPlot({
    req(matchup_results(), receiver_targets())
    df <- matchup_results()
    rt <- receiver_targets()

    # Receivers: left-to-right, weakest passer first (ascending FBK rate)
    # Servers:   top-to-bottom, best quality first (descending mean quality)
    rec_order <- rt$Player
    srv_order <- df %>%
      group_by(player) %>%
      summarise(avg_q = mean(quality), .groups = "drop") %>%
      arrange(avg_q) %>%   # ascending so rev() puts best at top in ggplot
      pull(player)

    df$receiver <- factor(df$receiver, levels = rec_order)
    df$player   <- factor(df$player,   levels = srv_order)

    primary_cols <- rt$Player[rt$Recommendation == "Primary Target"]

    ggplot(df, aes(x = receiver, y = player, fill = quality)) +
      geom_tile(color = "white", linewidth = 0.5) +
      # Outline primary-target columns
      geom_tile(
        data      = df %>% filter(receiver %in% primary_cols),
        color     = "#154734",
        fill      = NA,
        linewidth = 1.3
      ) +
      geom_text(aes(label = quality), size = 3.5, fontface = "bold") +
      scale_fill_gradient2(
        low      = "#d73027",
        mid      = "#ffffbf",
        high     = "#154734",
        midpoint = 50,
        limits   = c(0, 100),
        name     = "Quality\n(0\u2013100)"
      ) +
      labs(
        title    = paste("Serve Quality Matrix \u2014 Cal Poly vs.", input$opponent),
        subtitle = "Receivers ordered weakest \u2192 strongest passer. Outlined = Primary Target.",
        x        = NULL,
        y        = NULL
      ) +
      theme_minimal(base_size = 12) +
      theme(
        axis.text.x  = element_text(angle = 30, hjust = 1, size = 11),
        axis.text.y  = element_text(size = 11),
        plot.subtitle = element_text(size = 10, color = "#555555")
      )
  })

  output$recommendations_table <- renderTable({
    req(matchup_results())
    matchup_results() %>%
      group_by(player) %>%
      slice_max(quality, n = 1, with_ties = FALSE) %>%
      ungroup() %>%
      arrange(desc(quality)) %>%
      slice_head(n = 3) %>%
      mutate(
        Rank                 = row_number(),
        `Quality (0-100)` = quality,
        `P(Ace)`             = round(p_ace, 3),
        `P(FBK | In Play)`   = round(p_fbk, 3)
      ) %>%
      select(Rank,
             `Cal Poly Server` = player,
             `Target Receiver` = receiver,
             `Quality (0-100)`,
             `P(Ace)`,
             `P(FBK | In Play)`)
  })
}

shinyApp(ui = ui, server = server)
