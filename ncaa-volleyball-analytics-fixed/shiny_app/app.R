library(shiny)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(here)

# Resolve the data directory. The pipeline scripts write to `data/` at the
# project root, but a deployed shinyapps.io bundle ships its own copy inside
# shiny_app/ (there is no .Rproj in the bundle, so here() cannot find the root).
# Check both so the same app.R works in either layout.
data_dir <- local({
  candidates <- unique(c("data", tryCatch(here("data"), error = function(e) NULL)))
  found <- candidates[file.exists(file.path(candidates, "serve_quality.rds"))]
  if (!length(found)) {
    stop("Could not find serve_quality.rds. Looked in: ",
         paste(normalizePath(candidates, mustWork = FALSE), collapse = ", "),
         ". Run scripts/01-05 first, or see README > Reproducing the Data.")
  }
  found[1]
})

# Eager-loaded at startup — serve_quality.rds contains all feature columns
# (it is a superset of serves_featured.rds) plus OOF predictions and quality_index.
# The app is a pure presentation layer: every number shown here is produced by
# 05_model.R, so no model objects are loaded or scored at runtime.
serves      <- readRDS(file.path(data_dir, "serve_quality.rds"))
val_metrics <- if (file.exists(file.path(data_dir, "validation_metrics.rds"))) {
  readRDS(file.path(data_dir, "validation_metrics.rds"))
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

teams <- sort(unique(leaderboard$serve_team))

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- fluidPage(
  titlePanel("NCAA Volleyball Serve Quality Index — Cal Poly 2025"),

  sidebarLayout(
    sidebarPanel(
      selectInput("team", "Filter by Team:",
                  choices  = c("All Teams", teams),
                  selected = "Cal Poly"),
      sliderInput("min_serves", "Minimum Serves:",
                  min = 10, max = 100, value = 10),
      hr(),
      p("Serve Quality = P(Ace) − P(Error) − P(In Play) × P(FBK Against | In Play)"),
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
            style = "font-size:11px; color:#666;"),
          p(em(paste0("All three models beat a predict-the-mean baseline but AUC is ",
                      "low. Treat the index as a descriptive, receiver-adjusted ",
                      "summary of past serves — not a forecast of the next one.")),
            style = "font-size:11px; color:#666;")
        )
      }
    ),

    mainPanel(
      tabsetPanel(
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
        )
      )
    )
  )
)

# ── Server ────────────────────────────────────────────────────────────────────
server <- function(input, output) {

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
           x = "P(FBK Against) — lower is better",
           y = "P(Ace) — higher is better",
           size = "Serves") +
      theme_minimal(base_size = 13)
  })
}

shinyApp(ui = ui, server = server)
