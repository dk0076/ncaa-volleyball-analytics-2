# Big West Volleyball Serve Quality Index

An end-to-end sports analytics pipeline quantifying serve effectiveness across the Big West Conference using NCAA play-by-play data, XGBoost modeling, and an interactive R Shiny dashboard.

**[Live dashboard](ADD_SHINYAPPS_URL)** | [Example scouting report](reports/scouting_UCSB.html)

## Motivation

Pitch quality modeling in baseball, epitomized by Statcast-powered XGBoost pipelines, chains conditional probabilities to produce a single interpretable score per pitch. This project applies that framework to volleyball serving, a skill that lacks standardized analytical metrics at the collegiate level.

The core question: **what actually drives first-ball-kill outcomes, the server or the receiver?**

**Short answer:** the receiver. Team-level receiving tendency is the strongest predictor of first-ball-kill outcomes by a wide margin, and server features contribute less than half the signal of receiver-side features. A server with good raw numbers may simply be targeting weak passers. The Serve Quality Index corrects for this.

## Data

-   **Source:** NCAA stats website via the `ncaavolleyballr` R package
-   **Scope:** All matches involving Big West Conference women's volleyball teams plus major West Coast programs (former Pac-12, WCC), 2025 season
-   **Volume:** 542 matches, 87,219 serves, 596,014 play-by-play events
-   **Storage:** DuckDB local analytical database

## Key Findings

**Opponent team receiving tendency is the dominant predictor of FBK outcomes; individual receiver identity adds signal beyond that but is secondary.**

Top SHAP features on the holdout set (mean |SHAP|, M3):

| Feature | Mean \|SHAP\| |
|---|---|
| `opp_prior_fbk_rate` (team) | 0.067 |
| `score_diff` | 0.041 |
| `is_home` | 0.022 |
| `opp_prior_error_rate` | 0.022 |
| `opp_prior_ace_rate` | 0.021 |
| `receiver_prior_fbk_rate` (individual) | 0.020 |
| `prior_fbk_rate` (server) | 0.009 |

Team-level receiving tendency (`opp_prior_fbk_rate`) is the single strongest predictor. Individual receiver identity, once represented as a smoothed prior rate rather than an integer ID, contributes meaningful additional signal but is not dominant. Opponent team ace and error rates contribute at roughly equal magnitude to individual receiver rate, reflecting that team-level receiving quality is a stronger predictor than any single player's tendencies. Server features contribute less than half the signal of receiver-side features.

**What this means:** FBK outcomes are primarily determined by who receives the serve, not who serves it. A server whose opponents concede high FBK rates may be benefiting from targeting weak passers rather than generating serves that are inherently difficult to handle.

**The Serve Quality Index functions as a receiver-adjusted server rating.** Players who consistently face weak passers are penalized relative to raw FBK rates; those who face strong passers are credited. This makes it more useful for cross-opponent comparison than raw statistics.

This also mirrors a known ceiling in outcome-based sports modeling: without spatial tracking data (serve location, trajectory, speed), the model cannot separate server intent from receiver weakness. The volleyball equivalent of Statcast does not yet exist at the NCAA level.

## Metrics

### Raw Outcomes (from PBP)

-   **Ace** serve lands untouched
-   **Service Error** serve out or into net
-   **In Play** serve returned (not ace or error)
-   **FBK Against** receiving team scored immediately after reception; observed directly from `"First ball kill"` events in the PBP event log

### Server Features (rolling, prior matches only, no data leakage)

-   **prior_ace_rate** server's ace rate in all preceding matches
-   **prior_error_rate** server's error rate in all preceding matches
-   **prior_fbk_rate** server's FBK-against rate in all preceding matches
-   **match_ace_rate** server's ace rate so far in current match
-   **match_error_rate** server's error rate so far in current match

### Receiver / Team Features

-   **receiver_prior_fbk_rate** receiver's historical FBK concession rate
-   **opp_prior_fbk_rate** receiving team's overall historical FBK rate
-   **opp_prior_ace_rate** historical ace rate when serving against this team (captures opponent serve-receive quality for M1/M2)
-   **opp_prior_error_rate** historical service error rate when serving against this team

### Game State Features

-   **score_diff** server score minus receiver score at time of serve
-   **set_num** current set number
-   **is_home** whether the serving team is at home
-   **is_late_set** whether either team's score is >= 20 (>= 12 in set 5)

### Model Outputs (XGBoost)

-   **P(Ace)** predicted probability the serve is an ace
-   **P(Error)** predicted probability the serve is an error
-   **P(FBK)** predicted probability of FBK against (in-play serves only)
-   **Serve Quality = P(Ace) - P(Error) - P(In Play) x P(FBK Against | In Play)**

### Scouting Report Metrics

-   **FBK Rate (receivers)** fraction of receptions resulting in FBK against the server; lower = weaker passer = serve target
-   **Serve Quality Index (servers)** average predicted serve quality across all forward-chained out-of-fold serves, min-max scaled to 0-100 (the leaderboard excludes the first chronological block, which has no out-of-fold predictions)
-   **Matchup Quality** serve quality predicted for a specific server x receiver pairing at neutral game state (Set 2, tied score, away)

## Methodology

### Serve Outcome Chain

Each serve results in one of three mutually exclusive outcomes:

```
P(Ace)                    serve lands untouched
P(Error)                  serve out or into net
P(FBK Against | In Play)  opponent kills on first ball after reception
```

### Serve Quality Index

```
Serve Quality = P(Ace) - P(Error) - P(In Play) x P(FBK Against | In Play)
```

Higher scores indicate more effective serving. The formula uses unit weights: ace (+1), error (-1), and FBK against (-1 conditional on in-play) are treated as equal in magnitude. In practice an ace ends the rally entirely while FBK raises the opponent's rally-win probability rather than guaranteeing a point, so this slightly overvalues aces relative to FBK avoidance. A calibrated version would weight by empirical points-won probability per outcome; this is the unit-weight baseline.

The formula is applied identically to every serve from pre-serve features only, so the realized outcome of a serve never enters its own quality score. Because the receiver is unobserved for serves that end as aces or errors (no reception occurs), `receiver_prior_fbk_rate` is imputed with the opponent team's prior FBK rate for those serves so the conditional FBK term can be evaluated uniformly. Conditioning the penalty on whether the serve happened to be returned would leak the realized outcome into a metric that claims to be predictive.

### Feature Engineering

All player-level features are computed from prior matches only to prevent data leakage. Exclusive cumulative sums (`cumsum(x) - x`) are used throughout so no information from the current contest enters the feature values.

**Ace and error models (all serves):**

-   `prior_ace_rate`, `prior_error_rate` server's historical rates
-   `opp_prior_ace_rate`, `opp_prior_error_rate` historical rates when serving against this opponent (replaces integer team ID)
-   `match_ace_rate`, `match_error_rate` within-match rolling rates
-   `score_diff`, `set_num`, `is_home`, `is_late_set` game state

**FBK model (trained on in-play serves; predicts on all serves):**

-   All features above, plus:
-   `prior_fbk_rate` server's historical FBK rate against
-   `receiver_prior_fbk_rate` receiver's historical FBK concession rate (imputed with the opponent team's prior FBK rate when the receiver is unobserved, i.e. for aces and errors)
-   `opp_prior_fbk_rate` receiving team's historical FBK rate

**On feature encoding:** player and team identity is represented entirely through prior rate features rather than integer-encoded IDs. Integer ID encoding is problematic for tree models, since XGBoost treats arbitrary integers as continuous and loses the categorical structure. The prior rates computed chronologically in `04_prior_features.R` are target-encoded equivalents that are also interpretable.

**On prior rate smoothing (Bayesian shrinkage):** all prior rates use a Beta-Binomial posterior mean rather than a raw cumulative rate:

```
smoothed_rate = (cum_outcome + k x global_mean) / (cum_n + k)
```

The shrinkage parameter `k` is estimated per outcome type via empirical Bayes (method of moments on between-entity rate variance). Estimated values: k = 62 for ace rate, 28 for error rate, 50 for FBK rate. When `cum_n = 0` (first match) the formula returns the global mean exactly, replacing the previous hard imputation. As `cum_n` grows the smoothed rate converges to the empirical rate. Low-volume players are pulled toward the league mean in proportion to how noisy their estimates are.

Within-match rates use the same formula with the player's own career prior as the shrinkage target rather than the global mean:

```
match_ace_rate = (match_aces_prior + k x prior_ace_rate) / (match_serves_prior + k)
```

This eliminated the hard switch between "raw within-match rate" and "career prior at serve 1" that was the weakest point in the previous feature engineering. After applying shrinkage, M1 AUC improved from 0.549 to 0.588, exceeding the original integer-ID baseline (0.561).

### Modeling

Three XGBoost binary classifiers, one per outcome. Hyperparameters tuned with 5-fold cross-validation and early stopping (up to 500 rounds, `eta = 0.05`, `max_depth = 4`). Validated on a chronological holdout of the last 20 matches (3,304 serves).

**Forward-chained out-of-fold leaderboard.** Leaderboard scores are out-of-fold predictions, not in-sample fits. The season is split into 5 contiguous date blocks; block k (k = 2..5) is predicted by models trained only on blocks 1 through k-1, so no model weight ever sees future matches. Block 1 has no prior training data, receives no out-of-fold prediction, and is excluded from leaderboard averages (it still receives full-data model predictions for scouting use). The number of boosting rounds is tuned once on the full dataset and reused across blocks, a disclosed simplification that shares a single scalar, not model weights, across time.

Training uses `nthread = 1` and a fixed seed, so all results are bit-reproducible across runs.

### Validation Results

| Model | AUC | Baseline Logloss | Model Logloss |
|---|---|---|---|
| M1 - P(Ace) | 0.588 | 0.228 | 0.225 |
| M2 - P(Service Error) | 0.606 | 0.325 | 0.319 |
| M3 - P(FBK Against) | 0.564 | 0.664 | 0.658 |

AUC values are modest but consistent with the signal available from play-by-play data alone, since no tracking inputs (serve location, speed, type) are available at the NCAA level. All three models beat the predict-the-mean baseline. M1 in particular benefits from Bayesian shrinkage on within-match rates, which smoothly handles the high-noise early-match regime rather than switching hard between the raw rate and the career prior.

## Stack

-   **R** data wrangling, feature engineering, modeling, visualization
-   **DuckDB** local analytical database queried with SQL
-   **XGBoost** gradient boosted tree models with CV tuning
-   **R Shiny** interactive conference-wide serve quality dashboard

## Dependencies

``` r
install.packages(c(
  "dplyr", "tidyr", "duckdb", "DBI", "xgboost", "pROC",
  "ggplot2", "ggrepel", "here", "shiny",
  "knitr", "rmarkdown", "kableExtra", "ncaavolleyballr"
))
```

Exact package versions used for the published results are recorded in `session_info.txt` (written by `06_validation.R`). For a fully pinned environment, run `renv::init()` after installing and commit the generated `renv.lock`.

## Project Structure

```
scripts/
  01_data_pull.R           # Pull all Big West schedules and PBP via ncaavolleyballr
  02_sql_explore.R         # SQL analytical queries against DuckDB (optional exploration)
  03_feature_engineering.R # Rally-level feature construction (receiver, home/away, etc.)
  04_prior_features.R      # Rolling per-player prior rates (no data leakage)
  05_model.R               # XGBoost training, CV tuning, leaderboard, model artifacts
  06_validation.R          # Chronological holdout validation + SHAP feature importance
shiny_app/
  app.R                    # Interactive serve quality dashboard
reports/
  scouting_report.Rmd      # Parameterized pre-match scouting report
  scouting_UCSB.html       # Committed example of a rendered report
data/                      # Not tracked in git, generated by scripts
  volleyball.duckdb        # Raw PBP data
  big_west_contests.rds    # Contest metadata with dates (output of 01)
  serves_clean.rds         # Serve-level dataset with receiver and game state (output of 03)
  serves_featured.rds      # With rolling prior rates for all players (output of 04)
  serve_quality.rds        # Model predictions and quality scores (output of 05)
  models.rds               # Trained models and scaling parameters (output of 05)
session_info.txt           # R and package versions used for published results
LICENSE                    # MIT
```

## Reproducing the Data

`data/` is not tracked in git. Open the project via `ncaa-volleyball-analytics.Rproj` (sets the working directory automatically), then:

**1. Create the data folder**

``` r
dir.create("data", showWarnings = FALSE)
```

**2. Run scripts in order**

``` r
source("scripts/01_data_pull.R")   # live API pull, ~5 sec per match, expect 45-60 min
source("scripts/03_feature_engineering.R")
source("scripts/04_prior_features.R")
source("scripts/05_model.R")
source("scripts/06_validation.R")
```

Script 01 builds the contest list automatically from all team schedules via `ncaavolleyballr`. Skip it if `data/volleyball.duckdb` already exists. Script 02 is optional exploratory SQL and is not part of the pipeline.

**Archived data (recommended for exact reproduction):** the raw NCAA play-by-play data is public, but the live pull takes 45 to 60 minutes and the upstream site can change. A snapshot of `volleyball.duckdb` and `big_west_contests.rds` as pulled for the published results is attached to the v1.1 GitHub release. Download both into `data/` and start from script 03.

**3. Generate a scouting report**

``` r
rmarkdown::render(
  "reports/scouting_report.Rmd",
  params = list(opponent = "UC Santa Barbara"),  # replace with any team in the dataset
  output_file = "scouting_UCSB.html"
)
```

To see valid opponent names:

``` r
serves <- readRDS("data/serves_featured.rds")
sort(unique(c(serves$serve_team, serves$opp_team)))
```

**4. Launch the app**

``` r
shiny::runApp("shiny_app")
```

The app defaults to Cal Poly. Use the team dropdown to view any team or select "All Teams" for the full leaderboard. The Scouting Matchup tab loads model data on first access.

## Limitations and Future Work

-   **Tracking data** (serve location, type, speed) is the primary bottleneck; the model cannot separate server mechanics from receiver weakness without it
-   **Receiver targeting** identifying which player a server is targeting, rather than just who received, would add a strategic layer currently invisible in PBP data
-   **Multiple seasons** would stabilize ratings for low-volume players and enable year-over-year tracking
-   **Multinomial classifier** over {ace, error, in-play} would enforce the simplex constraint and remove the need to clamp `p_ace + p_error`; the current independent binary classifiers are not architecturally constrained to sum to <= 1
-   **Reception quality grades** (if available via DataVolley/VolleyMetrics) would replace the binary FBK outcome with a continuous reception quality score

## Author

Drew King, Statistics B.S., Cal Poly SLO | Sports Analytics | github.com/dk0076
