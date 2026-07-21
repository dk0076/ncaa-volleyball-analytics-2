library(duckdb)
library(dplyr)

con <- dbConnect(duckdb(), dbdir = "data/volleyball.duckdb", read_only = TRUE)

# --- SQL queries against the database ---

# 1. Event type distribution
dbGetQuery(con, "
  SELECT event, COUNT(*) as n
  FROM pbp
  GROUP BY event
  ORDER BY n DESC
")

# 2. Serve outcomes by team
dbGetQuery(con, "
  SELECT team, event, COUNT(*) as n
  FROM pbp
  WHERE event IN ('Serve', 'Ace', 'Service error')
  GROUP BY team, event
  ORDER BY team, event
")

# 3. First ball kill rate by team
dbGetQuery(con, "
  WITH serves AS (
    SELECT team, rally, contestid, set
    FROM pbp
    WHERE event IN ('Serve', 'Ace', 'Service error')
  ),
  fbk AS (
    SELECT team, rally, contestid, set
    FROM pbp
    WHERE event = 'First ball kill'
  )
  SELECT 
    s.team,
    COUNT(*) as total_serves,
    SUM(CASE WHEN f.rally IS NOT NULL THEN 1 ELSE 0 END) as fbk,
    ROUND(SUM(CASE WHEN f.rally IS NOT NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) as fbk_rate
  FROM serves s
  LEFT JOIN fbk f 
    ON s.contestid = f.contestid 
    AND s.set = f.set 
    AND s.rally = f.rally
    AND s.team != f.team
  GROUP BY s.team
  ORDER BY fbk_rate DESC
")

dbDisconnect(con)