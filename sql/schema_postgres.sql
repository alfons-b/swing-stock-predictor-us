-- Skema PostgreSQL (Aiven / PostgreSQL apa pun). Dihasilkan oleh `python main.py db-schema`.
-- Opsional: jalankan di SQL Editor provider. `python main.py setup` juga membuatnya otomatis.

CREATE TABLE IF NOT EXISTS stocks (
  id BIGSERIAL PRIMARY KEY,
  ticker TEXT NOT NULL,
  name TEXT,
  sector TEXT,
  subsector TEXT,
  board TEXT,
  listing_date DATE,
  delisting_date DATE,
  is_active BOOLEAN NOT NULL,
  previous_ticker TEXT,
  listed_shares DOUBLE PRECISION,
  fundamentals_checked_at TIMESTAMPTZ,
  sector_checked_at TIMESTAMPTZ,
  first_seen TIMESTAMPTZ,
  last_seen TIMESTAMPTZ,
  updated_at TIMESTAMPTZ,
  UNIQUE (ticker)
);

CREATE TABLE IF NOT EXISTS price_history (
  id BIGSERIAL PRIMARY KEY,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  date DATE NOT NULL,
  open DOUBLE PRECISION,
  high DOUBLE PRECISION,
  low DOUBLE PRECISION,
  close DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  value DOUBLE PRECISION,
  frequency DOUBLE PRECISION,
  is_adjusted BOOLEAN,
  source TEXT,
  ingested_at TIMESTAMPTZ,
  UNIQUE (stock_id, date)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
  id BIGSERIAL PRIMARY KEY,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  ex_date DATE NOT NULL,
  action TEXT NOT NULL,
  ratio DOUBLE PRECISION,
  amount DOUBLE PRECISION,
  source TEXT,
  processed_at TIMESTAMPTZ,
  UNIQUE (stock_id, ex_date, action)
);

CREATE TABLE IF NOT EXISTS market_index (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  date DATE NOT NULL,
  open DOUBLE PRECISION,
  high DOUBLE PRECISION,
  low DOUBLE PRECISION,
  close DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  source TEXT,
  ingested_at TIMESTAMPTZ,
  UNIQUE (symbol, date)
);

CREATE TABLE IF NOT EXISTS sector_data (
  id BIGSERIAL PRIMARY KEY,
  date DATE NOT NULL,
  sector TEXT NOT NULL,
  ret5 DOUBLE PRECISION,
  ret20 DOUBLE PRECISION,
  ret60 DOUBLE PRECISION,
  breadth DOUBLE PRECISION,
  volume_mom DOUBLE PRECISION,
  rs20 DOUBLE PRECISION,
  score DOUBLE PRECISION,
  rank BIGINT,
  n_stocks BIGINT,
  UNIQUE (date, sector)
);

CREATE TABLE IF NOT EXISTS features (
  id BIGSERIAL PRIMARY KEY,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  date DATE NOT NULL,
  feature_version TEXT NOT NULL,
  payload TEXT,
  UNIQUE (stock_id, date, feature_version)
);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  prediction_date DATE NOT NULL,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  ticker TEXT NOT NULL,
  model_version TEXT NOT NULL,
  horizon BIGINT,
  decision TEXT,
  setup TEXT,
  score DOUBLE PRECISION,
  market_regime TEXT,
  prob_bearish DOUBLE PRECISION,
  prob_neutral DOUBLE PRECISION,
  prob_bullish DOUBLE PRECISION,
  expected_return DOUBLE PRECISION,
  prob_hit_tp DOUBLE PRECISION,
  entry_low DOUBLE PRECISION,
  entry_ideal DOUBLE PRECISION,
  entry_high DOUBLE PRECISION,
  stop_loss DOUBLE PRECISION,
  tp1 DOUBLE PRECISION,
  tp2 DOUBLE PRECISION,
  risk_reward DOUBLE PRECISION,
  position_size BIGINT,
  lots BIGINT,
  capital_required DOUBLE PRECISION,
  estimated_loss DOUBLE PRECISION,
  confidence TEXT,
  close_price DOUBLE PRECISION,
  data_status TEXT,
  reasons TEXT,
  risks TEXT,
  reject_reasons TEXT,
  run_id TEXT,
  created_at TIMESTAMPTZ,
  value_score DOUBLE PRECISION,
  quality_score DOUBLE PRECISION,
  value_trap_risk TEXT,
  margin_of_safety DOUBLE PRECISION,
  valuation_status TEXT,
  foreign_flow_score DOUBLE PRECISION,
  foreign_flow_status TEXT,
  accumulation_score DOUBLE PRECISION,
  distribution_risk DOUBLE PRECISION,
  accumulation_status TEXT,
  accumulation_stage TEXT,
  UNIQUE (prediction_date, stock_id, model_version)
);

CREATE TABLE IF NOT EXISTS prediction_evaluations (
  id BIGSERIAL PRIMARY KEY,
  prediction_id BIGINT NOT NULL REFERENCES predictions(id),
  evaluated_at TIMESTAMPTZ,
  actual_return_3d DOUBLE PRECISION,
  actual_return_5d DOUBLE PRECISION,
  actual_return_10d DOUBLE PRECISION,
  mfe DOUBLE PRECISION,
  mae DOUBLE PRECISION,
  entry_filled BOOLEAN,
  hit_stop BOOLEAN,
  hit_tp1 BOOLEAN,
  hit_tp2 BOOLEAN,
  actual_class BIGINT,
  prediction_correct BOOLEAN,
  outcome TEXT,
  UNIQUE (prediction_id)
);

CREATE TABLE IF NOT EXISTS trading_signals (
  id BIGSERIAL PRIMARY KEY,
  signal_date DATE NOT NULL,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  ticker TEXT NOT NULL,
  rank BIGINT,
  decision TEXT,
  setup TEXT,
  score DOUBLE PRECISION,
  prob_bullish DOUBLE PRECISION,
  expected_return DOUBLE PRECISION,
  entry_low DOUBLE PRECISION,
  entry_ideal DOUBLE PRECISION,
  entry_high DOUBLE PRECISION,
  stop_loss DOUBLE PRECISION,
  tp1 DOUBLE PRECISION,
  tp2 DOUBLE PRECISION,
  risk_reward DOUBLE PRECISION,
  lots BIGINT,
  confidence TEXT,
  model_version TEXT,
  run_id TEXT,
  UNIQUE (signal_date, stock_id)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL,
  created_at TIMESTAMPTZ,
  kind TEXT,
  period_start DATE,
  period_end DATE,
  model_version TEXT,
  config_hash TEXT,
  metrics TEXT,
  benchmark TEXT,
  status TEXT,
  UNIQUE (run_id)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
  id BIGSERIAL PRIMARY KEY,
  backtest_run_id TEXT NOT NULL,
  ticker TEXT,
  setup TEXT,
  entry_date DATE,
  exit_date DATE,
  entry_price DOUBLE PRECISION,
  exit_price DOUBLE PRECISION,
  shares BIGINT,
  net_pnl DOUBLE PRECISION,
  net_return DOUBLE PRECISION,
  r_multiple DOUBLE PRECISION,
  exit_reason TEXT,
  fees DOUBLE PRECISION,
  UNIQUE (backtest_run_id, ticker, entry_date)
);

CREATE TABLE IF NOT EXISTS model_versions (
  id BIGSERIAL PRIMARY KEY,
  version TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ,
  promoted_at TIMESTAMPTZ,
  train_start DATE,
  train_end DATE,
  feature_version TEXT,
  config_hash TEXT,
  classifiers TEXT,
  metrics TEXT,
  storage_backend TEXT,
  storage_uri TEXT,
  artifact BYTEA,
  artifact_sha256 TEXT,
  artifact_bytes BIGINT,
  notes TEXT,
  UNIQUE (version)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL,
  run_type TEXT NOT NULL,
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  status TEXT,
  stocks_processed BIGINT,
  stocks_failed BIGINT,
  rows_inserted BIGINT,
  rows_updated BIGINT,
  model_version TEXT,
  data_status TEXT,
  market_date DATE,
  trigger TEXT,
  summary TEXT,
  error_message TEXT,
  created_at TIMESTAMPTZ,
  UNIQUE (run_id)
);

CREATE TABLE IF NOT EXISTS data_sources (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT,
  priority BIGINT,
  enabled BOOLEAN,
  last_success_at TIMESTAMPTZ,
  last_error_at TIMESTAMPTZ,
  last_error TEXT,
  rows_fetched_total BIGINT,
  category TEXT,
  status TEXT,
  detail TEXT,
  checked_at TIMESTAMPTZ,
  UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS system_logs (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT,
  ts TIMESTAMPTZ,
  level TEXT,
  logger TEXT,
  message TEXT
);

CREATE TABLE IF NOT EXISTS reports (
  id BIGSERIAL PRIMARY KEY,
  report_date DATE NOT NULL,
  kind TEXT NOT NULL,
  filename TEXT NOT NULL,
  content BYTEA,
  content_type TEXT,
  storage_backend TEXT,
  storage_uri TEXT,
  run_id TEXT,
  created_at TIMESTAMPTZ,
  UNIQUE (report_date, kind, filename)
);

CREATE TABLE IF NOT EXISTS financial_statements (
  id BIGSERIAL PRIMARY KEY,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  period_end DATE NOT NULL,
  period_type TEXT NOT NULL,
  fiscal_year BIGINT,
  currency TEXT,
  items TEXT,
  items_hash TEXT,
  publication_date DATE,
  first_known_date DATE NOT NULL,
  estimated_available_date DATE,
  retrieved_at TIMESTAMPTZ,
  source TEXT NOT NULL,
  version BIGINT NOT NULL,
  quality_status TEXT,
  quality_notes TEXT,
  UNIQUE (stock_id, period_end, period_type, source, version)
);

CREATE TABLE IF NOT EXISTS valuation_results (
  id BIGSERIAL PRIMARY KEY,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  as_of_date DATE NOT NULL,
  price DOUBLE PRECISION,
  sector_type TEXT,
  fair_value_low DOUBLE PRECISION,
  fair_value_base DOUBLE PRECISION,
  fair_value_high DOUBLE PRECISION,
  margin_of_safety DOUBLE PRECISION,
  margin_of_safety_conservative DOUBLE PRECISION,
  valuation_status TEXT,
  valuation_confidence TEXT,
  value_score DOUBLE PRECISION,
  quality_score DOUBLE PRECISION,
  value_trap_risk TEXT,
  methods TEXT,
  metrics TEXT,
  value_trap_reasons TEXT,
  fundamentals_known_date DATE,
  fundamentals_period_end DATE,
  data_status TEXT,
  run_id TEXT,
  created_at TIMESTAMPTZ,
  UNIQUE (stock_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS foreign_flow_history (
  id BIGSERIAL PRIMARY KEY,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  date DATE NOT NULL,
  market_segment TEXT NOT NULL,
  foreign_buy_shares DOUBLE PRECISION,
  foreign_sell_shares DOUBLE PRECISION,
  net_foreign_shares DOUBLE PRECISION,
  foreign_buy_value DOUBLE PRECISION,
  foreign_sell_value DOUBLE PRECISION,
  net_foreign_value DOUBLE PRECISION,
  value_type TEXT,
  total_volume_shares DOUBLE PRECISION,
  units TEXT,
  source TEXT,
  source_timestamp TIMESTAMPTZ,
  retrieved_at TIMESTAMPTZ,
  quality_status TEXT,
  quality_notes TEXT,
  UNIQUE (stock_id, date, market_segment)
);

CREATE TABLE IF NOT EXISTS accumulation_signals (
  id BIGSERIAL PRIMARY KEY,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  date DATE NOT NULL,
  accumulation_score DOUBLE PRECISION,
  distribution_risk DOUBLE PRECISION,
  status TEXT,
  stage TEXT,
  confidence TEXT,
  components TEXT,
  evidence TEXT,
  foreign_flow_status TEXT,
  run_id TEXT,
  UNIQUE (stock_id, date)
);

CREATE TABLE IF NOT EXISTS scan_coverage (
  id BIGSERIAL PRIMARY KEY,
  scan_date DATE NOT NULL,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  ticker TEXT NOT NULL,
  status TEXT NOT NULL,
  reason TEXT,
  last_data_date DATE,
  history_rows BIGINT,
  run_id TEXT,
  created_at TIMESTAMPTZ,
  UNIQUE (scan_date, stock_id)
);

CREATE TABLE IF NOT EXISTS challenger_models (
  id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  version TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ,
  train_start DATE,
  train_end DATE,
  eval_start DATE,
  eval_end DATE,
  params TEXT,
  metrics TEXT,
  artifact BYTEA,
  run_id TEXT,
  notes TEXT,
  UNIQUE (model_name, version)
);

CREATE TABLE IF NOT EXISTS model_forecasts (
  id BIGSERIAL PRIMARY KEY,
  forecast_date DATE NOT NULL,
  stock_id BIGINT NOT NULL REFERENCES stocks(id),
  ticker TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_version TEXT,
  horizon BIGINT,
  exp_return DOUBLE PRECISION,
  prob_up DOUBLE PRECISION,
  prob_bullish DOUBLE PRECISION,
  prob_bearish DOUBLE PRECISION,
  vol DOUBLE PRECISION,
  q10 DOUBLE PRECISION,
  q50 DOUBLE PRECISION,
  q90 DOUBLE PRECISION,
  rank_score DOUBLE PRECISION,
  run_id TEXT,
  created_at TIMESTAMPTZ,
  UNIQUE (forecast_date, stock_id, model_name)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
  version BIGINT PRIMARY KEY,
  applied_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_stocks_is_active ON stocks (is_active);

CREATE INDEX IF NOT EXISTS ix_stocks_sector ON stocks (sector);

CREATE INDEX IF NOT EXISTS ix_price_history_date ON price_history (date);

CREATE INDEX IF NOT EXISTS ix_corporate_actions_ex_date ON corporate_actions (ex_date);

CREATE INDEX IF NOT EXISTS ix_market_index_date ON market_index (date);

CREATE INDEX IF NOT EXISTS ix_sector_data_date ON sector_data (date);

CREATE INDEX IF NOT EXISTS ix_features_date ON features (date);

CREATE INDEX IF NOT EXISTS ix_predictions_ticker ON predictions (ticker);

CREATE INDEX IF NOT EXISTS ix_predictions_decision ON predictions (decision);

CREATE INDEX IF NOT EXISTS ix_predictions_model_version ON predictions (model_version);

CREATE INDEX IF NOT EXISTS ix_trading_signals_signal_date ON trading_signals (signal_date);

CREATE INDEX IF NOT EXISTS ix_backtest_runs_created_at ON backtest_runs (created_at);

CREATE INDEX IF NOT EXISTS ix_backtest_trades_backtest_run_id ON backtest_trades (backtest_run_id);

CREATE INDEX IF NOT EXISTS ix_model_versions_status ON model_versions (status);

CREATE INDEX IF NOT EXISTS ix_pipeline_runs_run_type_started_at ON pipeline_runs (run_type, started_at);

CREATE INDEX IF NOT EXISTS ix_system_logs_run_id ON system_logs (run_id);

CREATE INDEX IF NOT EXISTS ix_system_logs_ts ON system_logs (ts);

CREATE INDEX IF NOT EXISTS ix_reports_report_date ON reports (report_date);

CREATE INDEX IF NOT EXISTS ix_financial_statements_stock_id_first_known_date ON financial_statements (stock_id, first_known_date);

CREATE INDEX IF NOT EXISTS ix_valuation_results_as_of_date ON valuation_results (as_of_date);

CREATE INDEX IF NOT EXISTS ix_foreign_flow_history_date ON foreign_flow_history (date);

CREATE INDEX IF NOT EXISTS ix_accumulation_signals_date ON accumulation_signals (date);

CREATE INDEX IF NOT EXISTS ix_scan_coverage_scan_date_status ON scan_coverage (scan_date, status);

CREATE INDEX IF NOT EXISTS ix_challenger_models_model_name_status ON challenger_models (model_name, status);

CREATE INDEX IF NOT EXISTS ix_model_forecasts_forecast_date ON model_forecasts (forecast_date);

CREATE INDEX IF NOT EXISTS ix_model_forecasts_model_name_forecast_date ON model_forecasts (model_name, forecast_date);
