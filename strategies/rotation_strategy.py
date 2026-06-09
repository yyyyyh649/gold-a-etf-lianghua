"""Rotation strategy between gold futures and A-share ETF/index."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

Rebalance = Literal["daily", "weekly", "monthly"]
AlignMethod = Literal["inner", "a_share_ffill"]


@dataclass
class RotationConfig:
	lookback_days: int = 60
	rebalance: Rebalance = "weekly"
	fee_bps: float = 5.0  # one-way trading cost in basis points
	cash_symbol: str = "CASH"
	cash_annual_return: float = 0.0  # annualised return for cash position (e.g. 0.02 = 2%)
	align_method: AlignMethod = "a_share_ffill"
	"""How to handle CN/US trading calendar misalignment.

	- "inner":         只保留两个市场都有数据的日期。最保守，但会丢失大量数据点
	                   （如国庆期间黄金暴跌会被忽略）。
	- "a_share_ffill": 以 A 股交易日历为基准，用 ffill 补黄金最多 5 天缺失
	                   （覆盖美国假期）。保留更多数据点，回测更贴近真实。
	"""


def _prepare_prices(
	gold: pd.DataFrame,
	equity: pd.DataFrame,
	align_method: AlignMethod = "a_share_ffill",
) -> pd.DataFrame:
	"""Align gold and equity price series on a common date index.

	Args:
		align_method: See RotationConfig.align_method.
	"""
	gold_series = gold.set_index("Date")["Close"].rename("GOLD")
	equity_series = equity.set_index("Date")["Close"].rename("EQUITY")

	if align_method == "inner":
		prices = pd.concat([gold_series, equity_series], axis=1, join="inner")
	elif align_method == "a_share_ffill":
		prices = pd.concat([gold_series, equity_series], axis=1, join="outer").sort_index()
		prices["GOLD"] = prices["GOLD"].ffill(limit=5)
		prices = prices.dropna(subset=["EQUITY"])
	else:
		raise ValueError(f"Unknown align_method: {align_method}")

	prices = prices.dropna()
	return prices


def _rebalance_index(prices: pd.DataFrame, mode: Rebalance) -> pd.DatetimeIndex:
	if mode == "daily":
		return prices.index
	if mode == "weekly":
		return prices.resample("W-FRI").last().index
	if mode == "monthly":
		return prices.resample("M").last().index
	raise ValueError(f"Unsupported rebalance mode: {mode}")


def _compute_momentum_on_rebalance(
	prices: pd.DataFrame,
	mode: Rebalance,
	lookback_days: int,
) -> pd.DataFrame:
	"""Compute asset momentum on rebalance-frequency bars.

	Instead of computing pct_change(N) on daily data (which changes meaning
	depending on trading calendar density), this function:
	1. Resamples prices to the rebalance frequency (weekly/monthly)
	2. Computes returns over a number of bars that approximates the
	   same calendar span as ``lookback_days`` on daily data

	Mapping:
	- daily rebalance  → lookback = lookback_days bars (daily)
	- weekly rebalance → lookback = max(1, lookback_days // 5) bars (weeks)
	- monthly rebalance → lookback = max(1, lookback_days // 20) bars (months)
	"""
	if mode == "daily":
		resampled = prices
		periods = lookback_days
	elif mode == "weekly":
		resampled = prices.resample("W-FRI").last().dropna()
		periods = max(1, lookback_days // 5)
	elif mode == "monthly":
		resampled = prices.resample("M").last().dropna()
		periods = max(1, lookback_days // 20)
	else:
		raise ValueError(f"Unsupported rebalance mode: {mode}")

	momentum = resampled.pct_change(periods).dropna()
	momentum_daily = momentum.reindex(prices.index).ffill()
	return momentum_daily


def parameter_sweep(
	gold: pd.DataFrame,
	equity: pd.DataFrame,
	lookback_windows: Optional[List[int]] = None,
	rebalance: Rebalance = "weekly",
) -> pd.DataFrame:
	"""Run the strategy across multiple lookback windows and return a comparison table."""

	if lookback_windows is None:
		lookback_windows = [20, 60, 120]

	results: Dict[int, dict] = {}
	for lb in lookback_windows:
		cfg = RotationConfig(lookback_days=lb, rebalance=rebalance)
		result = generate_signals(gold, equity, config=cfg)
		metrics = performance_summary(result["portfolio_ret"])
		results[lb] = metrics

	rows = []
	for lb in lookback_windows:
		m = results[lb]
		rows.append(
			{
				"lookback": f"{lb}d",
				"cagr": m.get("cagr", float("nan")),
				"vol": m.get("vol", float("nan")),
				"sharpe": m.get("sharpe", float("nan")),
				"max_dd": m.get("max_drawdown", float("nan")),
				"calmar": m.get("calmar", float("nan")),
				"last_nav": m.get("last_value", float("nan")),
			}
		)
	return pd.DataFrame(rows).set_index("lookback")


def generate_signals(
	gold: pd.DataFrame,
	equity: pd.DataFrame,
	config: Optional[RotationConfig] = None,
) -> pd.DataFrame:
	"""Create allocation signals using lookback momentum.

	Returns a dataframe with columns: signal, position, gold_ret, equity_ret, portfolio_ret.
	position is one of ["GOLD", "EQUITY", config.cash_symbol].

	Look-ahead bias protection:
	- Momentum is computed using T-period returns ending at close(T).
	- Signal decision is made at close(T) using close(T) prices.
	- *Trade execution* happens at T+1 open (approximated as the next day's return).
	  This is enforced by ``position.shift(1)``.
	- Unit test in tests/ verifies that no future data leaks into the signal.
	"""

	cfg = config or RotationConfig()
	prices = _prepare_prices(gold, equity, cfg.align_method)

	if prices.empty:
		# No overlapping trading dates → nothing to simulate.
		return pd.DataFrame(
			columns=["signal", "position", "gold_ret", "equity_ret", "portfolio_ret"]
		)

	daily_ret = prices.pct_change().fillna(0.0)

	# --- Momentum on rebalance-frequency bars ---
	# Use resampled returns so the lookback period has a stable calendar meaning.
	momentum = _compute_momentum_on_rebalance(
		prices, cfg.rebalance, lookback_days=cfg.lookback_days,
	)

	rebalance_dates = _rebalance_index(prices, cfg.rebalance)
	momentum_reb = momentum.reindex(rebalance_dates).dropna()

	pick = momentum_reb.idxmax(axis=1)
	pick_df = pick.to_frame("position_raw")
	pick_df.loc[
		momentum_reb.max(axis=1) <= 0,
		"position_raw",
	] = cfg.cash_symbol

	# Decision signal at T (using T close), execution at T+1 to avoid look-ahead bias.
	signal_series = pick_df["position_raw"].reindex(daily_ret.index).ffill()
	signal_series = signal_series.fillna(cfg.cash_symbol)

	exec_position = signal_series.shift(1).fillna(cfg.cash_symbol)
	assert len(exec_position) == len(signal_series), (
		f"Length mismatch after shift: signal={len(signal_series)}, exec={len(exec_position)}"
	)

	# --- Fee calculation ---
	# Differentiate between asset↔asset and cash↔asset transitions:
	#   CASH → GOLD:  1 buy   → 1x fee
	#   GOLD → CASH:  1 sell  → 1x fee
	#   GOLD → EQUITY: 1 sell + 1 buy → 2x fee
	prev = exec_position.shift(1).fillna(cfg.cash_symbol)
	same = (exec_position == prev).astype(float)
	both_assets = (
		((exec_position != cfg.cash_symbol) & (prev != cfg.cash_symbol)).astype(float)
	)
	# When both are assets and they're different → 2x fee
	asset_swap = both_assets * (1 - same)
	# When one side is cash → 1x fee
	cash_touch = (1 - same) * (1 - asset_swap)

	turnover = (1 - same)  # any change triggers a trade
	fee = turnover * (cfg.fee_bps / 10000.0) * (1.0 + asset_swap)
	# 1x for cash edges, 2x for asset swaps

	# Cash return: convert annualised rate to daily equivalent
	cash_daily = (1 + cfg.cash_annual_return) ** (1 / 252) - 1

	gold_ret = daily_ret["GOLD"].rename("gold_ret")
	equity_ret = daily_ret["EQUITY"].rename("equity_ret")

	alloc_gold = (exec_position == "GOLD").astype(float)
	alloc_equity = (exec_position == "EQUITY").astype(float)
	alloc_cash = (exec_position == cfg.cash_symbol).astype(float)

	portfolio_ret = alloc_gold * gold_ret + alloc_equity * equity_ret + alloc_cash * cash_daily
	portfolio_ret = portfolio_ret - fee

	return pd.DataFrame(
		{
			"signal": signal_series,
			"position": exec_position,
			"gold_ret": gold_ret,
			"equity_ret": equity_ret,
			"portfolio_ret": portfolio_ret,
		}
	)


def performance_summary(returns: pd.Series, trading_days_per_year: int = 252) -> dict:
	"""Compute simple performance metrics from daily returns.

	Args:
		returns: Daily return series.
		trading_days_per_year: Annualisation factor (default 252 for daily equity data).
	"""

	if returns.empty:
		return {}

	daily_ret = returns
	cum_curve = (1 + daily_ret).cumprod()
	total_days = (cum_curve.index[-1] - cum_curve.index[0]).days
	years = total_days / 365.25 if total_days > 0 else 0

	last_val = cum_curve.iloc[-1]
	# Guard against negative or zero NAV (can happen in extreme drawdowns)
	if last_val > 0 and years > 0:
		cagr = last_val ** (1 / years) - 1
	else:
		cagr = -1.0 if last_val <= 0 else np.nan

	vol = daily_ret.std() * np.sqrt(trading_days_per_year)
	sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(trading_days_per_year) if daily_ret.std() != 0 else np.nan
	dd = (cum_curve / cum_curve.cummax() - 1).min()

	# Calmar ratio = CAGR / |max drawdown|
	calmar = cagr / abs(dd) if (dd != 0 and not np.isnan(cagr)) else np.nan

	return {
		"cagr": cagr,
		"vol": vol,
		"sharpe": sharpe,
		"max_drawdown": dd,
		"calmar": calmar,
		"last_value": last_val,
	}
