"""Quick validation script for gold vs A-share ETF rotation.

Usage:
    python backtests/validate_strategy.py
    python backtests/validate_strategy.py --sweep   # run multi-window comparison
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from src.data_fetcher import (
	fetch_a_share_index_or_etf,
	fetch_gold_futures,
	save_data,
)
from strategies.rotation_strategy import (
	RotationConfig,
	generate_signals,
	parameter_sweep,
	performance_summary,
	walk_forward_validation,
)
from adapters.variety_adapter import DummyAdapter

VALID_REBALANCE = {"daily", "weekly", "monthly"}

logger = logging.getLogger(__name__)


def compute_turnover(position_series: pd.Series, cash_symbol: str = "CASH") -> float:
	"""Average one-way turnover per trading day.

	A cash↔asset switch is one one-way trade (1x). An asset↔asset switch is two
	(sell the old asset, buy the new one), so it counts as 2x.
	"""
	if len(position_series) < 2:
		return 0.0
	prev = position_series.shift(1)
	changed = (position_series != prev) & prev.notna()
	asset_swap = changed & (position_series != cash_symbol) & (prev != cash_symbol)
	one_way_trades = changed.astype(int) + asset_swap.astype(int)
	return float(one_way_trades.sum()) / len(position_series)


def _trim_warmup(result: pd.DataFrame) -> pd.DataFrame:
	"""Drop the pre-signal warm-up window from a backtest result.

	Before the lookback window fills, both momentum series are NaN and the
	strategy is forced into cash. Keeping those days in the performance metrics
	biases the comparison: the strategy earns nothing while a buy-and-hold
	benchmark is fully invested.
	"""
	if result.empty or not {"mom_gold", "mom_equity"}.issubset(result.columns):
		return result
	valid = result[["mom_gold", "mom_equity"]].notna().all(axis=1)
	if not valid.any():
		return result
	return result.loc[valid.idxmax():]


def run_validation(
	equity_symbol: str = "510300",
	start_date: str = "2015-01-01",
	end_date: Optional[str] = None,
	lookback_days: int = 60,
	rebalance: str = "weekly",
	fee_bps: float = 5.0,
):
	if rebalance not in VALID_REBALANCE:
		raise ValueError(
			f"Invalid rebalance '{rebalance}'. Must be one of {VALID_REBALANCE}"
		)

	try:
		gold = fetch_gold_futures(start_date=start_date, end_date=end_date)
		equity = fetch_a_share_index_or_etf(
			symbol=equity_symbol, start_date=start_date, end_date=end_date
		)
	except Exception as exc:
		logger.error("Data fetch failed: %s", exc)
		raise SystemExit(1) from exc

	cfg = RotationConfig(
		lookback_days=lookback_days, rebalance=rebalance, fee_bps=fee_bps
	)
	result = generate_signals(gold, equity, config=cfg)

	# Drop the momentum warm-up so strategy and benchmark are compared over the
	# same investable window instead of the strategy sitting in cash early on.
	active = _trim_warmup(result)
	if active.empty:
		active = result
	warmup_days = len(result) - len(active)
	if warmup_days > 0:
		print(
			f"跳过预热期 {warmup_days} 个交易日"
			f"（首个有效动量信号: {active.index[0]:%Y-%m-%d}）\n"
		)

	# ---------- benchmark: buy & hold equity ----------
	benchmark_ret = active["equity_ret"]
	benchmark_curve = (1 + benchmark_ret).cumprod().rename("benchmark_curve")

	# ---------- strategy ----------
	portfolio_ret = active["portfolio_ret"]
	strategy_curve = (1 + portfolio_ret).cumprod().rename("strategy_curve")

	metrics = performance_summary(portfolio_ret)
	bench_metrics = performance_summary(benchmark_ret)

	# Turnover
	daily_turnover = compute_turnover(active["position"])
	annual_turnover = daily_turnover * 252

	# Preserve Date in output (warm-up rows keep NaN curves)
	output = result.copy()
	output["strategy_curve"] = strategy_curve
	output["benchmark_curve"] = benchmark_curve

	csv_path = save_data(output, f"backtest_{equity_symbol}.csv")

	print(f"Saved backtest results to {csv_path}\n")

	# ---------- 中文输出 ----------
	cn_labels = {
		"cagr": "年化收益率",
		"vol": "波动率",
		"sharpe": "夏普比率",
		"max_drawdown": "最大回撤",
		"calmar": "卡玛比率",
		"last_value": "期末净值",
	}

	strat_last = metrics.get("last_value")
	bench_last = bench_metrics.get("last_value")
	if isinstance(strat_last, float) and isinstance(bench_last, float):
		excess = (strat_last - bench_last) / bench_last * 100
		print(f"结论：策略累计净值 {strat_last:.2f} vs 基准 {bench_last:.2f}（超额 {excess:+.1f}%）")

	print(f"\n策略指标 vs 基准（买入持有）:")
	print(f"  {'指标':<10} {'策略':>10} {'基准':>10}")
	print(f"  {'-' * 30}")
	for k in ("cagr", "vol", "sharpe", "max_drawdown", "calmar"):
		label = cn_labels.get(k, k)
		sv = metrics.get(k, float("nan"))
		bv = bench_metrics.get(k, float("nan"))
		if k in ("cagr", "vol", "max_drawdown"):
			print(f"  {label:<10} {sv:>9.2%} {bv:>9.2%}")
		else:
			print(f"  {label:<10} {sv:>9.4f} {bv:>9.4f}")

	print(f"\n策略期末净值: {strat_last:.4f}")
	print(f"基准期末净值: {bench_last:.4f}")
	print(f"平均日换手率: {daily_turnover:.2%}")
	print(f"平均年化换手率: {annual_turnover:.1f}x")

	# adapter stub
	adapter = DummyAdapter()
	quote_stub = adapter.fetch_quote(equity_symbol)
	print(f"Adapter (stub) quote example: {quote_stub}")


def run_sweep(
	equity_symbol: str = "510300",
	start_date: str = "2015-01-01",
	end_date: Optional[str] = None,
	rebalance: str = "weekly",
):
	"""Run parameter sweep across multiple lookback windows."""
	try:
		gold = fetch_gold_futures(start_date=start_date, end_date=end_date)
		equity = fetch_a_share_index_or_etf(
			symbol=equity_symbol, start_date=start_date, end_date=end_date
		)
	except Exception as exc:
		logger.error("Data fetch failed: %s", exc)
		raise SystemExit(1) from exc

	sweep = parameter_sweep(gold, equity, rebalance=rebalance)

	print("\n多周期动量参数对比:")
	print(sweep.to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

	# Save sweep results
	path = save_data(sweep.reset_index(), f"sweep_{equity_symbol}.csv")
	print(f"\nSaved sweep results to {path}")


def run_walk_forward(
	equity_symbol: str = "510300",
	start_date: str = "2015-01-01",
	end_date: Optional[str] = None,
	rebalance: str = "weekly",
	train_days: int = 504,
	test_days: int = 126,
):
	"""Run rolling out-of-sample (walk-forward) validation."""
	try:
		gold = fetch_gold_futures(start_date=start_date, end_date=end_date)
		equity = fetch_a_share_index_or_etf(
			symbol=equity_symbol, start_date=start_date, end_date=end_date
		)
	except Exception as exc:
		logger.error("Data fetch failed: %s", exc)
		raise SystemExit(1) from exc

	try:
		wf = walk_forward_validation(
			gold, equity, train_days=train_days, test_days=test_days, rebalance=rebalance
		)
	except ValueError as exc:
		logger.error("Walk-forward validation failed: %s", exc)
		raise SystemExit(1) from exc

	splits = wf["splits"]
	metrics = wf["metrics"]
	bench_metrics = wf["benchmark_metrics"]

	print(
		f"\n样本外滚动验证（walk-forward） 训练={train_days}日 测试={test_days}日 "
		f"调仓={rebalance}"
	)
	print(f"切分数: {len(splits)}，参数分布: {splits['lookback'].value_counts().to_dict()}")
	print("\n切分明细:")
	print(
		splits.to_string(
			index=False,
			formatters={
				"train_start": lambda x: f"{x:%Y-%m-%d}",
				"train_end": lambda x: f"{x:%Y-%m-%d}",
				"test_start": lambda x: f"{x:%Y-%m-%d}",
				"test_end": lambda x: f"{x:%Y-%m-%d}",
				"train_score": lambda x: f"{x:.4f}",
				"test_return": lambda x: f"{x:.2%}",
				"benchmark_return": lambda x: f"{x:.2%}",
			},
		)
	)

	cn_labels = {
		"cagr": "年化收益率",
		"vol": "波动率",
		"sharpe": "夏普比率",
		"max_drawdown": "最大回撤",
		"calmar": "卡玛比率",
	}
	print("\n样本外绩效 vs 基准（买入持有）:")
	print(f"  {'指标':<10} {'策略':>10} {'基准':>10}")
	print(f"  {'-' * 30}")
	for k in ("cagr", "vol", "sharpe", "max_drawdown", "calmar"):
		label = cn_labels.get(k, k)
		sv = metrics.get(k, float("nan"))
		bv = bench_metrics.get(k, float("nan"))
		if k in ("cagr", "vol", "max_drawdown"):
			print(f"  {label:<10} {sv:>9.2%} {bv:>9.2%}")
		else:
			print(f"  {label:<10} {sv:>9.4f} {bv:>9.4f}")

	print(f"\n样本外策略期末净值: {metrics.get('last_value', float('nan')):.4f}")
	print(f"样本外基准期末净值: {bench_metrics.get('last_value', float('nan')):.4f}")

	oos = pd.concat(
		[
			wf["oos_returns"].rename("oos_strategy_ret"),
			wf["oos_benchmark_returns"].rename("oos_benchmark_ret"),
		],
		axis=1,
	)
	path = save_data(oos.reset_index().rename(columns={"index": "Date"}), f"walkforward_{equity_symbol}.csv")
	print(f"\nSaved walk-forward results to {path}")


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

	parser = argparse.ArgumentParser(description="Validate gold/ETF rotation strategy")
	parser.add_argument("--sweep", action="store_true", help="Run multi-window parameter comparison")
	parser.add_argument(
		"--walk-forward",
		action="store_true",
		help="Run rolling out-of-sample (walk-forward) validation",
	)
	parser.add_argument("--train-days", type=int, default=504, help="Walk-forward train window (trading days)")
	parser.add_argument("--test-days", type=int, default=126, help="Walk-forward test window (trading days)")
	parser.add_argument("--equity", default="510300", help="A-share ETF/index code")
	parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
	parser.add_argument("--lookback", type=int, default=60, help="Momentum lookback days")
	parser.add_argument("--rebalance", default="weekly", choices=sorted(VALID_REBALANCE), help="Rebalance frequency")
	parser.add_argument("--fee", type=float, default=5.0, help="One-way fee in bps")
	args = parser.parse_args()

	if args.sweep:
		run_sweep(equity_symbol=args.equity, start_date=args.start, rebalance=args.rebalance)
	elif args.walk_forward:
		run_walk_forward(
			equity_symbol=args.equity,
			start_date=args.start,
			rebalance=args.rebalance,
			train_days=args.train_days,
			test_days=args.test_days,
		)
	else:
		run_validation(
			equity_symbol=args.equity,
			start_date=args.start,
			lookback_days=args.lookback,
			rebalance=args.rebalance,
			fee_bps=args.fee,
		)
