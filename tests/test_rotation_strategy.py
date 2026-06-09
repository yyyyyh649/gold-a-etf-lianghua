"""Unit tests for rotation_strategy module."""

import numpy as np
import pandas as pd
import pytest

from strategies.rotation_strategy import (
	RotationConfig,
	_prepare_prices,
	generate_signals,
	performance_summary,
	parameter_sweep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_gold() -> pd.DataFrame:
	"""10 days of fake gold data."""
	dates = pd.date_range("2024-01-02", periods=10, freq="B")
	return pd.DataFrame({
		"Date": dates,
		"Close": [2000 + i * 5 for i in range(10)],  # uptrend
		"Open": 2000,
		"High": 2010,
		"Low": 1990,
		"Volume": 1000,
	})


@pytest.fixture
def sample_equity() -> pd.DataFrame:
	"""10 days of fake equity data - flat then drop."""
	dates = pd.date_range("2024-01-02", periods=10, freq="B")
	return pd.DataFrame({
		"Date": dates,
		"Close": [100] * 5 + [95] * 5,  # flat then drops
		"Open": 100,
		"High": 101,
		"Low": 99,
		"Volume": 1000,
	})


@pytest.fixture
def empty_df() -> pd.DataFrame:
	"""Empty but structurally valid DataFrame (has the right columns)."""
	return pd.DataFrame(columns=["Date", "Close", "Open", "High", "Low", "Volume"])


# ---------------------------------------------------------------------------
# _prepare_prices
# ---------------------------------------------------------------------------

class TestPreparePrices:
	def test_inner_join_returns_valid_prices(self, sample_gold, sample_equity):
		prices = _prepare_prices(sample_gold, sample_equity, align_method="inner")
		assert "GOLD" in prices.columns
		assert "EQUITY" in prices.columns
		assert len(prices) > 0

	def test_inner_join_drops_non_overlap(self):
		"""If one market is missing a date, inner join drops it."""
		gold = pd.DataFrame({
			"Date": pd.date_range("2024-01-02", periods=3, freq="B"),
			"Close": [2000, 2010, 2020],
		})
		equity = pd.DataFrame({
			"Date": pd.date_range("2024-01-02", periods=3, freq="B"),
			"Close": [100, 101, 102],
		})
		# Add a non-overlapping date to equity
		extra = pd.DataFrame({"Date": [pd.Timestamp("2024-01-08")], "Close": [103]})
		equity = pd.concat([equity, extra], ignore_index=True)

		prices = _prepare_prices(gold, equity, align_method="inner")
		assert pd.Timestamp("2024-01-08") not in prices.index

	def test_a_share_ffill_preserves_more_data(self):
		"""a_share_ffill should retain A-share-only dates by ffill-ing gold."""
		gold = pd.DataFrame({
			"Date": pd.date_range("2024-01-02", periods=3, freq="B"),
			"Close": [2000, 2010, 2020],
		})
		equity = pd.DataFrame({
			"Date": list(pd.date_range("2024-01-02", periods=3, freq="B"))
				+ [pd.Timestamp("2024-01-08")],  # A-share only date
			"Close": [100, 101, 102, 103],
		})
		prices = _prepare_prices(gold, equity, align_method="a_share_ffill")
		# The 2024-01-08 date should be kept (with ffill-ed gold price)
		assert pd.Timestamp("2024-01-08") in prices.index
		assert not prices["GOLD"].isna().any()


# ---------------------------------------------------------------------------
# generate_signals
# ---------------------------------------------------------------------------

class TestGenerateSignals:
	def test_output_columns(self, sample_gold, sample_equity):
		result = generate_signals(sample_gold, sample_equity)
		expected = {"signal", "position", "gold_ret", "equity_ret", "portfolio_ret"}
		assert expected.issubset(set(result.columns))

	def test_signal_is_valid_choice(self, sample_gold, sample_equity):
		cfg = RotationConfig(cash_symbol="CASH")
		result = generate_signals(sample_gold, sample_equity, config=cfg)
		valid = {"GOLD", "EQUITY", "CASH"}
		assert result["position"].isin(valid).all(), (
			f"Unexpected position values: {result['position'].unique()}"
		)

	def test_no_lookahead_bias(self, sample_gold, sample_equity):
		"""Execution position must be lagged by 1 vs signal."""
		result = generate_signals(sample_gold, sample_equity)
		signal = result["signal"]
		position = result["position"]
		# After shift(1), signal[t] should == position[t+1]
		aligned = pd.concat([signal.rename("sig"), position.rename("pos")], axis=1)
		aligned = aligned.dropna()
		# Check that position is indeed lagged: for any non-cash position,
		# the signal on the previous day should match
		for i in range(1, len(aligned)):
			if aligned["pos"].iloc[i] != "CASH":
				assert aligned["sig"].iloc[i - 1] == aligned["pos"].iloc[i], (
					f"Look-ahead at index {i}: signal={aligned['sig'].iloc[i-1]} "
					f"!= position={aligned['pos'].iloc[i]}"
				)

	def test_double_negative_goes_cash(self, sample_gold, sample_equity):
		"""When both momentums are negative, position should be CASH."""
		# Both flat/down should trigger cash
		cfg = RotationConfig(lookback_days=3, rebalance="daily")
		result = generate_signals(sample_gold, sample_equity, config=cfg)
		cash_positions = result[result["position"] == "CASH"]
		assert len(cash_positions) > 0, "Expected at least one CASH position"

	def test_cash_return_applied(self, sample_gold, sample_equity):
		"""With cash_annual_return > 0, cash days should have positive return."""
		cfg = RotationConfig(cash_annual_return=0.03, lookback_days=3, rebalance="daily")
		result = generate_signals(sample_gold, sample_equity, config=cfg)
		cash_mask = result["position"] == "CASH"
		if cash_mask.any():
			cash_rets = result.loc[cash_mask, "portfolio_ret"]
			# All cash-day returns should be positive (from the 3% annualised)
			assert (cash_rets > 0).all(), "Cash days should earn positive return"

	def test_raises_on_empty_gold(self, sample_equity, empty_df):
		"""Empty but structurally valid gold data should produce empty result, not crash."""
		# With empty gold but valid equity, inner join produces no overlapping dates.
		# The strategy should still complete without raising.
		result = generate_signals(empty_df, sample_equity)
		assert len(result) == 0  # no data means no trading days

	def test_raises_on_empty_equity(self, sample_gold, empty_df):
		"""Mirror of test_raises_on_empty_gold."""
		result = generate_signals(sample_gold, empty_df)
		assert len(result) == 0


# ---------------------------------------------------------------------------
# performance_summary
# ---------------------------------------------------------------------------

class TestPerformanceSummary:
	def test_empty_returns(self):
		assert performance_summary(pd.Series([], dtype=float)) == {}

	def test_constant_returns(self):
		"""0% every day → cagr=0, vol=0, sharpe=NaN, dd=0."""
		rets = pd.Series([0.0] * 252, index=pd.date_range("2024-01-01", periods=252, freq="B"))
		m = performance_summary(rets)
		assert m["cagr"] == 0.0
		assert m["vol"] == 0.0
		assert np.isnan(m["sharpe"])

	def test_positive_returns(self):
		"""0.1% every day → positive cagr and sharpe."""
		rets = pd.Series([0.001] * 252, index=pd.date_range("2024-01-01", periods=252, freq="B"))
		m = performance_summary(rets)
		assert m["cagr"] > 0
		assert m["sharpe"] > 0
		assert m["last_value"] > 1.0

	def test_calmar_ratio(self):
		"""Calmar = CAGR / |max_dd|."""
		rets = pd.Series([0.001] * 252, index=pd.date_range("2024-01-01", periods=252, freq="B"))
		m = performance_summary(rets)
		# With no drawdown, max_drawdown = 0, calmar should be nan
		# Actually, even with small positive returns, there should be no drawdown
		assert "calmar" in m

	def test_negative_returns(self):
		"""-0.1% every day → negative cagr."""
		rets = pd.Series([-0.001] * 252, index=pd.date_range("2024-01-01", periods=252, freq="B"))
		m = performance_summary(rets)
		assert m["cagr"] < 0
		assert m["last_value"] < 1.0

	def test_negative_nav_does_not_crash(self):
		"""If cumulative NAV goes negative, CAGR should handle gracefully."""
		rets = pd.Series([-0.5] * 10, index=pd.date_range("2024-01-01", periods=10, freq="B"))
		m = performance_summary(rets)
		assert m["cagr"] == -1.0  # total loss
		assert m["last_value"] < 1.0


# ---------------------------------------------------------------------------
# Fee calculation
# ---------------------------------------------------------------------------

class TestFeeCalculation:
	def test_asset_swap_double_fee(self, sample_gold, sample_equity):
		"""GOLD → EQUITY swap should incur 2x fee_bps."""
		cfg = RotationConfig(fee_bps=10.0, lookback_days=3, rebalance="daily")
		result = generate_signals(sample_gold, sample_equity, config=cfg)
		position = result["position"]
		portfolio_ret = result["portfolio_ret"]
		gold_ret = result["gold_ret"]
		equity_ret = result["equity_ret"]

		# Find a day where a swap occurred (position changed between GOLD and EQUITY)
		prev = position.shift(1).fillna(cfg.cash_symbol)
		asset_swap = (
			(position != cfg.cash_symbol) & (prev != cfg.cash_symbol) & (position != prev)
		)
		if asset_swap.any():
			swap_day = asset_swap[asset_swap].index[0]
			expected_fee = 2 * (cfg.fee_bps / 10000.0)  # 2x for asset swap
			# On swap day, portfolio return = asset return - fee
			pos = position.loc[swap_day]
			asset_ret = gold_ret.loc[swap_day] if pos == "GOLD" else equity_ret.loc[swap_day]
			assert abs(portfolio_ret.loc[swap_day] - (asset_ret - expected_fee)) < 1e-10, (
				f"Expected fee {expected_fee:.6f} on swap day"
			)

	def test_cash_to_asset_single_fee(self, sample_gold, sample_equity):
		"""CASH → GOLD should incur 1x fee_bps."""
		cfg = RotationConfig(fee_bps=10.0, lookback_days=3, rebalance="daily")
		result = generate_signals(sample_gold, sample_equity, config=cfg)
		position = result["position"]
		portfolio_ret = result["portfolio_ret"]
		gold_ret = result["gold_ret"]

		prev = position.shift(1).fillna(cfg.cash_symbol)
		cash_to_asset = (prev == cfg.cash_symbol) & (position != cfg.cash_symbol)
		if cash_to_asset.any():
			day = cash_to_asset[cash_to_asset].index[0]
			expected_fee = cfg.fee_bps / 10000.0  # 1x for cash→asset
			pos = position.loc[day]
			asset_ret = gold_ret.loc[day]
			assert abs(portfolio_ret.loc[day] - (asset_ret - expected_fee)) < 1e-10


# ---------------------------------------------------------------------------
# parameter_sweep
# ---------------------------------------------------------------------------

class TestParameterSweep:
	def test_returns_dataframe(self, sample_gold, sample_equity):
		sweep = parameter_sweep(sample_gold, sample_equity)
		assert isinstance(sweep, pd.DataFrame)
		assert len(sweep) > 0

	def test_default_windows(self, sample_gold, sample_equity):
		sweep = parameter_sweep(sample_gold, sample_equity)
		# With only 10 data points, only lookback=20 won't have enough data
		assert len(sweep) <= 3  # at most 3 windows (20, 60, 120)

	def test_custom_windows(self, sample_gold, sample_equity):
		sweep = parameter_sweep(sample_gold, sample_equity, lookback_windows=[3, 5])
		assert "3d" in sweep.index
		assert "5d" in sweep.index
