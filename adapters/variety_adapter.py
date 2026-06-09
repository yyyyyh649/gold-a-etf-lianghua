"""Adapter interface for connecting strategies to live trading venues.

This remains a placeholder until live trading is enabled.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class VarietyAdapter(ABC):
	"""Abstract base for instrument adapters (futures, ETFs, etc.)."""

	@abstractmethod
	def fetch_quote(self, symbol: str) -> Dict[str, Any]:
		"""Fetch latest quote for a symbol."""

	@abstractmethod
	def fetch_positions(self) -> List[Dict[str, Any]]:
		"""Get current portfolio holdings. Required for rebalancing.

		Each dict in the list should contain at minimum:
			{"symbol": str, "qty": float, "market_value": float}
		"""

	@abstractmethod
	def place_order(
		self, symbol: str, qty: float, side: str, order_type: str = "market"
	) -> Dict[str, Any]:
		"""Submit an order. Implementation is venue-specific."""


class DummyAdapter(VarietyAdapter):
	"""No-op adapter used during research and backtesting."""

	def fetch_quote(self, symbol: str) -> Dict[str, Any]:
		return {"symbol": symbol, "status": "stub", "price": None}

	def fetch_positions(self) -> List[Dict[str, Any]]:
		return []

	def place_order(
		self, symbol: str, qty: float, side: str, order_type: str = "market"
	) -> Dict[str, Any]:
		return {
			"symbol": symbol,
			"qty": qty,
			"side": side,
			"order_type": order_type,
			"status": "stub",
		}
