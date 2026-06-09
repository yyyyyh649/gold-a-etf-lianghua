# 数据获取模块
# 功能：
# 1. 获取黄金期货历史数据 (GC=F from yfinance)
# 2. 获取A股指数或ETF历史数据 (使用 akshare，例如 000001.SH 上证指数, 510300.SH 沪深300ETF)
# 3. 统一返回 pandas DataFrame，列名：Date, Open, High, Low, Close, Volume
# 4. 支持指定开始日期和结束日期
# 5. 添加保存到 CSV 的函数，路径为 data/ 目录下
# 6. 添加基本的错误处理和数据清洗

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_CACHE = DATA_DIR / "gold_gc_f.csv"


def _to_datetime(date_str: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format."""

    return datetime.strptime(date_str, "%Y-%m-%d")


def _clean_price_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Standardize price dataframe columns and ordering."""

    if df.empty:
        raise ValueError(f"No data returned for symbol {symbol}")

    df = df.rename(
        columns={
            "日期": "Date",
            "date": "Date",
            "开盘": "Open",
            "open": "Open",
            "最高": "High",
            "high": "High",
            "最低": "Low",
            "low": "Low",
            "收盘": "Close",
            "close": "Close",
            "结算价": "Close",
            "成交量": "Volume",
            "volume": "Volume",
            "成交量(手)": "Volume",
        }
    )

    # Normalize to date (naive) to reduce timezone drift across sources.
    # yfinance returns UTC-aware timestamps; akshare returns naive timestamps.
    # Convert any aware timestamp to Asia/Shanghai first, then strip tz,
    # so dates are anchored to China local time and never drift by 1 day.
    dt_col = pd.to_datetime(df["Date"])
    if dt_col.dt.tz is not None:
        dt_col = dt_col.dt.tz_convert("Asia/Shanghai")
    df["Date"] = dt_col.dt.tz_localize(None).dt.normalize()
    if "Volume" not in df.columns:
        df["Volume"] = 0

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} for symbol {symbol}")

    df = df[required_cols]
    df[["Open", "High", "Low", "Close", "Volume"]] = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Date")
    df["Symbol"] = symbol
    return df.reset_index(drop=True)


def _filter_date_range(df: pd.DataFrame, start: datetime, end: Optional[datetime]) -> pd.DataFrame:
    mask = (df["Date"] >= start) & ((df["Date"] <= end) if end else True)
    return df.loc[mask].reset_index(drop=True)


def _fetch_gold_from_akshare(start: datetime, end: Optional[datetime]) -> Optional[pd.DataFrame]:
    if not hasattr(ak, "futures_foreign_commodity_hist"):
        return None

    ak_start = start.strftime("%Y%m%d")
    ak_end = end.strftime("%Y%m%d") if end else datetime.today().strftime("%Y%m%d")
    candidates = [
        ("GC", "COMEX"),
        ("GOLD", "COMEX"),
        ("GC", "NYMEX"),
    ]

    for symbol, market in candidates:
        try:
            raw = ak.futures_foreign_commodity_hist(
                symbol=symbol,
                market=market,
                start_date=ak_start,
                end_date=ak_end,
            )
            if raw is None or raw.empty:
                continue

            cleaned = _clean_price_dataframe(raw, symbol="GC=F")
            cleaned = _filter_date_range(cleaned, start, end)
            if not cleaned.empty:
                return cleaned
        except Exception as exc:
            logger.warning("akshare %s/%s failed: %s", symbol, market, exc)
            continue

    return None


def _fetch_gold_from_stooq(start: datetime, end: Optional[datetime]) -> Optional[pd.DataFrame]:
    try:
        url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
        raw = pd.read_csv(url)
        if raw is None or raw.empty:
            return None
        cleaned = _clean_price_dataframe(raw, symbol="GC=F")
        cleaned = _filter_date_range(cleaned, start, end)
        return cleaned if not cleaned.empty else None
    except Exception as exc:
        logger.warning("Stooq gold fetch failed: %s", exc)
        return None


def fetch_gold_futures(
    start_date: str,
    end_date: Optional[str] = None,
    retries: int = 3,
    backoff_seconds: int = 5,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch COMEX gold futures with akshare first, fallback to stooq/Yahoo."""

    start = _to_datetime(start_date)
    end = _to_datetime(end_date) if end_date else None

    if use_cache and GOLD_CACHE.exists():
        cached = pd.read_csv(GOLD_CACHE, parse_dates=["Date"])
        cached = _clean_price_dataframe(cached, symbol="GC=F")
        cached_slice = _filter_date_range(cached, start, end)
        if not cached_slice.empty:
            return cached_slice

    # 1) Try akshare COMEX GC 合约
    ak_df = _fetch_gold_from_akshare(start, end)
    if ak_df is not None and not ak_df.empty:
        if use_cache:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            ak_df.to_csv(GOLD_CACHE, index=False)
        return ak_df

    # 2) Try stooq spot gold (XAUUSD)
    stooq_df = _fetch_gold_from_stooq(start, end)
    if stooq_df is not None and not stooq_df.empty:
        if use_cache:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            stooq_df.to_csv(GOLD_CACHE, index=False)
        return stooq_df

    # 3) Fallback to Yahoo Finance with retry/backoff
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                "GC=F",
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:  # yfinance raises YFRateLimitError on throttling
            last_exc = exc
            data = pd.DataFrame()

        if not data.empty:
            break

        if attempt < retries:
            time.sleep(backoff_seconds * attempt)

    if data.empty:
        if use_cache and GOLD_CACHE.exists():
            cached = pd.read_csv(GOLD_CACHE, parse_dates=["Date"])
            cached = _clean_price_dataframe(cached, symbol="GC=F")
            cached_slice = _filter_date_range(cached, start, end)
            if not cached_slice.empty:
                return cached_slice

        msg = "Failed to download GC=F from Yahoo Finance; likely rate limited."
        if last_exc:
            msg += f" Last error: {last_exc}"
        raise RuntimeError(msg)

    data = data.reset_index().rename(columns={"Adj Close": "Close"})
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
    cleaned = _clean_price_dataframe(data, symbol="GC=F")

    if use_cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(GOLD_CACHE, index=False)

    return cleaned


def fetch_a_share_index_or_etf(
    symbol: str, start_date: str, end_date: Optional[str] = None, is_etf: Optional[bool] = None
) -> pd.DataFrame:
    """Fetch A-share index or ETF using akshare.

    Args:
        symbol: e.g., "000001" for 上证指数 or "510300" for 沪深300ETF. Suffix (.SH/.SZ) is optional.
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD, defaults to today
        is_etf: force ETF mode when True, index mode when False, auto-detect when None
    """

    code = symbol.replace(".SH", "").replace(".SZ", "").replace(".sh", "").replace(".sz", "")
    start = _to_datetime(start_date)
    end_dt = _to_datetime(end_date) if end_date else datetime.today()
    start_str, end_str = start.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")

    if is_etf is None:
        is_etf = code.startswith(("5", "1"))

    if is_etf:
        raw = ak.fund_etf_hist_em(
            symbol=code,
            start_date=start_str,
            end_date=end_str,
            adjust="qfq",
        )
    else:
        raw = ak.index_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_str,
            end_date=end_str,
            adjust="",
        )

    return _clean_price_dataframe(raw, symbol=code)


def save_data(df: pd.DataFrame, filename: str) -> Path:
    """Save dataframe to the data/ directory as CSV."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return path