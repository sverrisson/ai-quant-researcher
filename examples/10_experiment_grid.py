"""Example 10 — parameter grid harness for cross-sectional momentum.

Runs a full experiment grid over:
    lookback x skip x vol-filter x asset/timeframe scenario

Then prints a ranked summary table so you can quickly compare variants.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ai_quant_lab.backtest import (
    BacktestConfig,
    BarSchedule,
    long_short_quantile_portfolio,
    vectorized_portfolio_backtest,
)
from ai_quant_lab.features.cross_sectional import cross_sectional_momentum
from ai_quant_lab.features.library import realized_volatility
from ai_quant_lab.validation.deflated_sharpe import deflated_sharpe


@dataclass(frozen=True)
class Scenario:
    """Synthetic universe setup used for an experiment slice."""

    name: str
    schedule: BarSchedule
    n_bars: int
    n_assets: int
    seed: int
    cost_bps: float
    common_vol: float
    idio_vol: float
    persistence: float


def synthetic_universe(config: Scenario) -> pd.DataFrame:
    """Create a synthetic price panel with mild momentum persistence."""
    rng = np.random.default_rng(config.seed)
    common = rng.normal(0.0002, config.common_vol, config.n_bars)[:, None]
    idio = rng.normal(0.0, config.idio_vol, (config.n_bars, config.n_assets))
    shocks = common + idio

    for t in range(1, config.n_bars):
        shocks[t] += config.persistence * shocks[t - 1]

    index = _make_index(config.schedule, config.n_bars)
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(shocks, axis=0)),
        index=index,
        columns=[f"A{i:03d}" for i in range(config.n_assets)],
    )


def _make_index(schedule: BarSchedule, n_bars: int) -> pd.DatetimeIndex:
    if schedule.interval == "1d":
        return pd.bdate_range(end="2026-01-01", periods=n_bars)
    if schedule.trading_days_per_year == 365:
        return pd.date_range(end="2026-01-01", periods=n_bars, freq="h")
    return pd.date_range(end="2026-01-01", periods=n_bars, freq="h")


def _vol_filter_positions(
    base_positions: pd.DataFrame,
    prices: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """Apply a volatility filter/regime gate to base long-short positions."""
    if mode == "none":
        return base_positions

    rv = pd.DataFrame(
        {asset: realized_volatility(prices[asset], 21) for asset in prices.columns}
    )

    if mode == "inverse_vol_weight":
        weighted = base_positions * (1.0 / rv).clip(upper=20.0)
        return _renormalize_dollar_neutral(weighted)

    if mode == "low_vol_universe":
        low_vol_mask = rv.rank(axis=1, pct=True) <= 0.6
        filtered = base_positions.where(low_vol_mask, 0.0)
        return _renormalize_dollar_neutral(filtered)

    if mode == "market_regime":
        market_proxy = prices.mean(axis=1)
        market_rv = realized_volatility(market_proxy, 21)
        threshold = market_rv.rolling(63, min_periods=20).median()
        allowed = market_rv <= threshold
        filtered = base_positions.where(allowed, 0.0)
        return _renormalize_dollar_neutral(filtered)

    raise ValueError(f"Unknown vol filter mode: {mode}")


def _renormalize_dollar_neutral(positions: pd.DataFrame) -> pd.DataFrame:
    long_side = positions.clip(lower=0.0)
    short_side = positions.clip(upper=0.0)

    long_gross = long_side.sum(axis=1).replace(0.0, np.nan)
    short_gross = short_side.abs().sum(axis=1).replace(0.0, np.nan)

    long_scaled = long_side.div(long_gross, axis=0).fillna(0.0) * 0.5
    short_scaled = short_side.div(short_gross, axis=0).fillna(0.0) * 0.5

    return long_scaled + short_scaled


def run_single_experiment(
    prices: pd.DataFrame,
    schedule: BarSchedule,
    lookback: int,
    skip: int,
    vol_filter: str,
    cost_bps: float,
    n_trials: int,
) -> dict[str, float | str | bool]:
    signal = cross_sectional_momentum(prices, lookback=lookback, skip=skip)
    base_positions = long_short_quantile_portfolio(signal, long_quantile=0.8, short_quantile=0.2)
    positions = _vol_filter_positions(base_positions, prices, vol_filter)

    config = BacktestConfig.from_schedule(schedule, cost_bps=cost_bps, execution_lag=1)
    backtest = vectorized_portfolio_backtest(
        positions.fillna(0.0),
        prices.pct_change().fillna(0.0),
        config=config,
    )
    dsr = deflated_sharpe(backtest.returns, n_trials=n_trials, annualization=config.annualization)

    return {
        "lookback": str(lookback),
        "skip": str(skip),
        "vol_filter": vol_filter,
        "annualization": float(config.annualization),
        "sharpe": float(backtest.metrics["sharpe_ratio"]),
        "annual_return": float(backtest.metrics["annualized_return"]),
        "annual_vol": float(backtest.metrics["annualized_volatility"]),
        "max_drawdown": float(backtest.metrics["max_drawdown"]),
        "annual_turnover": float(backtest.metrics["annual_turnover"]),
        "dsr_pvalue": float(dsr.pvalue),
        "passes_dsr_5pct": bool(dsr.pvalue < 0.05),
    }


def build_grid_results(
    scenarios: Iterable[Scenario],
    lookbacks: list[int],
    skips: list[int],
    vol_filters: list[str],
) -> pd.DataFrame:
    total_trials = len(list(scenarios)) * len(lookbacks) * len(skips) * len(vol_filters)

    rows: list[dict[str, float | str | bool]] = []
    for scenario in scenarios:
        prices = synthetic_universe(scenario)
        for lookback in lookbacks:
            for skip in skips:
                for vol_filter in vol_filters:
                    row = run_single_experiment(
                        prices=prices,
                        schedule=scenario.schedule,
                        lookback=lookback,
                        skip=skip,
                        vol_filter=vol_filter,
                        cost_bps=scenario.cost_bps,
                        n_trials=total_trials,
                    )
                    row["scenario"] = scenario.name
                    rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["passes_dsr_5pct", "sharpe", "annual_return", "dsr_pvalue"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    results["rank"] = np.arange(1, len(results) + 1)
    return results


def main() -> None:
    scenarios = [
        Scenario(
            name="equities_daily",
            schedule=BarSchedule.daily(),
            n_bars=1800,
            n_assets=30,
            seed=101,
            cost_bps=10.0,
            common_vol=0.007,
            idio_vol=0.012,
            persistence=0.045,
        ),
        Scenario(
            name="equities_hourly",
            schedule=BarSchedule.hourly(),
            n_bars=3500,
            n_assets=30,
            seed=102,
            cost_bps=4.0,
            common_vol=0.002,
            idio_vol=0.004,
            persistence=0.025,
        ),
        Scenario(
            name="crypto_hourly",
            schedule=BarSchedule.crypto_hourly(),
            n_bars=5000,
            n_assets=25,
            seed=103,
            cost_bps=12.0,
            common_vol=0.003,
            idio_vol=0.006,
            persistence=0.03,
        ),
    ]

    lookbacks = [5, 10, 21, 63, 126]
    skips = [1, 5]
    vol_filters = ["none", "inverse_vol_weight", "low_vol_universe", "market_regime"]

    results = build_grid_results(scenarios, lookbacks, skips, vol_filters)

    display_cols = [
        "rank",
        "scenario",
        "lookback",
        "skip",
        "vol_filter",
        "sharpe",
        "annual_return",
        "annual_vol",
        "max_drawdown",
        "annual_turnover",
        "dsr_pvalue",
        "passes_dsr_5pct",
    ]

    print()
    print("Full experiment grid complete")
    print(f"  scenarios={len(scenarios)}")
    print(f"  lookbacks={lookbacks}")
    print(f"  skips={skips}")
    print(f"  vol_filters={vol_filters}")
    print(f"  total_trials={len(results)}")
    print()
    print("Top 25 variants (ranked by DSR pass, then Sharpe, return, p-value):")
    print(results[display_cols].head(25).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()
