"""Deterministic fallback agents for offline runs.

These keep the research loop runnable when no Anthropic API key is configured.
They are intentionally simple and conservative: the goal is to exercise the
full pipeline locally, not to simulate Claude.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

def _hypothesis_bank() -> list[StrategyHypothesis]:
    from ai_quant_lab.agents.hypothesis import StrategyHypothesis

    return [
        StrategyHypothesis(
            hypothesis_id="mom_21",
            title="21-day momentum",
            rationale="Short-horizon trend following on a liquid daily tape.",
            spec={
                "signal": "21-day lagged return",
                "direction": "both",
                "holding_period_bars": 21,
                "rebalance_bars": 5,
                "position_bounds": [-1.0, 1.0],
            },
            expected_sharpe_range=(0.3, 0.8),
            works_in_regime="trending markets with persistent drift",
            breaks_in_regime="choppy mean-reverting tapes",
        ),
        StrategyHypothesis(
            hypothesis_id="meanrev_21",
            title="21-day mean reversion",
            rationale="Extremes in recent moves can mean-revert on a daily horizon.",
            spec={
                "signal": "21-day rolling z-score",
                "direction": "both",
                "holding_period_bars": 5,
                "rebalance_bars": 1,
                "position_bounds": [-1.0, 1.0],
            },
            expected_sharpe_range=(0.2, 0.7),
            works_in_regime="range-bound markets with overreaction",
            breaks_in_regime="strong directional breakouts",
        ),
        StrategyHypothesis(
            hypothesis_id="vol_target",
            title="Volatility-scaled trend",
            rationale="Trend exposure scaled down in noisy regimes can stabilize returns.",
            spec={
                "signal": "21-day lagged return divided by 21-day realized volatility",
                "direction": "both",
                "holding_period_bars": 21,
                "rebalance_bars": 5,
                "position_bounds": [-1.0, 1.0],
            },
            expected_sharpe_range=(0.3, 0.9),
            works_in_regime="persistent trends with regime shifts in volatility",
            breaks_in_regime="flat markets with unstable volatility",
        ),
    ]


@dataclass
class OfflineHypothesisAgent:
    """Cycles through a small deterministic hypothesis bank."""

    call_count: int = 0

    def propose(self, market_description, prior_trials_summary, **kwargs) -> StrategyHypothesis:
        hypotheses = _hypothesis_bank()
        hypothesis = hypotheses[self.call_count % len(hypotheses)]
        self.call_count += 1
        return hypothesis


@dataclass
class OfflineCriticAgent:
    """Passes the simpler offline hypotheses and kills obvious nonsense."""

    call_count: int = 0

    def review(self, hypothesis: StrategyHypothesis) -> CriticVerdict:
        from ai_quant_lab.agents.critic import CriticVerdict

        self.call_count += 1
        title = hypothesis.title.lower()
        suspicious = any(word in title for word in ("lookahead", "future", "leak"))
        return CriticVerdict(
            passes=not suspicious,
            reasoning=(
                "Looks like a standard OHLCV-only idea."
                if not suspicious
                else "Rejecting because the proposal appears to use forward-looking information."
            ),
            kill_reasons=([] if not suspicious else ["lookahead"]),
        )


@dataclass
class OfflineCodeAgent:
    """Render a conservative strategy implementation from the hypothesis."""

    call_count: int = 0

    def render(self, hypothesis: StrategyHypothesis) -> CodeArtifact:
        from ai_quant_lab.agents.code import CodeArtifact

        self.call_count += 1
        title = hypothesis.title.lower()

        if "mean reversion" in title:
            source = """
import numpy as np

def strategy(price_data: pd.Series) -> pd.Series:
    lagged = price_data.shift(1)
    mean = lagged.rolling(21, min_periods=21).mean()
    std = lagged.rolling(21, min_periods=21).std(ddof=1)
    zscore = (lagged - mean) / std
    return (-zscore.clip(-2, 2) / 2.0).clip(-1, 1).fillna(0.0)
""".strip()
        elif "volatility-scaled" in title or "volatility" in title:
            source = """
import numpy as np

def strategy(price_data: pd.Series) -> pd.Series:
    lagged = price_data.shift(1)
    momentum = lagged / lagged.shift(21) - 1.0
    vol = lagged.pct_change().rolling(21, min_periods=21).std(ddof=1)
    signal = momentum / vol.replace(0.0, np.nan)
    return signal.clip(-2, 2).fillna(0.0).clip(-1, 1)
""".strip()
        else:
            source = """
import numpy as np

def strategy(price_data: pd.Series) -> pd.Series:
    lagged = price_data.shift(1)
    momentum = lagged / lagged.shift(21) - 1.0
    return np.sign(momentum).clip(-1, 1).fillna(0.0)
""".strip()

        return CodeArtifact(source=source)


@dataclass
class OfflineRiskAgent:
    """Conservative default risk assessment for offline runs."""

    call_count: int = 0

    def assess(
        self,
        hypothesis_title: str,
        positions: pd.Series,
        returns: pd.Series,
        volatility: pd.Series | None = None,
    ) -> RiskAssessment:
        from ai_quant_lab.agents.risk import RiskAssessment

        self.call_count += 1
        return RiskAssessment(
            risk_score=5,
            concerns=["Offline fallback mode used; size conservatively."],
            size_recommendation=0.5,
        )