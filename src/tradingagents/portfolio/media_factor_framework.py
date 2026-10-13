"""Frozen v5 heterogeneous experts for point-in-time media experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from tradingagents.portfolio.bar_framework import ResidualPortfolioOptimizer, _softmax
from tradingagents.portfolio.fixed_share_framework import (
    FixedShareConfig,
    MonthlyFixedShareArbitrator,
)
from tradingagents.portfolio.pit_factor_framework import EqualWeightArbitrator


MEDIA_EXPERTS = (
    "trend",
    "meanrev",
    "risk_parity",
    "cross_sectional",
    "fundamental",
    "media_text",
)


@dataclass(frozen=True)
class MediaFactorConfig(FixedShareConfig):
    """Parameters fixed before the v5 external data are inspected."""

    minimum_factor_coverage: float = 0.60
    benchmark_on_empty_allocation: bool = False


class HMATFMedia:
    """Price and point-in-time information experts with fixed-share fusion."""

    def __init__(
        self,
        n_assets: int,
        config: Optional[MediaFactorConfig] = None,
        enabled_experts: Sequence[str] = MEDIA_EXPERTS,
        mechanism: str = "fixed_share",
    ) -> None:
        selected = config or MediaFactorConfig()
        unknown = set(enabled_experts) - set(MEDIA_EXPERTS)
        if unknown:
            raise ValueError(f"unknown media experts: {sorted(unknown)}")
        self.config = selected
        self.n_assets = int(n_assets)
        self.enabled_experts = tuple(enabled_experts)
        if mechanism == "fixed_share":
            self.arbitrator = MonthlyFixedShareArbitrator(selected, MEDIA_EXPERTS)
        elif mechanism == "equal":
            self.arbitrator = EqualWeightArbitrator(MEDIA_EXPERTS)
        else:
            raise ValueError(f"unsupported arbitration mechanism: {mechanism}")
        self.optimizer = ResidualPortfolioOptimizer(selected)
        self.current_weights = np.zeros(self.n_assets)
        self.expert_current = {
            name: np.zeros(self.n_assets) for name in MEDIA_EXPERTS
        }
        self.expert_committed_targets = {name: None for name in MEDIA_EXPERTS}

    def _multifrequency_trend(self, prices: np.ndarray) -> np.ndarray:
        log_prices = np.log(np.asarray(prices, dtype=float))
        daily = np.diff(log_prices, axis=0)
        annual_vol = np.std(daily[-60:], axis=0) * np.sqrt(252) + 1e-8
        signals = []
        for horizon in (21, 63, 126, 252):
            raw = (log_prices[-1] - log_prices[-1 - horizon]) / (
                annual_vol * np.sqrt(horizon / 252)
            )
            signals.append((raw - np.mean(raw)) / (np.std(raw) + 1e-8))
        stacked = np.vstack(signals)
        agreement = np.abs(np.mean(np.sign(stacked), axis=0))
        score = np.mean(stacked, axis=0) * (0.5 + 0.5 * agreement)
        return _softmax(score, self.config.score_temperature)

    def _factor_target(self, score: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if score is None:
            return None
        values = np.asarray(score, dtype=float)
        if values.shape != (self.n_assets,) or not np.all(np.isfinite(values)):
            return None
        return _softmax(values, self.config.score_temperature)

    def market_regime(self, prices: np.ndarray) -> str:
        """Classify the current market using only information available now."""
        log_prices = np.log(np.asarray(prices, dtype=float))
        market_log_price = np.mean(log_prices, axis=1)
        daily = np.diff(market_log_price)
        direction = "up" if market_log_price[-1] >= market_log_price[-64] else "down"
        recent_volatility = float(np.std(daily[-21:]))
        trailing_volatility = float(np.std(daily[-252:]))
        volatility = "high_vol" if recent_volatility >= trailing_volatility else "low_vol"
        return f"{direction}_{volatility}"

    def expert_targets(
        self,
        prices: np.ndarray,
        factor_scores: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        values = np.asarray(prices, dtype=float)
        if len(values) < 253:
            raise ValueError("at least 253 historical observations are required")
        log_prices = np.log(values)
        daily = np.diff(log_prices, axis=0)
        vol = np.std(daily[-60:], axis=0) * np.sqrt(252) + 1e-8
        reversal = -(log_prices[-1] - log_prices[-21]) / vol
        inverse_volatility = 1.0 / vol
        momentum_12_1 = log_prices[-22] - log_prices[-253]
        momentum_6_1 = log_prices[-22] - log_prices[-127]
        low_volatility = -np.std(daily[-60:], axis=0)
        cross_sectional = np.mean(
            np.vstack(
                [
                    _cross_sectional_z(momentum_12_1),
                    _cross_sectional_z(momentum_6_1),
                    _cross_sectional_z(low_volatility),
                ]
            ),
            axis=0,
        )
        factors = factor_scores or {}
        targets: Dict[str, Optional[np.ndarray]] = {
            "trend": self._multifrequency_trend(values),
            "meanrev": _softmax(reversal, self.config.score_temperature),
            "risk_parity": inverse_volatility / inverse_volatility.sum(),
            "cross_sectional": self._factor_target(cross_sectional),
            "fundamental": self._factor_target(factors.get("fundamental")),
            "media_text": self._factor_target(factors.get("media_text")),
        }
        return {
            name: target if name in self.enabled_experts else None
            for name, target in targets.items()
        }

    def rebalance(
        self,
        prices: np.ndarray,
        covariance: np.ndarray,
        benchmark: np.ndarray,
        factor_scores: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, object]:
        benchmark_weights = np.asarray(benchmark, dtype=float)
        targets = self.expert_targets(prices, factor_scores)
        available = {name: target is not None for name, target in targets.items()}
        regime = self.market_regime(prices)
        expert_weights = self.arbitrator.allocate(available, regime=regime)
        names = [name for name in MEDIA_EXPERTS if available[name]]
        allocated_weight = float(sum(expert_weights[name] for name in names))
        if not names or (
            allocated_weight <= 1e-15
            and self.config.benchmark_on_empty_allocation
        ):
            combined = benchmark_weights
        else:
            combined = sum(
                expert_weights[name] * np.asarray(targets[name], dtype=float)
                for name in names
            )
            if allocated_weight > 1e-15:
                combined = combined / allocated_weight
        target = self.optimizer.optimize(
            benchmark_weights, combined, covariance, self.current_weights
        )
        return {
            "target_weights": target,
            "benchmark_weights": benchmark_weights,
            "expert_targets": targets,
            "expert_weights": expert_weights,
            "available": available,
            "regime": regime,
            "used_benchmark_fallback": bool(
                names
                and allocated_weight <= 1e-15
                and self.config.benchmark_on_empty_allocation
            ),
        }

    def record_returns(
        self,
        asset_returns: np.ndarray,
        benchmark_weights: np.ndarray,
        expert_targets: Mapping[str, Optional[np.ndarray]],
        cost_rate: float,
    ) -> Dict[str, float]:
        returns = np.asarray(asset_returns, dtype=float)
        benchmark_return = float(np.asarray(benchmark_weights) @ returns)
        active: Dict[str, float] = {}
        for name, optional_target in expert_targets.items():
            if optional_target is None:
                continue
            target = np.asarray(optional_target, dtype=float)
            committed = self.expert_committed_targets[name]
            changed = committed is None or not np.allclose(
                target, committed, atol=1e-12
            )
            if changed:
                turnover = float(
                    np.sum(np.abs(target - self.expert_current[name]))
                )
                self.expert_current[name] = target.copy()
                self.expert_committed_targets[name] = target.copy()
            else:
                turnover = 0.0
            gross = float(self.expert_current[name] @ returns)
            active[name] = gross - turnover * cost_rate - benchmark_return
            denominator = 1.0 + gross
            if denominator > 0:
                self.expert_current[name] = (
                    self.expert_current[name] * (1.0 + returns) / denominator
                )
        self.arbitrator.update(active)
        return active

    def commit(self, target_weights: np.ndarray) -> None:
        self.current_weights = np.asarray(target_weights, dtype=float).copy()


class HMATFMediaEqualFusion(HMATFMedia):
    def __init__(self, n_assets: int, enabled_experts=MEDIA_EXPERTS) -> None:
        super().__init__(n_assets, enabled_experts=enabled_experts, mechanism="equal")


class HMATFMediaVanillaHedge(HMATFMedia):
    def __init__(self, n_assets: int, enabled_experts=MEDIA_EXPERTS) -> None:
        super().__init__(
            n_assets,
            MediaFactorConfig(fixed_share_alpha=0.0),
            enabled_experts=enabled_experts,
        )


def _cross_sectional_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or len(array) < 2:
        return np.zeros_like(array)
    low, high = np.quantile(array, [0.05, 0.95])
    clipped = np.clip(array, low, high)
    scale = float(np.std(clipped))
    return (
        np.zeros_like(array)
        if scale <= 1e-12
        else (clipped - float(np.mean(clipped))) / scale
    )
