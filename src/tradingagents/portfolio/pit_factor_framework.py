"""Point-in-time heterogeneous experts with fixed-share arbitration.

This v4 research branch leaves every preregistered v1-v3 implementation
untouched.  The expert set is generic: price, accounting, disclosure-event,
and cross-sectional signals share only the constrained portfolio optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from tradingagents.portfolio.bar_framework import ResidualPortfolioOptimizer, _softmax
from tradingagents.portfolio.fixed_share_framework import (
    FixedShareConfig,
    MonthlyFixedShareArbitrator,
)


PIT_EXPERTS = (
    "trend",
    "meanrev",
    "risk_parity",
    "fundamental",
    "event",
    "cross_sectional",
)


@dataclass(frozen=True)
class PITFactorConfig(FixedShareConfig):
    """Frozen v4 portfolio and arbitration parameters."""

    minimum_factor_coverage: float = 0.60


class EqualWeightArbitrator:
    """Static mechanism baseline over whichever experts are available."""

    def __init__(self, expert_names: Sequence[str] = PIT_EXPERTS) -> None:
        self.expert_names = tuple(expert_names)
        self.weights = {name: 0.0 for name in self.expert_names}

    def allocate(self, available: Mapping[str, bool]) -> Dict[str, float]:
        names = [name for name in self.expert_names if available.get(name, False)]
        self.weights = {name: 0.0 for name in self.expert_names}
        if names:
            self.weights.update({name: 1.0 / len(names) for name in names})
        return dict(self.weights)

    def update(self, active_net_returns: Mapping[str, float]) -> None:
        del active_net_returns


class HMATFPIT:
    """Six-source HMATF with monthly cost-aware fixed-share arbitration."""

    def __init__(
        self,
        n_assets: int,
        config: Optional[PITFactorConfig] = None,
        enabled_experts: Sequence[str] = PIT_EXPERTS,
        mechanism: str = "fixed_share",
    ) -> None:
        selected = config or PITFactorConfig()
        unknown = set(enabled_experts) - set(PIT_EXPERTS)
        if unknown:
            raise ValueError(f"unknown PIT experts: {sorted(unknown)}")
        self.config = selected
        self.n_assets = n_assets
        self.enabled_experts = tuple(enabled_experts)
        if mechanism == "fixed_share":
            self.arbitrator = MonthlyFixedShareArbitrator(selected, PIT_EXPERTS)
        elif mechanism == "equal":
            self.arbitrator = EqualWeightArbitrator(PIT_EXPERTS)
        else:
            raise ValueError(f"unsupported arbitration mechanism: {mechanism}")
        self.optimizer = ResidualPortfolioOptimizer(selected)
        self.current_weights = np.zeros(n_assets)
        self.expert_current = {name: np.zeros(n_assets) for name in PIT_EXPERTS}
        self.expert_committed_targets: Dict[str, Optional[np.ndarray]] = {
            name: None for name in PIT_EXPERTS
        }

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
        composite = np.mean(stacked, axis=0) * (0.5 + 0.5 * agreement)
        return _softmax(composite, self.config.score_temperature)

    def _factor_target(self, scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if scores is None:
            return None
        values = np.asarray(scores, dtype=float)
        if values.shape != (self.n_assets,) or not np.all(np.isfinite(values)):
            return None
        return _softmax(values, self.config.score_temperature)

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
        inv_vol = 1.0 / vol
        factors = factor_scores or {}
        targets: Dict[str, Optional[np.ndarray]] = {
            "trend": self._multifrequency_trend(values),
            "meanrev": _softmax(reversal, self.config.score_temperature),
            "risk_parity": inv_vol / inv_vol.sum(),
            "fundamental": self._factor_target(factors.get("fundamental")),
            "event": self._factor_target(factors.get("event")),
            "cross_sectional": self._factor_target(factors.get("cross_sectional")),
        }
        return {
            name: target if name in self.enabled_experts else None
            for name, target in targets.items()
        }

    def rebalance(
        self,
        prices: np.ndarray,
        covariance: np.ndarray,
        benchmark: Optional[np.ndarray] = None,
        factor_scores: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, object]:
        benchmark_weights = (
            np.full(self.n_assets, 1.0 / self.n_assets)
            if benchmark is None
            else np.asarray(benchmark, dtype=float)
        )
        targets = self.expert_targets(prices, factor_scores)
        available = {name: target is not None for name, target in targets.items()}
        expert_weights = self.arbitrator.allocate(available)
        names = [name for name in PIT_EXPERTS if available[name]]
        combined = (
            sum(
                expert_weights[name] * np.asarray(targets[name], dtype=float)
                for name in names
            )
            if names
            else benchmark_weights
        )
        target = self.optimizer.optimize(
            benchmark_weights, combined, covariance, self.current_weights
        )
        return {
            "target_weights": target,
            "benchmark_weights": benchmark_weights,
            "expert_targets": targets,
            "expert_weights": expert_weights,
            "available": available,
        }

    def record_returns(
        self,
        asset_returns: np.ndarray,
        benchmark_weights: np.ndarray,
        expert_targets: Mapping[str, Optional[np.ndarray]],
        cost_rate: float,
    ) -> Dict[str, float]:
        asset_returns = np.asarray(asset_returns, dtype=float)
        benchmark_return = float(np.asarray(benchmark_weights) @ asset_returns)
        active_returns: Dict[str, float] = {}
        for name, optional_target in expert_targets.items():
            if optional_target is None:
                continue
            target = np.asarray(optional_target, dtype=float)
            committed = self.expert_committed_targets[name]
            changed = committed is None or not np.allclose(
                target, committed, atol=1e-12
            )
            if changed:
                turnover = float(np.sum(np.abs(target - self.expert_current[name])))
                self.expert_current[name] = target.copy()
                self.expert_committed_targets[name] = target.copy()
            else:
                turnover = 0.0
            gross_return = float(self.expert_current[name] @ asset_returns)
            net_return = gross_return - turnover * cost_rate
            active_returns[name] = net_return - benchmark_return
            denominator = 1.0 + gross_return
            if denominator > 0:
                self.expert_current[name] = (
                    self.expert_current[name]
                    * (1.0 + asset_returns)
                    / denominator
                )
        self.arbitrator.update(active_returns)
        return active_returns

    def commit(self, target_weights: np.ndarray) -> None:
        self.current_weights = np.asarray(target_weights, dtype=float).copy()


class HMATFPITVanillaHedge(HMATFPIT):
    """PIT ablation without fixed-share probability transfer."""

    def __init__(self, n_assets: int, enabled_experts: Sequence[str] = PIT_EXPERTS):
        super().__init__(
            n_assets,
            PITFactorConfig(fixed_share_alpha=0.0),
            enabled_experts,
            mechanism="fixed_share",
        )


class HMATFPITEqualFusion(HMATFPIT):
    """Static equal fusion under identical experts and constraints."""

    def __init__(self, n_assets: int, enabled_experts: Sequence[str] = PIT_EXPERTS):
        super().__init__(n_assets, enabled_experts=enabled_experts, mechanism="equal")
