"""Context-aware residual allocation for heterogeneous trading experts.

This module extends HMATF-BAR without modifying the frozen preregistered v1
implementation. All contextual features are computed from observations available
at the rebalance timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from tradingagents.portfolio.bar_framework import (
    EXPERTS,
    BARConfig,
    HMATFBAR,
    ResidualPortfolioOptimizer,
    _softmax,
)


REGIMES = ("up_calm", "up_volatile", "down_calm", "down_volatile")


@dataclass(frozen=True)
class ContextualBARConfig(BARConfig):
    context_decay: float = 0.99
    context_shrinkage: float = 40.0
    uncertainty_penalty: float = 0.25
    contextual_eta: float = 1400.0
    expert_weight_floor: float = 0.05
    use_context: bool = True
    use_uncertainty: bool = True
    use_multifrequency_trend: bool = True


class ContextualRiskArbitrator:
    """Partially pool global and regime-specific expert evidence.

    Allocation at time t uses statistics updated only through t-1. The current
    regime is stored only so subsequently realized returns can update the
    corresponding regime statistics.
    """

    def __init__(self, config: ContextualBARConfig):
        self.config = config
        self.weights = {name: 0.0 for name in EXPERTS}
        self.current_regime = REGIMES[0]
        self.global_mass = {name: 0.0 for name in EXPERTS}
        self.global_sum = {name: 0.0 for name in EXPERTS}
        self.global_sq_sum = {name: 0.0 for name in EXPERTS}
        self.regime_mass = {
            regime: {name: 0.0 for name in EXPERTS} for regime in REGIMES
        }
        self.regime_sum = {
            regime: {name: 0.0 for name in EXPERTS} for regime in REGIMES
        }
        self.regime_sq_sum = {
            regime: {name: 0.0 for name in EXPERTS} for regime in REGIMES
        }

    @staticmethod
    def _moments(mass: float, total: float, sq_total: float) -> Tuple[float, float]:
        if mass <= 1e-12:
            return 0.0, 0.0
        mean = total / mass
        variance = max(0.0, sq_total / mass - mean * mean)
        return mean, variance

    def allocate(
        self,
        available: Mapping[str, bool],
        regime: str,
    ) -> Dict[str, float]:
        if regime not in REGIMES:
            raise ValueError(f"unsupported market regime: {regime}")
        self.current_regime = regime
        names = [name for name in EXPERTS if available.get(name, False)]
        if not names:
            self.weights = {name: 0.0 for name in EXPERTS}
            return dict(self.weights)

        scores = []
        for name in names:
            global_mean, global_var = self._moments(
                self.global_mass[name],
                self.global_sum[name],
                self.global_sq_sum[name],
            )
            local_mass = self.regime_mass[regime][name]
            local_mean, local_var = self._moments(
                local_mass,
                self.regime_sum[regime][name],
                self.regime_sq_sum[regime][name],
            )
            shrink = (
                local_mass / (local_mass + self.config.context_shrinkage)
                if self.config.use_context else 0.0
            )
            mean = (1.0 - shrink) * global_mean + shrink * local_mean
            variance = (1.0 - shrink) * global_var + shrink * local_var
            effective_mass = max(
                1.0,
                (1.0 - shrink) * self.global_mass[name] + shrink * local_mass,
            )
            uncertainty = (
                np.sqrt(variance / effective_mass)
                if self.config.use_uncertainty else 0.0
            )
            scores.append(mean - self.config.uncertainty_penalty * uncertainty)

        raw = _softmax(np.asarray(scores), self.config.contextual_eta)
        floor = min(self.config.expert_weight_floor, 0.99 / len(names))
        allocated = floor + (1.0 - floor * len(names)) * raw
        self.weights = {name: 0.0 for name in EXPERTS}
        self.weights.update({name: float(allocated[i]) for i, name in enumerate(names)})
        return dict(self.weights)

    def update(self, active_net_returns: Mapping[str, float]) -> None:
        decay = self.config.context_decay
        regime = self.current_regime
        for name, raw_value in active_net_returns.items():
            if name not in self.global_mass:
                continue
            value = float(raw_value) if np.isfinite(raw_value) else 0.0
            reward = value - self.config.arbitrator_risk_aversion * value * value
            for mass, total, sq_total in (
                (self.global_mass, self.global_sum, self.global_sq_sum),
                (
                    self.regime_mass[regime],
                    self.regime_sum[regime],
                    self.regime_sq_sum[regime],
                ),
            ):
                mass[name] = decay * mass[name] + 1.0
                total[name] = decay * total[name] + reward
                sq_total[name] = decay * sq_total[name] + reward * reward


class HMATFCAR(HMATFBAR):
    """HMATF with multi-frequency evidence and context-aware arbitration."""

    def __init__(self, n_assets: int, config: Optional[ContextualBARConfig] = None):
        contextual_config = config or ContextualBARConfig()
        super().__init__(n_assets, contextual_config)
        self.config = contextual_config
        self.arbitrator = ContextualRiskArbitrator(contextual_config)
        self.optimizer = ResidualPortfolioOptimizer(contextual_config)
        self.last_regime = REGIMES[0]

    @staticmethod
    def market_regime(prices: np.ndarray) -> str:
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 253:
            raise ValueError("at least 253 historical observations are required")
        daily = np.diff(np.log(prices), axis=0)
        benchmark_daily = np.mean(daily, axis=1)
        short_return = float(np.sum(benchmark_daily[-63:]))
        short_vol = float(np.std(benchmark_daily[-63:]) * np.sqrt(252))
        long_vol = float(np.std(benchmark_daily[-252:]) * np.sqrt(252))
        direction = "up" if short_return >= 0.0 else "down"
        volatility = "volatile" if short_vol > long_vol else "calm"
        return f"{direction}_{volatility}"

    def _multifrequency_trend(self, prices: np.ndarray) -> np.ndarray:
        log_prices = np.log(np.asarray(prices, dtype=float))
        daily = np.diff(log_prices, axis=0)
        annual_vol = np.std(daily[-60:], axis=0) * np.sqrt(252) + 1e-8
        signals = []
        for horizon in (21, 63, 126, 252):
            raw = (log_prices[-1] - log_prices[-1 - horizon]) / (
                annual_vol * np.sqrt(horizon / 252)
            )
            scale = np.std(raw)
            signals.append((raw - np.mean(raw)) / (scale + 1e-8))
        stacked = np.vstack(signals)
        agreement = np.abs(np.mean(np.sign(stacked), axis=0))
        composite = np.mean(stacked, axis=0) * (0.5 + 0.5 * agreement)
        return _softmax(composite, self.config.score_temperature)

    def expert_targets(
        self,
        prices: np.ndarray,
        factor_scores: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        targets = super().expert_targets(prices, factor_scores)
        if self.config.use_multifrequency_trend:
            targets["trend"] = self._multifrequency_trend(prices)
        return targets

    def rebalance(
        self,
        prices: np.ndarray,
        covariance: np.ndarray,
        benchmark: Optional[np.ndarray] = None,
        factor_scores: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, object]:
        benchmark_weights = (
            np.full(self.n_assets, 1.0 / self.n_assets)
            if benchmark is None else np.asarray(benchmark, dtype=float)
        )
        targets = self.expert_targets(prices, factor_scores)
        available = {name: target is not None for name, target in targets.items()}
        regime = self.market_regime(prices)
        expert_weights = self.arbitrator.allocate(available, regime)
        combined = sum(
            expert_weights[name] * targets[name]
            for name in EXPERTS if targets[name] is not None
        )
        target = self.optimizer.optimize(
            benchmark_weights,
            combined,
            covariance,
            self.current_weights,
        )
        self.last_regime = regime
        return {
            "target_weights": target,
            "benchmark_weights": benchmark_weights,
            "expert_targets": targets,
            "expert_weights": expert_weights,
            "available": available,
            "regime": regime,
        }


class HMATFRAM(HMATFCAR):
    """Reliability-aware multi-frequency arbitration selected on development data."""

    def __init__(self, n_assets: int, config: Optional[ContextualBARConfig] = None):
        selected = config or ContextualBARConfig(use_context=False)
        if selected.use_context:
            raise ValueError("HMATF-RAM requires use_context=False")
        super().__init__(n_assets, selected)

class HMATFMFA(HMATFCAR):
    """Multi-frequency evidence with reliability-normalized arbitration."""

    def __init__(self, n_assets: int, config: Optional[ContextualBARConfig] = None):
        selected = config or ContextualBARConfig(
            use_context=False,
            use_uncertainty=False,
        )
        if selected.use_context or selected.use_uncertainty:
            raise ValueError(
                "HMATF-MFA requires context and uncertainty penalties to be disabled"
            )
        super().__init__(n_assets, selected)