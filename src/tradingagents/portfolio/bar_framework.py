"""Benchmark-aware residual allocation for heterogeneous trading experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np


EXPERTS = ("trend", "meanrev", "risk_parity", "fundamental", "event")


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=20.0, neginf=-20.0)
    logits = temperature * values
    logits -= np.max(logits)
    weights = np.exp(np.clip(logits, -50.0, 50.0))
    return weights / weights.sum()


@dataclass(frozen=True)
class BARConfig:
    active_budget: float = 0.25
    tracking_error_limit: float = 0.06
    max_asset_weight: float = 0.30
    no_trade_band: float = 0.01
    arbitrator_eta: float = 30.0
    arbitrator_decay: float = 0.99
    arbitrator_risk_aversion: float = 1.0
    score_temperature: float = 1.5


class ActiveReturnArbitrator:
    """Allocate active risk from lagged counterfactual net active returns."""

    def __init__(self, config: BARConfig):
        self.config = config
        self.utility = {name: 0.0 for name in EXPERTS}
        self.weights = {name: 0.0 for name in EXPERTS}

    def allocate(self, available: Mapping[str, bool]) -> Dict[str, float]:
        names = [name for name in EXPERTS if available.get(name, False)]
        if not names:
            self.weights = {name: 0.0 for name in EXPERTS}
            return dict(self.weights)
        logits = np.asarray([self.utility[name] for name in names])
        allocated = _softmax(logits, self.config.arbitrator_eta)
        self.weights = {name: 0.0 for name in EXPERTS}
        self.weights.update({name: float(allocated[i]) for i, name in enumerate(names)})
        return dict(self.weights)

    def update(self, active_net_returns: Mapping[str, float]) -> None:
        for name in EXPERTS:
            if name not in active_net_returns:
                continue
            value = float(active_net_returns[name])
            if not np.isfinite(value):
                value = 0.0
            utility = value - self.config.arbitrator_risk_aversion * value * value
            self.utility[name] = self.config.arbitrator_decay * self.utility[name] + utility


class ResidualPortfolioOptimizer:
    """Project an expert target onto active-risk and long-only constraints."""

    def __init__(self, config: BARConfig):
        self.config = config

    def optimize(
        self,
        benchmark: np.ndarray,
        expert_target: np.ndarray,
        covariance: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        benchmark = np.asarray(benchmark, dtype=float)
        delta = np.asarray(expert_target, dtype=float) - benchmark
        delta -= np.mean(delta)

        l1 = float(np.sum(np.abs(delta)))
        if l1 > self.config.active_budget and l1 > 0:
            delta *= self.config.active_budget / l1

        active_vol = float(np.sqrt(max(0.0, delta @ covariance @ delta)))
        if active_vol > self.config.tracking_error_limit and active_vol > 0:
            delta *= self.config.tracking_error_limit / active_vol

        target = np.clip(benchmark + delta, 0.0, self.config.max_asset_weight)
        target = self._normalize_capped(target)
        if np.sum(np.abs(target - current)) < self.config.no_trade_band:
            return np.asarray(current, dtype=float).copy()
        return target

    def _normalize_capped(self, weights: np.ndarray) -> np.ndarray:
        weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
        for _ in range(10):
            total = weights.sum()
            if total <= 0:
                return np.full(len(weights), 1.0 / len(weights))
            weights /= total
            over = weights > self.config.max_asset_weight
            if not np.any(over):
                break
            excess = float(np.sum(weights[over] - self.config.max_asset_weight))
            weights[over] = self.config.max_asset_weight
            under = ~over
            if np.any(under):
                capacity = self.config.max_asset_weight - weights[under]
                capacity_sum = capacity.sum()
                if capacity_sum > 0:
                    weights[under] += excess * capacity / capacity_sum
        return weights / weights.sum()


class HMATFBAR:
    """Heterogeneous multi-agent benchmark-aware residual allocator."""

    def __init__(self, n_assets: int, config: Optional[BARConfig] = None):
        self.config = config or BARConfig()
        self.n_assets = n_assets
        self.arbitrator = ActiveReturnArbitrator(self.config)
        self.optimizer = ResidualPortfolioOptimizer(self.config)
        self.current_weights = np.zeros(n_assets)
        self.expert_current = {name: np.zeros(n_assets) for name in EXPERTS}

    def expert_targets(
        self,
        prices: np.ndarray,
        factor_scores: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 253:
            raise ValueError("at least 253 historical observations are required")
        log_prices = np.log(prices)
        daily = np.diff(log_prices, axis=0)
        vol = np.std(daily[-60:], axis=0) * np.sqrt(252) + 1e-8
        momentum = (log_prices[-1] - log_prices[-253]) / vol
        reversal = -(log_prices[-1] - log_prices[-21]) / vol
        inv_vol = 1.0 / vol
        factors = factor_scores or {}
        return {
            "trend": _softmax(momentum, self.config.score_temperature),
            "meanrev": _softmax(reversal, self.config.score_temperature),
            "risk_parity": inv_vol / inv_vol.sum(),
            "fundamental": self._factor_target(factors.get("fundamental")),
            "event": self._factor_target(factors.get("event")),
        }

    def rebalance(
        self,
        prices: np.ndarray,
        covariance: np.ndarray,
        benchmark: Optional[np.ndarray] = None,
        factor_scores: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, object]:
        benchmark = (
            np.full(self.n_assets, 1.0 / self.n_assets)
            if benchmark is None else np.asarray(benchmark, dtype=float)
        )
        targets = self.expert_targets(prices, factor_scores)
        available = {name: target is not None for name, target in targets.items()}
        expert_weights = self.arbitrator.allocate(available)
        combined = sum(
            expert_weights[name] * targets[name]
            for name in EXPERTS if targets[name] is not None
        )
        target = self.optimizer.optimize(benchmark, combined, covariance, self.current_weights)
        return {
            "target_weights": target,
            "benchmark_weights": benchmark,
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
        benchmark_return = float(np.asarray(benchmark_weights) @ asset_returns)
        active_returns = {}
        for name, target in expert_targets.items():
            if target is None:
                continue
            turnover = float(np.sum(np.abs(target - self.expert_current[name])))
            gross_return = float(target @ asset_returns)
            net_return = gross_return - turnover * cost_rate
            active_returns[name] = net_return - benchmark_return
            denominator = 1.0 + gross_return
            self.expert_current[name] = (
                target * (1.0 + asset_returns) / denominator
                if denominator > 0 else target.copy()
            )
        self.arbitrator.update(active_returns)
        return active_returns

    def commit(self, target_weights: np.ndarray) -> None:
        self.current_weights = np.asarray(target_weights, dtype=float).copy()

    def _factor_target(self, scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if scores is None:
            return None
        values = np.asarray(scores, dtype=float)
        if values.shape != (self.n_assets,) or not np.all(np.isfinite(values)):
            return None
        return _softmax(values, self.config.score_temperature)
