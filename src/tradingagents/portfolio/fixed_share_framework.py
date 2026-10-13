"""Cost-aware fixed-share arbitration for heterogeneous trading experts.

The module is a v3 research branch.  It does not modify the preregistered v2
implementation.  Expert rewards are accumulated over calendar months, and an
allocation for month t is updated only from rewards completed by month t-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np

from tradingagents.portfolio.bar_framework import (
    EXPERTS,
    ResidualPortfolioOptimizer,
)
from tradingagents.portfolio.contextual_bar_framework import (
    ContextualBARConfig,
    HMATFMFA,
)


@dataclass(frozen=True)
class FixedShareConfig(ContextualBARConfig):
    """Frozen parameters from the v3 external-validation protocol."""

    use_context: bool = False
    use_uncertainty: bool = False
    fixed_share_eta: float = 0.35
    fixed_share_alpha: float = 0.02
    monthly_reward_bound: float = 0.05


class MonthlyFixedShareArbitrator:
    """Track a switching expert sequence with fixed-share exponential weights.

    For monthly gains in [-B, B], K experts, T completed rounds, and a
    comparator sequence with S switches, the implemented update satisfies the
    standard exponential-weights certificate

      regret <= B * (C(S) / eta + eta * T / 2),

    where C(S) is the negative log prior probability of that sequence under the
    fixed-share Markov prior.  The certificate concerns clipped arbitration
    rewards, not realized portfolio returns after the optimizer projection.
    """

    def __init__(
        self,
        config: FixedShareConfig,
        expert_names: tuple[str, ...] = EXPERTS,
    ) -> None:
        if config.fixed_share_eta <= 0:
            raise ValueError("fixed_share_eta must be positive")
        if not 0 <= config.fixed_share_alpha < 1:
            raise ValueError("fixed_share_alpha must be in [0, 1)")
        if config.monthly_reward_bound <= 0:
            raise ValueError("monthly_reward_bound must be positive")
        self.config = config
        self.expert_names = tuple(expert_names)
        self.weights = {name: 0.0 for name in self.expert_names}
        self.pending = {name: 0.0 for name in self.expert_names}
        self.pending_observations = 0
        self.last_available: tuple[str, ...] = ()
        self.reward_history: List[Dict[str, float]] = []
        self.mixture_reward_history: List[float] = []
        self.availability_history: List[tuple[str, ...]] = []
        self.diagnostic_start = 0

    def _project_available(self, names: tuple[str, ...]) -> None:
        if not names:
            self.weights = {name: 0.0 for name in self.expert_names}
            return
        values = np.asarray([self.weights[name] for name in names], dtype=float)
        if values.sum() <= 1e-15:
            values = np.full(len(names), 1.0 / len(names))
        else:
            values /= values.sum()
        self.weights = {name: 0.0 for name in self.expert_names}
        self.weights.update({name: float(values[i]) for i, name in enumerate(names)})

    def _commit_pending_round(self) -> None:
        if self.pending_observations == 0 or not self.last_available:
            return
        names = self.last_available
        bound = self.config.monthly_reward_bound
        raw = {
            name: float(np.clip(self.pending[name], -bound, bound))
            for name in names
        }
        prior = np.asarray([self.weights[name] for name in names], dtype=float)
        prior /= prior.sum()
        self.reward_history.append(dict(raw))
        self.availability_history.append(tuple(names))
        self.mixture_reward_history.append(
            float(sum(self.weights[name] * raw[name] for name in names))
        )

        gains = np.asarray([raw[name] / bound for name in names], dtype=float)
        logits = np.log(np.maximum(prior, 1e-300)) + self.config.fixed_share_eta * gains
        logits -= np.max(logits)
        posterior = np.exp(logits)
        posterior /= posterior.sum()
        alpha = self.config.fixed_share_alpha
        if alpha > 0 and len(names) > 1:
            shared = (
                (1.0 - alpha) * posterior
                + alpha * (1.0 - posterior) / (len(names) - 1)
            )
        else:
            shared = posterior
        self.weights = {name: 0.0 for name in self.expert_names}
        self.weights.update({name: float(shared[i]) for i, name in enumerate(names)})
        self.pending = {name: 0.0 for name in self.expert_names}
        self.pending_observations = 0

    def allocate(
        self,
        available: Mapping[str, bool],
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        del regime
        self._commit_pending_round()
        names = tuple(name for name in self.expert_names if available.get(name, False))
        self._project_available(names)
        self.last_available = names
        return dict(self.weights)

    def update(self, active_net_returns: Mapping[str, float]) -> None:
        if not self.last_available:
            return
        for name in self.last_available:
            value = float(active_net_returns.get(name, 0.0))
            self.pending[name] += value if np.isfinite(value) else 0.0
        self.pending_observations += 1

    @property
    def completed_rounds(self) -> int:
        return len(self.reward_history) - self.diagnostic_start

    def mark_diagnostic_start(self) -> None:
        """Exclude preceding warm-up rounds from subsequent diagnostics."""
        self.diagnostic_start = len(self.reward_history)

    def finalize_pending_round(self) -> None:
        """Commit the final completed month for evaluation-only diagnostics."""
        self._commit_pending_round()

    def _diagnostic_rewards(self) -> List[Dict[str, float]]:
        return self.reward_history[self.diagnostic_start:]

    def _diagnostic_availability(self) -> List[tuple[str, ...]]:
        return self.availability_history[self.diagnostic_start:]

    def _diagnostic_mixture_rewards(self) -> List[float]:
        return self.mixture_reward_history[self.diagnostic_start:]

    def constant_availability(self) -> bool:
        availability = self._diagnostic_availability()
        return bool(availability) and all(
            names == availability[0]
            for names in availability
        )

    def theoretical_regret_bound(self, switches: int) -> float:
        """Return the cumulative reward-regret certificate in raw return units."""
        rounds = self.completed_rounds
        if rounds == 0:
            return 0.0
        if not self.constant_availability():
            raise ValueError("certificate requires a constant available expert set")
        experts = len(self._diagnostic_availability()[0])
        if experts < 2:
            return 0.0
        if switches < 0 or switches > rounds - 1:
            raise ValueError("switches must be between 0 and T-1")
        alpha = self.config.fixed_share_alpha
        if switches > 0 and alpha <= 0:
            return float("inf")
        complexity = np.log(experts)
        if switches:
            complexity += switches * np.log((experts - 1) / alpha)
        stays = rounds - 1 - switches
        if alpha > 0:
            complexity += stays * np.log(1.0 / (1.0 - alpha))
        eta = self.config.fixed_share_eta
        return float(
            self.config.monthly_reward_bound
            * (complexity / eta + eta * rounds / 2.0)
        )

    def best_switching_reward(self, switches: int) -> float:
        """Offline best clipped expert path with at most ``switches`` changes."""
        rounds = self.completed_rounds
        if rounds == 0:
            return 0.0
        if not self.constant_availability():
            raise ValueError("diagnostic requires a constant available expert set")
        names = self._diagnostic_availability()[0]
        rewards = np.asarray([
            [round_reward[name] for name in names]
            for round_reward in self._diagnostic_rewards()
        ])
        max_switches = min(max(0, switches), rounds - 1)
        dp = np.full((max_switches + 1, len(names)), -np.inf)
        dp[0] = rewards[0]
        for t in range(1, rounds):
            updated = np.full_like(dp, -np.inf)
            for used in range(max_switches + 1):
                updated[used] = np.maximum(updated[used], dp[used] + rewards[t])
                if used == 0:
                    continue
                for current in range(len(names)):
                    previous = np.delete(dp[used - 1], current)
                    updated[used, current] = max(
                        updated[used, current],
                        float(np.max(previous)) + rewards[t, current],
                    )
            dp = updated
        return float(np.max(dp))

    def realized_tracking_regret(self, switches: int) -> float:
        return float(
            self.best_switching_reward(switches)
            - np.sum(self._diagnostic_mixture_rewards())
        )


class FixedWeightArbitrator:
    """Equal-weight mechanism baseline with the same allocation interface."""

    def __init__(self) -> None:
        self.weights = {name: 0.0 for name in EXPERTS}

    def allocate(
        self,
        available: Mapping[str, bool],
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        del regime
        names = [name for name in EXPERTS if available.get(name, False)]
        self.weights = {name: 0.0 for name in EXPERTS}
        if names:
            self.weights.update({name: 1.0 / len(names) for name in names})
        return dict(self.weights)

    def update(self, active_net_returns: Mapping[str, float]) -> None:
        del active_net_returns


class HMATFFixedShare(HMATFMFA):
    """HMATF-MFA expert layer with theoretically grounded fixed-share arbitration."""

    def __init__(self, n_assets: int, config: Optional[FixedShareConfig] = None):
        selected = config or FixedShareConfig()
        super().__init__(n_assets, selected)
        self.config = selected
        self.arbitrator = MonthlyFixedShareArbitrator(selected)
        self.optimizer = ResidualPortfolioOptimizer(selected)
        self.expert_committed_targets: Dict[str, Optional[np.ndarray]] = {
            name: None for name in EXPERTS
        }

    def record_returns(
        self,
        asset_returns: np.ndarray,
        benchmark_weights: np.ndarray,
        expert_targets: Mapping[str, Optional[np.ndarray]],
        cost_rate: float,
    ) -> Dict[str, float]:
        """Track monthly expert portfolios without artificial daily rebalancing."""
        asset_returns = np.asarray(asset_returns, dtype=float)
        benchmark_return = float(np.asarray(benchmark_weights) @ asset_returns)
        active_returns: Dict[str, float] = {}
        for name, optional_target in expert_targets.items():
            if optional_target is None:
                continue
            target = np.asarray(optional_target, dtype=float)
            committed = self.expert_committed_targets[name]
            changed = committed is None or not np.allclose(target, committed, atol=1e-12)
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
                    self.expert_current[name] * (1.0 + asset_returns) / denominator
                )
        self.arbitrator.update(active_returns)
        return active_returns


class HMATFVanillaHedge(HMATFFixedShare):
    """Ablation without probability sharing between experts."""

    def __init__(self, n_assets: int, config: Optional[FixedShareConfig] = None):
        selected = config or FixedShareConfig(fixed_share_alpha=0.0)
        if selected.fixed_share_alpha != 0.0:
            raise ValueError("vanilla Hedge requires fixed_share_alpha=0")
        super().__init__(n_assets, selected)


class HMATFEqualFusion(HMATFMFA):
    """Static equal-fusion baseline under the same optimizer and experts."""

    def __init__(self, n_assets: int, config: Optional[FixedShareConfig] = None):
        selected = config or FixedShareConfig()
        super().__init__(n_assets, selected)
        self.arbitrator = FixedWeightArbitrator()


class HMATFPluralityVote(HMATFEqualFusion):
    """Vote for each expert's top-three assets, then project the vote target."""

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
        names = [name for name in EXPERTS if available[name]]
        votes = np.zeros(self.n_assets)
        top_k = min(3, self.n_assets)
        for name in names:
            target = np.asarray(targets[name], dtype=float)
            selected = np.argpartition(target, -top_k)[-top_k:]
            votes[selected] += 1.0
        vote_target = votes / votes.sum() if votes.sum() else benchmark_weights
        target_weights = self.optimizer.optimize(
            benchmark_weights, vote_target, covariance, self.current_weights
        )
        return {
            "target_weights": target_weights,
            "benchmark_weights": benchmark_weights,
            "expert_targets": targets,
            "expert_weights": {name: 1.0 / len(names) if name in names else 0.0 for name in EXPERTS},
            "available": available,
            "regime": self.market_regime(prices),
        }


class HMATFSingleExpert(HMATFMFA):
    """Single-agent mechanism baseline under the shared portfolio optimizer."""

    def __init__(
        self,
        n_assets: int,
        expert_name: str,
        config: Optional[FixedShareConfig] = None,
    ) -> None:
        if expert_name not in {"trend", "meanrev", "risk_parity"}:
            raise ValueError(f"unsupported single expert: {expert_name}")
        selected = config or FixedShareConfig()
        super().__init__(n_assets, selected)
        self.selected_expert = expert_name

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
        selected_target = np.asarray(targets[self.selected_expert], dtype=float)
        target_weights = self.optimizer.optimize(
            benchmark_weights, selected_target, covariance, self.current_weights
        )
        expert_weights = {name: 0.0 for name in EXPERTS}
        expert_weights[self.selected_expert] = 1.0
        return {
            "target_weights": target_weights,
            "benchmark_weights": benchmark_weights,
            "expert_targets": targets,
            "expert_weights": expert_weights,
            "available": {name: target is not None for name, target in targets.items()},
            "regime": self.market_regime(prices),
        }
