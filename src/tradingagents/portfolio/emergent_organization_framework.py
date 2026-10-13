"""Budget-constrained organization learning over anonymous trading capabilities.

The controller decides which capability agents may report to the portfolio
arbitrator. It observes outcomes only for agents that were actually active,
so inactive agents re-enter through uncertainty-driven exploration rather
than free counterfactual feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import prod, sqrt
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from tradingagents.portfolio.media_factor_framework import (
    MEDIA_EXPERTS,
    MediaFactorConfig,
    HMATFMedia,
)


@dataclass(frozen=True)
class ResourceProfile:
    tokens: int
    latency_ms: float
    failure_probability: float


DEFAULT_RESOURCE_PROFILES: Dict[str, ResourceProfile] = {
    "trend": ResourceProfile(650, 180.0, 0.020),
    "meanrev": ResourceProfile(450, 120.0, 0.015),
    "risk_parity": ResourceProfile(350, 100.0, 0.010),
    "cross_sectional": ResourceProfile(700, 220.0, 0.025),
    "fundamental": ResourceProfile(1200, 480.0, 0.040),
    "media_text": ResourceProfile(2200, 900.0, 0.080),
}


@dataclass(frozen=True)
class OrganizationConfig:
    token_budget: int = 4200
    latency_budget_ms: float = 1050.0
    joint_failure_budget: float = 0.18
    communication_tokens: int = 48
    communication_latency_ms: float = 18.0
    lookback_rounds: int = 12
    minimum_agents: int = 1
    maximum_agents: int = 6
    cold_start_reward: float = 0.004
    accuracy_weight: float = 0.004
    exploration_weight: float = 0.004
    staleness_weight: float = 0.0004
    risk_aversion: float = 0.20
    diversity_weight: float = 0.0015
    resource_penalty: float = 0.0025
    context_weight: float = 0.0
    ensemble_risk_weight: float = 0.0
    switching_penalty: float = 0.0
    anchor_agent: Optional[str] = None
    anchor_weight: float = 0.0
    failure_loss: float = 0.020
    softmax_temperature: float = 0.020
    exit_window: int = 3
    exit_reward_threshold: float = -0.010
    cooldown_rounds: int = 3
    enable_exit: bool = True
    enforce_budgets: bool = True
    inject_failures: bool = True
    fixed_agents: Optional[tuple[str, ...]] = None
    seed: int = 0


class BudgetedOrganizationController:
    """Select a dynamic capability subset under hard service constraints."""

    def __init__(
        self,
        config: Optional[OrganizationConfig] = None,
        expert_names: Sequence[str] = MEDIA_EXPERTS,
        profiles: Optional[Mapping[str, ResourceProfile]] = None,
    ) -> None:
        self.config = config or OrganizationConfig()
        self.expert_names = tuple(expert_names)
        supplied = dict(profiles or DEFAULT_RESOURCE_PROFILES)
        missing = set(self.expert_names) - set(supplied)
        if missing:
            raise ValueError(f"missing resource profiles: {sorted(missing)}")
        self.profiles = {name: supplied[name] for name in self.expert_names}
        self.weights = {name: 0.0 for name in self.expert_names}
        self.observations: Dict[str, list[dict[str, object]]] = {
            name: [] for name in self.expert_names
        }
        self.cooldown_until = {name: 0 for name in self.expert_names}
        self.last_observed_round = {name: -1 for name in self.expert_names}
        self.pending_returns = {name: 0.0 for name in self.expert_names}
        self.pending_hits = {name: 0 for name in self.expert_names}
        self.pending_days = {name: 0 for name in self.expert_names}
        self.last_active: tuple[str, ...] = ()
        self.last_regime: Optional[str] = None
        self.has_open_round = False
        self.round_index = 0
        self.decision_history: list[dict[str, object]] = []
        self.exit_events: list[dict[str, object]] = []

    def _resource_use(self, names: Sequence[str]) -> dict[str, float]:
        names = tuple(names)
        edge_count = len(names)
        tokens = sum(self.profiles[name].tokens for name in names)
        tokens += edge_count * self.config.communication_tokens
        maximum_latency = max(
            (self.profiles[name].latency_ms for name in names), default=0.0
        )
        latency = maximum_latency + edge_count * self.config.communication_latency_ms
        joint_failure = 1.0 - prod(
            1.0 - self.profiles[name].failure_probability for name in names
        )
        return {
            "tokens": float(tokens),
            "latency_ms": float(latency),
            "joint_failure_probability": float(joint_failure),
            "communication_edges": float(edge_count),
        }

    def _within_budgets(self, names: Sequence[str]) -> bool:
        usage = self._resource_use(names)
        return bool(
            usage["tokens"] <= self.config.token_budget
            and usage["latency_ms"] <= self.config.latency_budget_ms
            and usage["joint_failure_probability"] <= self.config.joint_failure_budget
        )

    def _feasible(self, names: Sequence[str]) -> bool:
        return bool(not self.config.enforce_budgets or self._within_budgets(names))

    def _recent(self, name: str) -> list[dict[str, object]]:
        return self.observations[name][-self.config.lookback_rounds :]

    def _agent_score(self, name: str, regime: Optional[str] = None) -> float:
        recent = self._recent(name)
        if recent:
            rewards = np.asarray([row["reward"] for row in recent], dtype=float)
            accuracies = np.asarray([row["accuracy"] for row in recent], dtype=float)
            mean = float(np.mean(rewards))
            accuracy = float(np.mean(accuracies))
            risk = float(np.std(rewards))
            contextual = [row for row in recent if row.get("regime") == regime]
            if contextual and self.config.context_weight > 0.0:
                weight = float(np.clip(self.config.context_weight, 0.0, 1.0))
                contextual_rewards = np.asarray(
                    [row["reward"] for row in contextual], dtype=float
                )
                contextual_accuracy = float(
                    np.mean([row["accuracy"] for row in contextual])
                )
                mean = (1.0 - weight) * mean + weight * float(
                    np.mean(contextual_rewards)
                )
                accuracy = (1.0 - weight) * accuracy + weight * contextual_accuracy
                risk = (1.0 - weight) * risk + weight * float(
                    np.std(contextual_rewards)
                )
        else:
            mean = self.config.cold_start_reward
            accuracy = 0.5
            risk = 0.0
        count = len(self.observations[name])
        exploration = self.config.exploration_weight * sqrt(
            np.log(self.round_index + 2.0) / (count + 1.0)
        )
        stale = max(0, self.round_index - self.last_observed_round[name] - 1)
        staleness = self.config.staleness_weight * min(stale, 6)
        reliability = self.profiles[name].failure_probability * self.config.failure_loss
        return float(
            mean
            - self.config.risk_aversion * risk
            + self.config.accuracy_weight * (accuracy - 0.5)
            + exploration
            + staleness
            - reliability
        )

    def _paired_rewards(
        self, left: str, right: str, regime: Optional[str] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        left_by_round = {
            int(row["round"]): row for row in self._recent(left)
        }
        right_by_round = {
            int(row["round"]): row for row in self._recent(right)
        }
        common = sorted(set(left_by_round) & set(right_by_round))
        if regime is not None and self.config.context_weight > 0.0:
            contextual = [
                index
                for index in common
                if left_by_round[index].get("regime") == regime
                and right_by_round[index].get("regime") == regime
            ]
            if len(contextual) >= 3:
                common = contextual
        x = np.asarray(
            [left_by_round[index]["reward"] for index in common], dtype=float
        )
        y = np.asarray(
            [right_by_round[index]["reward"] for index in common], dtype=float
        )
        return x, y

    def _correlation(
        self, left: str, right: str, regime: Optional[str] = None
    ) -> float:
        x, y = self._paired_rewards(left, right, regime)
        if len(x) < 3:
            return 0.0
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return 0.0
        return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))

    def _ensemble_risk(
        self, names: Sequence[str], regime: Optional[str] = None
    ) -> float:
        names = tuple(names)
        if not names:
            return 0.0
        covariance = np.zeros((len(names), len(names)), dtype=float)
        for i, left in enumerate(names):
            left_rewards = np.asarray(
                [row["reward"] for row in self._recent(left)], dtype=float
            )
            covariance[i, i] = float(np.var(left_rewards)) if len(left_rewards) else 0.0
            for j in range(i + 1, len(names)):
                x, y = self._paired_rewards(left, names[j], regime)
                value = float(np.cov(x, y, ddof=0)[0, 1]) if len(x) >= 3 else 0.0
                covariance[i, j] = covariance[j, i] = value
        weights = np.full(len(names), 1.0 / len(names))
        variance = float(weights @ covariance @ weights)
        return float(np.sqrt(max(0.0, variance)))

    def _switching_rate(self, names: Sequence[str]) -> float:
        current = set(names)
        previous = set(self.last_active)
        union = current | previous
        return float(len(current.symmetric_difference(previous)) / max(1, len(union)))

    def _subset_score(
        self, names: Sequence[str], regime: Optional[str] = None
    ) -> float:
        names = tuple(names)
        scores = np.asarray(
            [self._agent_score(name, regime) for name in names], dtype=float
        )
        expected = float(np.mean(scores))
        if len(names) > 1:
            correlations = [
                self._correlation(left, right, regime)
                for left, right in combinations(names, 2)
            ]
            diversity = float(np.mean([1.0 - max(value, 0.0) for value in correlations]))
        else:
            diversity = 0.0
        usage = self._resource_use(names)
        resource_load = (
            usage["tokens"] / self.config.token_budget
            + usage["latency_ms"] / self.config.latency_budget_ms
            + usage["joint_failure_probability"] / self.config.joint_failure_budget
        ) / 3.0
        return float(
            expected
            + self.config.diversity_weight * diversity
            - self.config.resource_penalty * resource_load
            - self.config.ensemble_risk_weight * self._ensemble_risk(names, regime)
            - self.config.switching_penalty * self._switching_rate(names)
        )

    def _commit_pending(self) -> None:
        if not self.has_open_round:
            return
        for name in self.last_active:
            days = self.pending_days[name]
            if days == 0:
                continue
            reward = float(np.clip(self.pending_returns[name], -0.10, 0.10))
            accuracy = float(self.pending_hits[name] / days)
            self.observations[name].append(
                {
                    "round": float(self.round_index),
                    "reward": reward,
                    "accuracy": accuracy,
                    "regime": self.last_regime,
                }
            )
            self.last_observed_round[name] = self.round_index
            recent = self._recent(name)
            if self.config.enable_exit and len(recent) >= self.config.exit_window:
                window = recent[-self.config.exit_window :]
                average = float(np.mean([row["reward"] for row in window]))
                if average < self.config.exit_reward_threshold:
                    until = self.round_index + self.config.cooldown_rounds + 1
                    if self.cooldown_until[name] < until:
                        self.cooldown_until[name] = until
                        self.exit_events.append(
                            {
                                "round": self.round_index,
                                "agent": name,
                                "average_reward": average,
                                "cooldown_until": until,
                            }
                        )
        self.pending_returns = {name: 0.0 for name in self.expert_names}
        self.pending_hits = {name: 0 for name in self.expert_names}
        self.pending_days = {name: 0 for name in self.expert_names}
        self.round_index += 1
        self.has_open_round = False

    def _failed(self, name: str) -> bool:
        if not self.config.inject_failures:
            return False
        agent_index = self.expert_names.index(name)
        seed = np.random.SeedSequence(
            [self.config.seed, self.round_index, agent_index]
        )
        draw = np.random.default_rng(seed).random()
        return bool(draw < self.profiles[name].failure_probability)

    def _candidate_set(self, available: Mapping[str, bool]) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.expert_names
            if available.get(name, False) and self.cooldown_until[name] <= self.round_index
        )

    def _select(
        self, available: Mapping[str, bool], regime: Optional[str] = None
    ) -> tuple[str, ...]:
        candidates = self._candidate_set(available)
        if self.config.fixed_agents is not None:
            fixed = tuple(name for name in self.config.fixed_agents if name in candidates)
            return fixed if fixed else candidates[:1]
        feasible: list[tuple[float, tuple[str, ...]]] = []
        upper = min(self.config.maximum_agents, len(candidates))
        for size in range(self.config.minimum_agents, upper + 1):
            for subset in combinations(candidates, size):
                if (
                    self.config.anchor_agent in candidates
                    and self.config.anchor_agent not in subset
                ):
                    continue
                if self._feasible(subset):
                    feasible.append((self._subset_score(subset, regime), subset))
        if feasible:
            return max(feasible, key=lambda item: (item[0], -len(item[1]), item[1]))[1]
        cheapest = sorted(
            candidates,
            key=lambda name: (
                self.profiles[name].tokens,
                self.profiles[name].latency_ms,
                name,
            ),
        )
        return tuple(cheapest[:1])

    def allocate(
        self, available: Mapping[str, bool], regime: Optional[str] = None
    ) -> Dict[str, float]:
        self._commit_pending()
        intended = self._select(available, regime)
        usage = self._resource_use(intended)
        failures = tuple(
            name
            for name in intended
            if self._failed(name)
        )
        active = tuple(name for name in intended if name not in failures)
        raw = np.asarray(
            [self._agent_score(name, regime) for name in active], dtype=float
        )
        if len(active):
            temperature = max(self.config.softmax_temperature, 1e-8)
            logits = (raw - np.max(raw)) / temperature
            selected_weights = np.exp(logits)
            selected_weights /= selected_weights.sum()
            anchor = self.config.anchor_agent
            if anchor in active and self.config.anchor_weight > 0.0:
                anchor_weight = float(np.clip(self.config.anchor_weight, 0.0, 1.0))
                selected_weights *= 1.0 - anchor_weight
                selected_weights[active.index(anchor)] += anchor_weight
        else:
            selected_weights = np.asarray([], dtype=float)
        self.weights = {name: 0.0 for name in self.expert_names}
        self.weights.update(
            {name: float(selected_weights[index]) for index, name in enumerate(active)}
        )
        previous = self.last_active
        self.last_active = active
        self.last_regime = regime
        self.has_open_round = True
        self.decision_history.append(
            {
                "round": self.round_index,
                "regime": regime,
                "intended_agents": list(intended),
                "active_agents": list(active),
                "failed_agents": list(failures),
                "agent_count": len(active),
                "communication_edges": len(active),
                "attempted_communication_edges": len(intended),
                "tokens": usage["tokens"],
                "latency_ms": usage["latency_ms"],
                "joint_failure_probability": usage["joint_failure_probability"],
                "budget_feasible": self._within_budgets(intended),
                "topology_churn": len(set(previous).symmetric_difference(active)),
                "weights": dict(self.weights),
            }
        )
        return dict(self.weights)

    def update(self, active_net_returns: Mapping[str, float]) -> None:
        for name in self.last_active:
            value = float(active_net_returns.get(name, np.nan))
            if not np.isfinite(value):
                continue
            self.pending_returns[name] += value
            self.pending_hits[name] += int(value > 0.0)
            self.pending_days[name] += 1

    def finalize_pending_round(self) -> None:
        self._commit_pending()


class HMATFEmergentOrganization(HMATFMedia):
    """HMATF media experts with a learned resource-aware organization."""

    def __init__(
        self,
        n_assets: int,
        organization_config: Optional[OrganizationConfig] = None,
        enabled_experts: Sequence[str] = MEDIA_EXPERTS,
        portfolio_config: Optional[MediaFactorConfig] = None,
        resource_profiles: Optional[Mapping[str, ResourceProfile]] = None,
    ) -> None:
        super().__init__(
            n_assets,
            config=portfolio_config,
            enabled_experts=enabled_experts,
        )
        self.arbitrator = BudgetedOrganizationController(
            organization_config,
            expert_names=enabled_experts,
            profiles=resource_profiles,
        )
