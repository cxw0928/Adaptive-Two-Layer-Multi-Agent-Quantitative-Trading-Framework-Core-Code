from __future__ import annotations

from dataclasses import replace

import numpy as np

from tradingagents.portfolio.emergent_organization_framework import (
    BudgetedOrganizationController,
    DEFAULT_RESOURCE_PROFILES,
    HMATFEmergentOrganization,
    OrganizationConfig,
    ResourceProfile,
)
from tradingagents.portfolio.media_factor_framework import MEDIA_EXPERTS
from tradingagents.portfolio.media_factor_framework import MediaFactorConfig


def test_selected_organization_respects_hard_budgets_and_normalizes_weights():
    controller = BudgetedOrganizationController(
        OrganizationConfig(seed=7, inject_failures=False)
    )
    weights = controller.allocate({name: True for name in MEDIA_EXPERTS})
    decision = controller.decision_history[-1]
    assert decision["budget_feasible"]
    assert decision["tokens"] <= controller.config.token_budget
    assert decision["latency_ms"] <= controller.config.latency_budget_ms
    assert np.isclose(sum(weights.values()), 1.0)
    assert decision["communication_edges"] == decision["agent_count"]


def test_fixed_all_exposes_resource_violation_for_matched_accounting():
    controller = BudgetedOrganizationController(
        OrganizationConfig(
            fixed_agents=MEDIA_EXPERTS,
            enforce_budgets=False,
            inject_failures=False,
        )
    )
    controller.allocate({name: True for name in MEDIA_EXPERTS})
    decision = controller.decision_history[-1]
    assert decision["agent_count"] == len(MEDIA_EXPERTS)
    assert decision["tokens"] > 4200
    assert not decision["budget_feasible"]


def test_failed_agent_does_not_report_or_receive_outcome_feedback():
    profiles = dict(DEFAULT_RESOURCE_PROFILES)
    profiles["trend"] = ResourceProfile(650, 180.0, 1.0)
    controller = BudgetedOrganizationController(
        OrganizationConfig(
            fixed_agents=("trend",),
            enforce_budgets=False,
            inject_failures=True,
            seed=3,
        ),
        profiles=profiles,
    )
    weights = controller.allocate({name: True for name in MEDIA_EXPERTS})
    assert controller.decision_history[-1]["failed_agents"] == ["trend"]
    assert sum(weights.values()) == 0.0
    controller.update({"trend": 0.05})
    controller.allocate({name: True for name in MEDIA_EXPERTS})
    assert controller.observations["trend"] == []


def test_persistently_harmful_active_agent_exits_and_can_reenter():
    config = OrganizationConfig(
        fixed_agents=("trend",),
        enforce_budgets=False,
        inject_failures=False,
        exit_window=3,
        cooldown_rounds=2,
        exit_reward_threshold=-0.01,
    )
    controller = BudgetedOrganizationController(config)
    available = {name: True for name in MEDIA_EXPERTS}
    for _ in range(3):
        controller.allocate(available)
        for _ in range(20):
            controller.update({"trend": -0.001})
    controller.allocate(available)
    assert controller.exit_events
    assert controller.decision_history[-1]["active_agents"] != ["trend"]
    for _ in range(2):
        substitute = controller.decision_history[-1]["active_agents"]
        controller.update({name: 0.001 for name in substitute})
        controller.allocate(available)
    assert controller.decision_history[-1]["active_agents"] == ["trend"]


def test_disabling_exit_is_a_clean_ablation():
    base = OrganizationConfig(inject_failures=False)
    no_exit = replace(base, enable_exit=False)
    assert base.enable_exit
    assert not no_exit.enable_exit


def test_switching_penalty_prefers_the_previous_topology_when_scores_tie():
    controller = BudgetedOrganizationController(
        OrganizationConfig(
            inject_failures=False,
            cold_start_reward=0.0,
            accuracy_weight=0.0,
            exploration_weight=0.0,
            staleness_weight=0.0,
            risk_aversion=0.0,
            diversity_weight=0.0,
            resource_penalty=0.0,
            switching_penalty=0.1,
            maximum_agents=2,
        ),
        expert_names=("trend", "meanrev"),
    )
    controller.last_active = ("trend",)
    selected = controller._select(
        {"trend": True, "meanrev": True}, regime="up_low_vol"
    )
    assert selected == ("trend",)


def test_context_is_stored_with_delayed_feedback():
    controller = BudgetedOrganizationController(
        OrganizationConfig(
            fixed_agents=("trend",), inject_failures=False, enable_exit=False
        ),
        expert_names=("trend",),
    )
    controller.allocate({"trend": True}, regime="down_high_vol")
    controller.update({"trend": 0.01})
    controller.allocate({"trend": True}, regime="up_low_vol")
    assert controller.observations["trend"][0]["regime"] == "down_high_vol"


def test_emergent_model_can_use_a_price_only_capability_pool():
    model = HMATFEmergentOrganization(
        3,
        OrganizationConfig(inject_failures=False),
        enabled_experts=("trend", "risk_parity"),
    )
    prices = np.exp(np.linspace(0.0, 0.2, 253))[:, None] * np.array([[1.0, 1.2, 0.8]])
    decision = model.rebalance(
        prices,
        np.eye(3) * 0.02,
        np.full(3, 1.0 / 3.0),
    )
    assert set(decision["expert_weights"]) == {"trend", "risk_parity"}
    assert decision["regime"] in {
        "up_high_vol", "up_low_vol", "down_high_vol", "down_low_vol"
    }


def test_failed_empty_organization_can_fall_back_to_benchmark():
    profiles = dict(DEFAULT_RESOURCE_PROFILES)
    profiles["trend"] = ResourceProfile(650, 180.0, 1.0)
    model = HMATFEmergentOrganization(
        3,
        OrganizationConfig(
            fixed_agents=("trend",),
            enforce_budgets=False,
            inject_failures=True,
        ),
        enabled_experts=("trend",),
        portfolio_config=MediaFactorConfig(
            max_asset_weight=0.80,
            benchmark_on_empty_allocation=True,
        ),
        resource_profiles=profiles,
    )
    prices = np.exp(np.linspace(0.0, 0.2, 253))[:, None] * np.array([[1.0, 1.2, 0.8]])
    benchmark = np.array([0.2, 0.3, 0.5])
    decision = model.rebalance(prices, np.eye(3) * 0.02, benchmark)
    assert decision["used_benchmark_fallback"]
    assert np.allclose(decision["target_weights"], benchmark)


def test_anchor_agent_is_selected_and_receives_minimum_weight():
    controller = BudgetedOrganizationController(
        OrganizationConfig(
            inject_failures=False,
            anchor_agent="risk_parity",
            anchor_weight=0.50,
        ),
        expert_names=("trend", "meanrev", "risk_parity"),
    )
    weights = controller.allocate(
        {"trend": True, "meanrev": True, "risk_parity": True}
    )
    assert "risk_parity" in controller.decision_history[-1]["active_agents"]
    assert weights["risk_parity"] >= 0.50
    assert np.isclose(sum(weights.values()), 1.0)
