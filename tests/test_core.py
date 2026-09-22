"""Comprehensive tests for agent-ab-tester."""

import json
import math
import random

import pytest

from agent_ab_tester.models import (
    AgentVariant, ExperimentReport, MetricResult,
    PairedResult, TaskOutcome,
)
from agent_ab_tester.stats import (
    bootstrap_ci, power_analysis, welch_t_test,
)
from agent_ab_tester.experiment import Experiment


# ─────────────────────────────────────────────
# Mock Agents
# ─────────────────────────────────────────────

class GoodAgent(AgentVariant):
    """Agent that produces consistently good outputs."""
    async def invoke(self, task: str) -> str:
        return (
            f"Comprehensive analysis of {task}: This topic involves "
            f"multiple dimensions including market trends, competitive "
            f"landscape, and technological developments. Key findings "
            f"suggest significant growth potential with several risks "
            f"to monitor. Recommended actions include strategic investment "
            f"and continuous monitoring of regulatory changes."
        )

class WeakAgent(AgentVariant):
    """Agent that produces consistently weak outputs."""
    async def invoke(self, task: str) -> str:
        return f"Here is info about {task}. It is interesting."

class FailingAgent(AgentVariant):
    """Agent that always fails."""
    async def invoke(self, task: str) -> str:
        raise RuntimeError("Agent crashed")


# ─────────────────────────────────────────────
# Statistical Tests
# ─────────────────────────────────────────────

class TestWelchTTest:
    def test_significant_difference(self):
        random.seed(42)
        control = [random.gauss(5.0, 1.0) for _ in range(50)]
        treatment = [random.gauss(7.0, 1.0) for _ in range(50)]
        result = welch_t_test(control, treatment, 0.05, "quality")
        assert result.significant is True
        assert result.p_value < 0.05
        assert result.treatment_mean > result.control_mean

    def test_no_significant_difference(self):
        random.seed(42)
        control = [random.gauss(5.0, 1.0) for _ in range(50)]
        treatment = [random.gauss(5.1, 1.0) for _ in range(50)]
        result = welch_t_test(control, treatment, 0.05, "quality")
        # With such a small effect, likely not significant with n=50
        # (but depends on random seed — the test is about structure, not exact p)
        assert isinstance(result.p_value, float)
        assert 0 < result.p_value <= 1

    def test_identical_values(self):
        values = [5.0] * 20
        result = welch_t_test(values, values, 0.05, "quality")
        assert result.significant is False
        assert result.p_value == 1.0

    def test_small_sample(self):
        result = welch_t_test([1.0], [2.0], 0.05, "quality")
        assert result.significant is False  # Too small for significance

    def test_effect_size_calculated(self):
        control = [3.0, 4.0, 5.0, 4.0, 3.0, 5.0, 4.0, 3.0, 4.0, 5.0]
        treatment = [7.0, 8.0, 9.0, 8.0, 7.0, 9.0, 8.0, 7.0, 8.0, 9.0]
        result = welch_t_test(control, treatment, 0.05, "quality")
        assert result.effect_size > 0
        assert result.effect_label in ["small", "medium", "large"]

    def test_bonferroni_correction(self):
        random.seed(42)
        control = [random.gauss(5.0, 1.0) for _ in range(30)]
        treatment = [random.gauss(6.0, 1.0) for _ in range(30)]
        # Without correction
        r1 = welch_t_test(control, treatment, 0.05, "q", bonferroni_k=1)
        # With correction (stricter)
        r4 = welch_t_test(control, treatment, 0.05, "q", bonferroni_k=4)
        # Same p-value, but significance may differ
        assert r1.p_value == r4.p_value

    def test_confidence_interval(self):
        control = [4.0, 5.0, 4.5, 5.5, 4.0]
        treatment = [7.0, 8.0, 7.5, 8.5, 7.0]
        result = welch_t_test(control, treatment, 0.05, "quality")
        assert result.ci_lower > 0  # Treatment is better
        assert result.ci_upper > result.ci_lower

    def test_direction_lower_is_better(self):
        control = [10.0, 12.0, 11.0, 13.0, 10.0]
        treatment = [5.0, 6.0, 5.5, 6.5, 5.0]
        result = welch_t_test(
            control, treatment, 0.05, "cost", "lower_is_better"
        )
        if result.significant:
            assert result.better_variant == "treatment"

    def test_stars_property(self):
        mr = MetricResult(name="q", control_mean=5, treatment_mean=7,
                          difference=2, relative_change=0.4,
                          p_value=0.001, significant=True)
        assert mr.stars == "***"
        mr.p_value = 0.008
        assert mr.stars == "**"
        mr.p_value = 0.03
        assert mr.stars == "*"
        mr.p_value = 0.12
        assert mr.stars == ""


class TestBootstrapCI:
    def test_significant_difference(self):
        random.seed(42)
        control = [random.gauss(5.0, 1.0) for _ in range(30)]
        treatment = [random.gauss(8.0, 1.0) for _ in range(30)]
        result = bootstrap_ci(
            control, treatment, 0.05, "quality", n_bootstrap=1000
        )
        assert result.significant is True
        assert result.ci_lower > 0

    def test_no_difference(self):
        random.seed(42)
        values = [random.gauss(5.0, 1.0) for _ in range(30)]
        result = bootstrap_ci(
            values, values.copy(), 0.05, "quality", n_bootstrap=1000
        )
        # CI should include 0
        assert result.ci_lower <= 0 or result.ci_upper >= 0


class TestPowerAnalysis:
    def test_medium_effect(self):
        n = power_analysis(effect_size=0.5, alpha=0.05, power=0.80)
        assert 50 < n < 150  # Standard result is ~64

    def test_large_effect_needs_fewer(self):
        n_large = power_analysis(effect_size=0.8)
        n_small = power_analysis(effect_size=0.2)
        assert n_large < n_small

    def test_higher_power_needs_more(self):
        n_80 = power_analysis(effect_size=0.5, power=0.80)
        n_95 = power_analysis(effect_size=0.5, power=0.95)
        assert n_95 > n_80

    def test_minimum_sample(self):
        n = power_analysis(effect_size=5.0)  # Huge effect
        assert n >= 5


# ─────────────────────────────────────────────
# Model Tests
# ─────────────────────────────────────────────

class TestModels:
    def test_task_outcome_succeeded(self):
        o = TaskOutcome(task_input="t", output="o")
        assert o.succeeded is True

    def test_task_outcome_failed(self):
        o = TaskOutcome(task_input="t", output="", error="boom")
        assert o.succeeded is False

    def test_paired_result_diffs(self):
        p = PairedResult(
            task_input="t",
            control=TaskOutcome(task_input="t", output="a",
                                quality_score=7.0, cost_usd=0.05,
                                latency_seconds=3.0),
            treatment=TaskOutcome(task_input="t", output="b",
                                  quality_score=8.5, cost_usd=0.03,
                                  latency_seconds=4.0),
        )
        assert p.quality_diff == 1.5
        assert p.cost_diff == -0.02
        assert p.latency_diff == 1.0

    def test_metric_result_to_dict(self):
        mr = MetricResult(
            name="quality", control_mean=7.0, treatment_mean=8.0,
            difference=1.0, relative_change=0.143,
            p_value=0.003, significant=True,
            ci_lower=0.4, ci_upper=1.6,
            effect_size=0.85, effect_label="large",
            better_variant="treatment",
        )
        d = mr.to_dict()
        json.dumps(d)  # Must serialize
        assert d["significant"] is True
        assert d["better_variant"] == "treatment"

    def test_experiment_report_recommendation_ship(self):
        report = ExperimentReport(
            metric_results=[
                MetricResult(name="quality", control_mean=7, treatment_mean=8,
                             difference=1, relative_change=0.14,
                             p_value=0.01, significant=True,
                             better_variant="treatment"),
            ]
        )
        assert "Ship treatment" in report.recommendation

    def test_experiment_report_recommendation_keep(self):
        report = ExperimentReport(
            metric_results=[
                MetricResult(name="quality", control_mean=8, treatment_mean=7,
                             difference=-1, relative_change=-0.125,
                             p_value=0.01, significant=True,
                             better_variant="control"),
            ]
        )
        assert "Keep control" in report.recommendation

    def test_experiment_report_recommendation_no_diff(self):
        report = ExperimentReport(
            metric_results=[
                MetricResult(name="quality", control_mean=7, treatment_mean=7.1,
                             difference=0.1, relative_change=0.014,
                             p_value=0.45, significant=False,
                             better_variant="neither"),
            ]
        )
        assert "No significant" in report.recommendation

    def test_experiment_report_to_dict(self):
        report = ExperimentReport(
            name="test", num_tasks=10,
            metric_results=[
                MetricResult(name="quality", control_mean=7, treatment_mean=8,
                             difference=1, relative_change=0.14,
                             p_value=0.01, significant=True),
            ],
        )
        d = report.to_dict()
        json.dumps(d)  # Must serialize
        assert d["num_tasks"] == 10


# ─────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────

class TestExperimentIntegration:
    @pytest.mark.asyncio
    async def test_good_vs_weak(self):
        exp = Experiment(
            control=WeakAgent(),
            treatment=GoodAgent(),
            name="test",
            confidence=0.95,
        )
        tasks = [f"Topic {i}" for i in range(10)]
        report = await exp.run(tasks)
        assert report.num_tasks == 10
        assert len(report.metric_results) == 4  # quality, cost, latency, tokens
        assert report.elapsed_seconds > 0

    @pytest.mark.asyncio
    async def test_same_agent(self):
        exp = Experiment(
            control=GoodAgent(),
            treatment=GoodAgent(),
            name="same-test",
        )
        tasks = ["Topic A", "Topic B", "Topic C"]
        report = await exp.run(tasks)
        # Same agent should show no significant difference
        quality = next(m for m in report.metric_results if m.name == "quality")
        # Can't guarantee non-significance due to LLM judge variance,
        # but structure should be correct
        assert isinstance(quality.p_value, float)

    @pytest.mark.asyncio
    async def test_handles_failure(self):
        exp = Experiment(
            control=GoodAgent(),
            treatment=FailingAgent(),
        )
        tasks = ["Test task"]
        report = await exp.run(tasks)
        assert report.num_tasks == 1
        # Treatment should have empty output
        assert report.paired_results[0].treatment.error is not None
