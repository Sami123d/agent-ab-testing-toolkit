"""agent-ab-tester: A/B test your AI agents with statistical rigor."""

from .models import AgentVariant, ExperimentReport, MetricResult, PairedResult, TaskOutcome
from .experiment import Experiment
from .stats import bootstrap_ci, power_analysis, welch_t_test

__version__ = "0.1.0"
__all__ = [
    "AgentVariant", "Experiment", "ExperimentReport",
    "MetricResult", "PairedResult", "TaskOutcome",
    "bootstrap_ci", "power_analysis", "welch_t_test",
]
