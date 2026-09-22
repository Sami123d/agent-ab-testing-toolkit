"""Data models for A/B testing agents."""

from __future__ import annotations

import subprocess
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class Message:
    """A single message in a conversation."""
    role: str # "user", "assistant", "system"
    content: str


@dataclass
class Conversation:
    """A list of messages representing a multi-turn task."""
    messages: List[Message]
    
    @classmethod
    def from_text(cls, text: str) -> Conversation:
        return cls(messages=[Message(role="user", content=text)])


class AgentVariant(ABC):
    """Abstract base class for an agent variant (e.g. Prompt A, Model B)."""

    @abstractmethod
    def setup(self):
        """Initialize the agent. Called once before all tasks."""

    @abstractmethod
    async def invoke(self, task: Union[str, Conversation]) -> str:
        """Run the agent on a task and return the output string."""
        ...


@dataclass
class TaskOutcome:
    """Result from running one variant on one task."""

    task_input: Union[str, Conversation]
    output: str
    quality_score: float = 0.0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    tokens: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class PairedResult:
    """Paired results for a single task: control vs treatment."""

    task_input: Union[str, Conversation]
    control: TaskOutcome
    treatment: TaskOutcome

    @property
    def score_diff(self) -> float:
        return self.treatment.quality_score - self.control.quality_score


@dataclass
class MetricResult:
    """Summary statistics for a single metric."""

    name: str
    control_mean: float
    treatment_mean: float
    difference: float
    relative_change: float # (T-C)/C
    p_value: float
    conf_interval: tuple[float, float]
    significant: bool

    def to_dict(self) -> dict:
        return {
            "metric": self.name,
            "control": round(self.control_mean, 4),
            "treatment": round(self.treatment_mean, 4),
            "diff": round(self.difference, 4),
            "rel_change": f"{self.relative_change:+.1%}",
            "p_value": round(self.p_value, 4),
            "ci": [round(self.conf_interval[0], 4), round(self.conf_interval[1], 4)],
            "sig": self.significant,
        }


@dataclass
class ExperimentReport:
    """Final report containing all metadata and statistical results."""

    name: str
    num_tasks: int
    confidence: float
    method: str
    metric_results: list[MetricResult]
    paired_results: list[PairedResult]
    
    # Telemetry
    total_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    control_name: str = "Control"
    treatment_name: str = "Treatment"
    bayesian_p_better: float = 0.0 # P(Treatment > Control)
    version: str = "1.0.0"
    git_commit: Optional[str] = None

    @property
    def quality_pvalue(self) -> float:
        for m in self.metric_results:
            if m.name == "quality":
                return m.p_value
        return 1.0

    @property
    def recommendation(self) -> str:
        sig = any(m.significant and m.difference > 0 for m in self.metric_results if m.name == "quality")
        better_p = self.bayesian_p_better > 0.95
        
        if sig or better_p:
            return "Ship treatment! Statistically superior quality."
        
        # Check cost/latency if quality is parity
        quality_parity = self.quality_pvalue > 0.1
        cost_savings = any(m.difference < 0 and m.significant for m in self.metric_results if m.name == "cost")
        
        if quality_parity and cost_savings:
            return "Ship treatment — same quality, lower cost"
        return "No significant difference — keep control (safer)"

    @classmethod
    def get_git_revision(cls) -> Optional[str]:
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        except Exception:
            return None

    def to_dict(self) -> dict:
        return {
            "experiment": self.name,
            "version": self.version,
            "git_commit": self.git_commit,
            "num_tasks": self.num_tasks,
            "confidence": self.confidence,
            "method": self.method,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "bayesian_p_better": round(self.bayesian_p_better, 4),
            "recommendation": self.recommendation,
            "metrics": [m.to_dict() for m in self.metric_results],
        }
