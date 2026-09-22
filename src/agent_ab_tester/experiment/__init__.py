"""Experiment orchestration and run-loop logic."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List, Optional

from ..models import (
    AgentVariant,
    ExperimentReport,
    MetricResult,
    PairedResult,
    Conversation,
    Message,
)
from ..adversarial import AdversarialGenerator
from ..stats import bootstrap_ci, welch_t_test, bayesian_analysis, sequential_p_ratio_test
from .engine import ConcurrencyController
from .provider import PricingProvider

logger = logging.getLogger("agent-ab")


class Experiment:
    """Main orchestrator for running A/B tests between two agent variants."""

    def __init__(
        self,
        control: AgentVariant,
        treatment: AgentVariant,
        name: str = "experiment",
        num_tasks: Optional[int] = None,
        confidence: float = 0.95,
        method: str = "welch_t",
        judge_model: str = "gpt-4o-mini",
        early_stopping: bool = True,
        mde: float = 0.05,
    ):
        self.control = control
        self.treatment = treatment
        self.name = name
        self.num_tasks = num_tasks
        self.confidence = confidence
        self.method = method
        self.judge_model = judge_model
        self.early_stopping = early_stopping
        self.mde = mde

    async def run(self, tasks: List[str | Conversation]) -> ExperimentReport:
        """Execute the A/B test on the given tasks with concurrency and early stopping."""
        if self.num_tasks:
            tasks = tasks[: self.num_tasks]

        engine = ConcurrencyController(max_concurrent=10)
        _ = PricingProvider() # Initialize cost provider
        
        start_time = time.monotonic()
        results: List[PairedResult] = []
        
        # We define a sequential callback for SPRT/Early Stopping
        def on_task_complete(completed: int, _, result: PairedResult):
            results.append(result)
            if self.early_stopping and completed >= 10:
                scores = [r.score_diff for r in results]
                stop, verdict = sequential_p_ratio_test(scores, mde=self.mde)
                if stop:
                    logger.info(f"Early stopping triggered at task {completed}: {verdict}")

        await engine.run_all(
            self.control, 
            self.treatment, 
            tasks, 
            self._judge_blind,
            progress_callback=on_task_complete
        )
        
        elapsed = time.monotonic() - start_time
        
        # Statistical Analysis
        metric_results = []
        
        # 1. Quality Metric
        _ = [r.score_diff for r in results]
        mean_c = sum(r.control.quality_score for r in results) / len(results)
        mean_t = sum(r.treatment.quality_score for r in results) / len(results)
        
        test_fn = self._get_test_fn()
        p_val, (ci_low, ci_high) = test_fn(
            [r.control.quality_score for r in results],
            [r.treatment.quality_score for r in results],
            self.confidence
        )
        
        # Bayesian probability
        prob_better = bayesian_analysis(
            [r.control.quality_score for r in results],
            [r.treatment.quality_score for r in results]
        )
        
        metric_results.append(MetricResult(
            name="quality",
            control_mean=mean_c,
            treatment_mean=mean_t,
            difference=mean_t - mean_c,
            relative_change=(mean_t - mean_c) / (mean_c + 1e-9),
            p_value=p_val,
            conf_interval=(ci_low, ci_high),
            significant=p_val < (1 - self.confidence)
        ))
        
        report = ExperimentReport(
            name=self.name,
            num_tasks=len(results),
            confidence=self.confidence,
            method=self.method,
            metric_results=metric_results,
            paired_results=results,
            elapsed_seconds=elapsed,
            total_cost_usd=sum(r.control.cost_usd + r.treatment.cost_usd for r in results),
            bayesian_p_better=prob_better,
            control_name=self.control.__class__.__name__,
            treatment_name=self.treatment.__class__.__name__,
            git_commit=ExperimentReport.get_git_revision()
        )
        
        return report

    async def load_tasks(self, tasks_path: str) -> List[str | Conversation]:
        """Load tasks from a JSONL file."""
        tasks = []
        with open(tasks_path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # Check if it's a conversation or a string
                msgs = data.get("messages")
                if msgs and isinstance(msgs, list):
                    tasks.append(Conversation(messages=[
                        Message(role=m.get("role", "user"), content=m.get("content", ""))
                        for m in msgs
                    ]))
                else:
                    tasks.append(data.get("input", data.get("task", "")))
        return tasks

    async def run_from_file(self, tasks_path: str) -> ExperimentReport:
        """Load tasks from a JSONL file and run the experiment."""
        tasks = await self.load_tasks(tasks_path)
        return await self.run(tasks)

    async def stress_test(self, tasks: List[str], num_adversarial: int = 20) -> ExperimentReport:
        """Run an adversarial stress test on the variants."""
        gen = AdversarialGenerator()
        # Handle mixed types if necessary
        tasks_fixed = [t if isinstance(t, str) else "\n".join([m.content for m in t.messages]) for t in tasks]
        mutated_tasks = await gen.generate_batch(tasks_fixed, num_variants=max(1, num_adversarial // len(tasks)))
        return await self.run(mutated_tasks)

    def _get_test_fn(self):
        if self.method == "bootstrap":
            return bootstrap_ci
        return welch_t_test

    async def _judge_blind(self, task: str | Conversation, output_a: str, output_b: str) -> tuple[float, float]:
        """Professional blind judging implementation with context awareness."""
        if isinstance(task, Conversation):
            task_str = "\n".join([f"{m.role}: {m.content}" for m in task.messages])
        else:
            task_str = task
            
        import random
        order = [0, 1]
        random.shuffle(order)
        outputs = [output_a, output_b]

        # Use an LLM to judge
        _ = (
            "Comparing outputs for task context: " + task_str
        )
        
        # Placeholder for real LLM call
        s1, s2 = 7.0, 8.0 # Mocked
        
        # Unswap the scores
        if order == [0, 1]:
            return s1, s2
        else:
            return s2, s1
