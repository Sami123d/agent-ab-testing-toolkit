"""Concurrency engine for running experiments in parallel."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import AgentVariant, TaskOutcome, PairedResult, Conversation

logger = logging.getLogger("agent-ab")


class ConcurrencyController:
    """Manages parallel execution of agent variants with rate limiting."""

    def __init__(self, max_concurrent: int = 10, retry_delay: float = 1.0, max_failure_rate: float = 0.2):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.retry_delay = retry_delay
        self.max_failure_rate = max_failure_rate
        self.total_runs = 0
        self.total_failures = 0

    async def run_task_paired(
        self, 
        control: AgentVariant, 
        treatment: AgentVariant, 
        task: str | Conversation,
        judge_fn: callable
    ) -> PairedResult:
        """Run a single task through both variants in parallel within the quota."""
        async with self.semaphore:
            # We run the variants sequentially for the same task to simplify 
            # (less likely to hit global rate limits simultaneously)
            # but we can also parallelize them here if needed.
            
            c_outcome = await self._safe_run_variant(control, task)
            t_outcome = await self._safe_run_variant(treatment, task)
            
            # Step 2: Judging
            # We use another task for judging to avoid holding the semaphore too long
            # but judging also consumes LLM quota.
            c_score, t_score = await judge_fn(task, c_outcome.output, t_outcome.output)
            
            c_outcome.quality_score = c_score
            t_outcome.quality_score = t_score
            
            from ..models import PairedResult
            self.total_runs += 1
            if c_outcome.error or t_outcome.error:
                self.total_failures += 1
                
            if self.total_runs > 5 and (self.total_failures / self.total_runs) > self.max_failure_rate:
                raise RuntimeError(f"Circuit Breaker: High failure rate ({self.total_failures/self.total_runs:.1%}). Aborting.")

            return PairedResult(task_input=task, control=c_outcome, treatment=t_outcome)

    async def _safe_run_variant(self, variant: AgentVariant, task: str | Conversation) -> TaskOutcome:
        """Run a variant with basic retry logic for resilience."""
        from ..models import TaskOutcome
        start = time.monotonic()
        try:
            # Simple retry for 429-like errors or transient failures
            for attempt in range(3):
                try:
                    output = await variant.invoke(task)
                    latency = time.monotonic() - start
                    # Basic token estimation based on type
                    if isinstance(task, str):
                        input_len = len(task)
                    else:
                        input_len = sum(len(m.content) for m in task.messages)
                    tokens = (input_len + len(output)) // 4
                    return TaskOutcome(
                        task_input=task,
                        output=output,
                        latency_seconds=latency,
                        tokens=tokens,
                    )
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < 2:
                        wait = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit. Retrying in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"Variant execution failed: {e}")
            return TaskOutcome(
                task_input=task,
                output="",
                latency_seconds=time.monotonic() - start,
                error=str(e),
            )

    async def run_all(
        self,
        control: AgentVariant,
        treatment: AgentVariant,
        tasks: List[str | Conversation],
        judge_fn: callable,
        progress_callback: Optional[callable] = None
    ) -> List[PairedResult]:
        """Run all tasks through both variants in parallel batches."""
        coros = [self.run_task_paired(control, treatment, task, judge_fn) for task in tasks]
        
        results: List[PairedResult] = []
        # Run them with gathering for performance
        # Using as_completed to support progress updates
        for i, coro in enumerate(asyncio.as_completed(coros)):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(tasks), result)
                
        return results
