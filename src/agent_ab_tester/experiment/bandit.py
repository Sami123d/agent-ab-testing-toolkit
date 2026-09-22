"""Multi-Armed Bandit optimizer for dynamic variant allocation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class VariantStats:
    """Historical data for a single variant in a bandit problem."""
    name: str
    successes: float = 1.0  # Prior success
    failures: float = 1.0   # Prior failure
    total: int = 0
    sum_scores: float = 0.0


class ThompsonSampler:
    """Implements Thompson Sampling for balancing Exploration vs Exploitation."""

    def __init__(self, variant_names: List[str]):
        self.variants = {name: VariantStats(name) for name in variant_names}

    def select_variant(self) -> str:
        """Pick the best variant by sampling from their Beta/Normal posteriors."""
        best_score = -1.0
        best_variant = list(self.variants.keys())[0]

        for name, stats in self.variants.items():
            mean = stats.sum_scores / max(stats.total, 1)
            uncertainty = 2.0 / math.sqrt(max(stats.total, 1))
            sample = random.gauss(mean, uncertainty)
            
            if sample > best_score:
                best_score = sample
                best_variant = name

        return best_variant

    def update(self, variant_name: str, score: float):
        """Update stats after a variant run."""
        if variant_name in self.variants:
            stats = self.variants[variant_name]
            stats.total += 1
            stats.sum_scores += score


class BanditExperiment:
    """Orchestrator for a bandit-optimized run."""
    
    def __init__(self, variants: Dict[str, callable], tasks: List[str]):
        self.variants = variants
        self.tasks = tasks
        self.sampler = ThompsonSampler(list(variants.keys()))
        self.results = []

    async def run(self, max_runs: int = 100):
        """Execute the optimized run."""
        pass
