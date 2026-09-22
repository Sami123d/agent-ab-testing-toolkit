"""Pricing and metadata provider for agent-ab-tester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelPricing:
    input_price_per_1m: float
    output_price_per_1m: float


class PricingProvider:
    """Provides up-to-date pricing for various LLM models."""

    # Default pricing (March 2026)
    _DEFAULTS: Dict[str, ModelPricing] = {
        "gpt-4o": ModelPricing(2.50, 10.00),
        "gpt-4o-mini": ModelPricing(0.15, 0.60),
        "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00),
        "claude-3-5-haiku-20241022": ModelPricing(1.00, 5.00),
        "gemini-2.0-flash": ModelPricing(0.10, 0.40),
        "gemini-1.5-pro": ModelPricing(1.25, 5.00),
    }

    def __init__(self, custom_pricing: Dict[str, ModelPricing] | None = None):
        self.pricing = self._DEFAULTS.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)

    def get_pricing(self, model_name: str) -> ModelPricing:
        """Get pricing for a model. Defaults to gpt-4o-mini if unknown."""
        # Clean model name (handle common aliases/versions)
        clean_name = model_name.lower().split(":")[0]
        
        # Try exact match
        if clean_name in self.pricing:
            return self.pricing[clean_name]
        
        # Try substring match
        for key in self.pricing:
            if key in clean_name:
                return self.pricing[key]
        
        # Fallback
        return self.pricing["gpt-4o-mini"]

    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate total USD cost for a run."""
        pricing = self.get_pricing(model_name)
        input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_1m
        return input_cost + output_cost
