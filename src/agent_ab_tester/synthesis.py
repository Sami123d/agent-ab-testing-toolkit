"""Synthetic Task Generation for cold-start A/B testing."""

from __future__ import annotations

from typing import List, Optional


class TaskSynthesizer:
    """Generates a diverse set of synthetic tasks for AI agents."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    async def generate_tasks(self, domain_description: str, num_tasks: int = 10, seed_tasks: Optional[List[str]] = None) -> List[str]:
        """Generate N diverse tasks for a specific domain."""
        
        system_prompt = (
            "You are an expert User Researcher at a top AI company. "
            f"Your goal is to generate {num_tasks} diverse, realistic, and challenging tasks "
            f"for an AI agent in the {domain_description} domain."
        )
        
        user_prompt = "Generate the tasks as a JSON list of strings. Each task should be unique."
        if seed_tasks:
            user_prompt += "\n\nBase your generation on these seed examples:\n" + "\n".join(seed_tasks)
            
        # Simulated LLM call logic
        logger_payload = {"system": system_prompt, "user": user_prompt}
        mock_tasks = [
            f"Synthetic Task for {domain_description} #{i+1}: "
            f"How would a user handle a complex query about {domain_description}?"
            for i in range(num_tasks)
        ]
        return mock_tasks

    async def expand_dataset(self, existing_tasks: List[str], target_size: int = 50) -> List[str]:
        """Take a small set of real tasks and bootstrap them into a large evaluation suite."""
        num_to_generate = target_size - len(existing_tasks)
        if num_to_generate <= 0:
            return existing_tasks
            
        new_tasks = await self.generate_tasks(
            domain_description="Existing Task Patterns", 
            num_tasks=num_to_generate, 
            seed_tasks=existing_tasks
        )
        return existing_tasks + new_tasks
