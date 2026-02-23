"""Prompt templates for storyteller workflow."""

from novel_summarizer.storyteller.prompts.entity import ENTITY_PROMPT_VERSION
from novel_summarizer.storyteller.prompts.narration import NARRATION_PROMPT_VERSION
from novel_summarizer.storyteller.prompts.state_mutation import STATE_MUTATION_PROMPT_VERSION

__all__ = ["ENTITY_PROMPT_VERSION", "NARRATION_PROMPT_VERSION", "STATE_MUTATION_PROMPT_VERSION"]