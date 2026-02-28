from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.llm.factory import OpenAIChatClient
from novel_summarizer.storyteller.nodes import (
    consistency_check,
    evidence_verify,
    entity_extract,
    memory_commit,
    memory_retrieve,
    refine_narration,
    state_lookup,
    state_update,
    storyteller_generate,
)
from novel_summarizer.storyteller.state import StorytellerState
from novel_summarizer.storage.repo import SQLAlchemyRepo


def build_storyteller_graph(
    *,
    repo: SQLAlchemyRepo,
    config: AppConfigRoot,
    book_id: int,
    entity_llm_client: OpenAIChatClient | None = None,
    narration_llm_client: OpenAIChatClient | None = None,
    refine_llm_client: OpenAIChatClient | None = None,
):
    workflow = StateGraph(StorytellerState)

    async def _entity_extract(state: StorytellerState) -> dict:
        return await entity_extract.run(state, config=config, llm_client=entity_llm_client)

    async def _state_lookup(state: StorytellerState) -> dict:
        return await state_lookup.run(state, repo=repo, config=config, book_id=book_id)

    async def _memory_retrieve(state: StorytellerState) -> dict:
        return await memory_retrieve.run(state, config=config, book_id=book_id)

    async def _storyteller_generate(state: StorytellerState) -> dict:
        return await storyteller_generate.run(state, config=config, llm_client=narration_llm_client)

    async def _consistency_check(state: StorytellerState) -> dict:
        return await consistency_check.run(state, config=config)

    async def _evidence_verify(state: StorytellerState) -> dict:
        return await evidence_verify.run(state, config=config)

    async def _refine_narration(state: StorytellerState) -> dict:
        return await refine_narration.run(state, config=config, llm_client=refine_llm_client)

    async def _state_update(state: StorytellerState) -> dict:
        return await state_update.run(state, repo=repo, config=config, book_id=book_id)

    async def _memory_commit(state: StorytellerState) -> dict:
        return await memory_commit.run(state, config=config, book_id=book_id)

    workflow.add_node("entity_extract", _entity_extract)
    workflow.add_node("state_lookup", _state_lookup)
    workflow.add_node("memory_retrieve", _memory_retrieve)
    workflow.add_node("storyteller_generate", _storyteller_generate)
    workflow.add_node("consistency_check", _consistency_check)
    workflow.add_node("evidence_verify", _evidence_verify)
    workflow.add_node("refine_narration", _refine_narration)
    workflow.add_node("state_update", _state_update)
    workflow.add_node("memory_commit", _memory_commit)

    workflow.add_edge(START, "entity_extract")
    workflow.add_edge("entity_extract", "state_lookup")
    workflow.add_edge("state_lookup", "memory_retrieve")
    workflow.add_edge("memory_retrieve", "storyteller_generate")
    workflow.add_edge("storyteller_generate", "consistency_check")
    workflow.add_edge("consistency_check", "evidence_verify")
    workflow.add_edge("evidence_verify", "refine_narration")
    workflow.add_edge("refine_narration", "state_update")
    workflow.add_edge("state_update", "memory_commit")
    workflow.add_edge("memory_commit", END)

    return workflow.compile()


def build_storyteller_draft_graph(
    *,
    config: AppConfigRoot,
    entity_llm_client: OpenAIChatClient | None = None,
    narration_llm_client: OpenAIChatClient | None = None,
    refine_llm_client: OpenAIChatClient | None = None,
):
    """Builds a graph that generates a narration draft without mutating persistent state.

    The caller is responsible for providing any world_state fields (character_states, item_states,
    recent_events, world_facts) in the input state.
    """

    workflow = StateGraph(StorytellerState)

    async def _entity_extract(state: StorytellerState) -> dict:
        return await entity_extract.run(state, config=config, llm_client=entity_llm_client)

    async def _memory_retrieve(state: StorytellerState) -> dict:
        # book_id is part of the state; keep API aligned with node.
        book_id = int(state.get("book_id") or 0)
        return await memory_retrieve.run(state, config=config, book_id=book_id)

    async def _storyteller_generate(state: StorytellerState) -> dict:
        return await storyteller_generate.run(state, config=config, llm_client=narration_llm_client)

    async def _consistency_check(state: StorytellerState) -> dict:
        return await consistency_check.run(state, config=config)

    async def _evidence_verify(state: StorytellerState) -> dict:
        return await evidence_verify.run(state, config=config)

    async def _refine_narration(state: StorytellerState) -> dict:
        return await refine_narration.run(state, config=config, llm_client=refine_llm_client)

    workflow.add_node("entity_extract", _entity_extract)
    workflow.add_node("memory_retrieve", _memory_retrieve)
    workflow.add_node("storyteller_generate", _storyteller_generate)
    workflow.add_node("consistency_check", _consistency_check)
    workflow.add_node("evidence_verify", _evidence_verify)
    workflow.add_node("refine_narration", _refine_narration)

    workflow.add_edge(START, "entity_extract")
    workflow.add_edge("entity_extract", "memory_retrieve")
    workflow.add_edge("memory_retrieve", "storyteller_generate")
    workflow.add_edge("storyteller_generate", "consistency_check")
    workflow.add_edge("consistency_check", "evidence_verify")
    workflow.add_edge("evidence_verify", "refine_narration")
    workflow.add_edge("refine_narration", END)

    return workflow.compile()