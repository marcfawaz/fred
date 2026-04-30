from __future__ import annotations

from typing import get_type_hints
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import START

from agentic_backend.agents.v1.production.aegis.aegis_rag_expert import Aegis
from agentic_backend.agents.v1.production.aegis.structures import AegisGraphState


def _build_subject() -> Aegis:
    """Why this exists: build an Aegis instance without runtime init for contract tests.

    How to use:
        Call in unit tests that only need graph wiring or pure node methods.
    """
    return object.__new__(Aegis)


class _ModelCopyOnly:
    """Why this exists: exercise the pydantic-v2-style non-streaming clone path.

    How to use:
        Instantiate in tests and inspect `seen_update` after calling the Aegis
        helper to confirm it requested `disable_streaming=True`.
    """

    def __init__(self) -> None:
        self.seen_update: dict[str, object] | None = None

    def model_copy(self, *, update: dict[str, object]) -> "_ModelCopyOnly":
        self.seen_update = update
        clone = _ModelCopyOnly()
        clone.seen_update = update
        return clone


class _CopyOnly:
    """Why this exists: exercise the pydantic-v1-style non-streaming clone path.

    How to use:
        Instantiate in tests and inspect `seen_update` after calling the Aegis
        helper to confirm it requested `disable_streaming=True`.
    """

    def __init__(self) -> None:
        self.seen_update: dict[str, object] | None = None

    def copy(self, *, update: dict[str, object]) -> "_CopyOnly":
        self.seen_update = update
        clone = _CopyOnly()
        clone.seen_update = update
        return clone


class _BindOnly:
    """Why this exists: exercise the bind(stream=False) internal-model fallback path.

    How to use:
        Instantiate in tests and inspect `seen_kwargs` after calling the Aegis
        helper to confirm it requested `stream=False`.
    """

    def __init__(self) -> None:
        self.seen_kwargs: dict[str, object] | None = None

    def bind(self, **kwargs: object) -> "_BindOnly":
        self.seen_kwargs = kwargs
        clone = _BindOnly()
        clone.seen_kwargs = kwargs
        return clone


@pytest.mark.asyncio
async def test_finalize_success_returns_final_message_delta() -> None:
    agent = _build_subject()
    answer = AIMessage(content="Final answer")
    state: AegisGraphState = {
        "draft_answer": answer,
        "sources": [],
        "documents": ["sentinel"],  # type: ignore[list-item]
        "iteration": 2,
        "self_check": None,
        "followup_queries": ["q"],
        "decision": "corrective_retrieve",
        "irrelevant_documents": ["sentinel"],  # type: ignore[list-item]
    }

    update = await agent._finalize_success(state)

    [final] = update["messages"]
    assert isinstance(final, AIMessage)
    assert final is not answer
    assert final.content == "Final answer"
    assert final.response_metadata["extras"]["agent"] == "Aegis"
    assert final.response_metadata["extras"]["final"] is True
    assert update["draft_answer"] is None
    assert update["documents"] == []
    assert update["sources"] == []
    assert update["iteration"] == 0
    assert update["followup_queries"] == []
    assert update["decision"] is None
    assert update["irrelevant_documents"] == []
    assert "messages" not in state


@pytest.mark.asyncio
async def test_finalize_best_effort_returns_final_message_delta() -> None:
    agent = _build_subject()
    answer = AIMessage(content="Best effort")
    state: AegisGraphState = {
        "draft_answer": answer,
        "sources": [],
        "documents": [],
        "iteration": 1,
        "self_check": None,
        "followup_queries": ["q"],
        "decision": "finalize_best_effort",
        "irrelevant_documents": [],
    }

    update = await agent._finalize_best_effort(state)

    [final] = update["messages"]
    assert isinstance(final, AIMessage)
    assert final is not answer
    assert final.content == "Best effort"
    assert final.response_metadata["extras"]["agent"] == "Aegis"
    assert final.response_metadata["extras"]["final"] is True
    assert update["draft_answer"] is None
    assert update["documents"] == []
    assert update["sources"] == []
    assert update["iteration"] == 0
    assert update["followup_queries"] == []
    assert update["decision"] is None
    assert update["irrelevant_documents"] == []
    assert "messages" not in state


@pytest.mark.asyncio
async def test_finalize_success_uses_fallback_message_when_draft_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _build_subject()
    agent._get_text_content = MagicMock(return_value="")  # type: ignore[method-assign]

    with caplog.at_level("INFO"):
        update = await agent._finalize_success({"sources": []})

    [message] = update["messages"]
    assert isinstance(message, AIMessage)
    assert "Aegis could not produce a final answer" in message.content
    assert message.response_metadata["extras"]["final"] is True
    assert "[AEGIS][FINAL_EMIT] mode=success" in caplog.text


def test_make_internal_model_prefers_model_copy() -> None:
    agent = _build_subject()
    model = _ModelCopyOnly()

    clone = agent._make_internal_model(model)

    assert clone is not model
    assert model.seen_update == {"disable_streaming": True}


def test_make_internal_model_uses_copy_fallback() -> None:
    agent = _build_subject()
    model = _CopyOnly()

    clone = agent._make_internal_model(model)

    assert clone is not model
    assert model.seen_update == {"disable_streaming": True}


def test_make_internal_model_uses_bind_fallback() -> None:
    agent = _build_subject()
    model = _BindOnly()

    clone = agent._make_internal_model(model)

    assert clone is not model
    assert model.seen_kwargs == {"stream": False}


def test_make_internal_model_returns_original_when_copy_is_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _build_subject()
    model = object()

    with caplog.at_level("WARNING"):
        clone = agent._make_internal_model(model)

    assert clone is model
    assert "could not create internal non-streaming model" in caplog.text


def test_build_final_answer_message_preserves_safe_metadata_only() -> None:
    agent = _build_subject()
    draft = AIMessage(
        content="Final answer",
        response_metadata={
            "model_name": "m",
            "finish_reason": "stop",
            "token_usage": {"total_tokens": 3},
            "extras": {"node": "generate_draft"},
            "tool_calls": [{"ignored": True}],
        },
    )

    final = agent._build_final_answer_message({"draft_answer": draft}, [])

    assert final is not draft
    assert final.content == "Final answer"
    assert final.response_metadata["model_name"] == "m"
    assert final.response_metadata["finish_reason"] == "stop"
    assert final.response_metadata["token_usage"] == {"total_tokens": 3}
    assert final.response_metadata["extras"]["node"] == "generate_draft"
    assert final.response_metadata["extras"]["agent"] == "Aegis"
    assert final.response_metadata["extras"]["final"] is True
    assert "tool_calls" not in final.response_metadata


def test_build_graph_uses_start_edge_for_retrieve() -> None:
    agent = _build_subject()

    graph = agent._build_graph()

    assert (START, "retrieve") in graph.edges


def test_aegis_graph_state_uses_langgraph_message_accumulator() -> None:
    hints = get_type_hints(AegisGraphState, include_extras=True)

    assert "add_messages" in repr(hints["messages"])
