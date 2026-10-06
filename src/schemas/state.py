"""AgentCore Platform v1.0"""

# INS-C2-016 — State
#
# State must be a flat TypedDict, never a Pydantic model: LangGraph checkpoints
# use msgpack serialisation, and a rich object there corrupts silently rather
# than failing. Extend AgentState with agent-specific fields only, and never put
# credentials or secrets in it — every field is written to the checkpoint store.
#
# The two-layer nested composition means both layers share this one schema: the
# outer backbone writes validated_input / enriched_context / result, the inner
# workflow writes the retrieval fields.
#
# Every dict/list-valued field is stored as a JSON-serialised Optional[str] via
# the to_json() / from_json() helpers below — one contract at every producer and
# consumer. Typing such a field as a bare dict/list causes msgpack
# serialisation failures.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialise a value to a JSON string for storage in State."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialise a JSON string read back out of State."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for INS-C2-016.

    All shared fields (user_input, input_context, status, session_id,
    node_history, error_log, hitl_*, ...) are inherited from AgentState.
    formatted_output is inherited too and is not re-declared here.
    """

    # ------------------------------------------------------------------
    # Outer layer — written by PreProcessNode (pre_process backbone slot)
    # ------------------------------------------------------------------

    # Validated and normalised underwriting question.
    # Produced by PreProcessNode; consumed by the inner InputValidateNode.
    validated_input: NotRequired[Optional[str]]

    # JSON-serialised request metadata. Shape: {"source": str, "channel": str}.
    # channel is caller-supplied and is restricted to an inert token before it
    # is stored here.
    enriched_context: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer — the declared runtime settings and the retrieval pipeline
    # ------------------------------------------------------------------

    # JSON-serialised, already-validated settings declared in config/config.yaml
    # and seeded by DomainWorkflowGraph._extra_initial_state(). Node
    # constructors take no arguments and execute(state) takes no config
    # argument, so this is the route by which a declared value reaches a node.
    runtime_settings: NotRequired[Optional[str]]

    # Normalised retrieval query produced by InputValidateNode.
    retrieval_query: NotRequired[Optional[str]]

    # JSON-serialised list of retrieved candidate passages.
    # Each item: {doc_id, title, source, score, text}.
    retrieved_documents: NotRequired[Optional[str]]

    # JSON-serialised list of reranked, threshold-filtered evidence.
    # Same item shape as retrieved_documents; only passages at or above the
    # relative score threshold are kept.
    reranked_documents: NotRequired[Optional[str]]

    # JSON-serialised grounded answer object. Shape:
    # {answer_text: str, citations: list[str], confidence: str,
    #  grounded: bool, evidence_count: int}
    generated_answer: NotRequired[Optional[str]]

    # Final human-readable underwriting answer document (plain text, cited).
    # Assembled by the inner OutputFormatNode from generated_answer + evidence.
    rag_answer: NotRequired[Optional[str]]

    # Number of grounded evidence passages used to answer.
    retrieved_count: NotRequired[Optional[int]]

    # ------------------------------------------------------------------
    # Outer layer — written by PostProcessNode (post_process backbone slot)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller. Set to the answer document once
    # the output boundary has cleared it; withheld (replaced with a notice) if
    # it has not. formatted_output, inherited from AgentState, is set with it.
    result: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing — framework-managed; do NOT write these from node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    # node_history is inherited from AgentState
