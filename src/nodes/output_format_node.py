"""AgentCore Platform v1.0"""

# INS-C2-016 — OutputFormatNode (inner domain node 5, last in the inner graph)
# Assembles the final underwriting answer document from generated_answer plus
# the reranked evidence, and enforces this template's stated output invariant
# before the document leaves the inner graph.
#
# The invariant: every citation in the rendered document is the doc_id of a
# passage in the evidence set. That is what "grounded" means here — the agent
# answers from retrieved guideline passages and never invents a risk score or
# premium — and it is the only output invariant this template claims. It
# renders no monetary aggregates, so the rounding grid some templates enforce
# does not apply; this check is enforced in its place.
#
# The check is cheap and it is not redundant with the composition step: the two
# are separated precisely so a future change to composition cannot quietly
# start emitting an uncited determination.
#
# Inner node — ANONYMOUS trust.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json
from src.services.source_disclosure import source_label

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72


def _uncited(answer_obj: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    """Return the citations that name no passage in the evidence set.

    Empty list = the grounding invariant holds. A non-empty list means the
    document would attribute a determination to a passage it was not given.
    """
    evidence_ids = {str(item.get("doc_id", "")) for item in evidence}
    citations = answer_obj.get("citations") or []
    return sorted({str(c) for c in citations} - evidence_ids)


def _assemble(
    answer_obj: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    """Assemble the final cited underwriting answer document."""
    confidence = str(answer_obj.get("confidence", "low")).upper()
    grounded = bool(answer_obj.get("grounded", False))
    citations = answer_obj.get("citations", []) or []

    lines = [
        _SEPARATOR,
        "INSURANCE UNDERWRITING — GROUNDED ANSWER",
        f"Grounded: {'YES' if grounded else 'NO (insufficient evidence)'}   Confidence: {confidence}",
        _SEPARATOR,
        "",
        str(answer_obj.get("answer_text", "")).strip(),
        "",
        _SUBSEP,
        "Citations:",
    ]
    if citations:
        for c in citations:
            lines.append(f"  - [{c}]")
    else:
        lines.append("  (none — no grounded evidence)")

    if evidence:
        lines += ["", _SUBSEP, "Evidence used:"]
        for e in evidence:
            lines.append(
                f"  [{e.get('doc_id', 'N/A')}] {e.get('title', '')} "
                f"({e.get('source', '')}, score={float(e.get('score', 0.0)):.2f})"
            )
    lines.append(_SEPARATOR)
    return "\n".join(lines)


class OutputFormatNode(FunctionNode):
    """Assemble the final grounded underwriting answer document (inner node).

    Reads generated_answer and reranked_documents from State, checks the
    grounding invariant, renders the cited answer text, and writes it to
    rag_answer (and result) for the outer PostProcessNode output boundary.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        generated_answer:   str  — JSON-serialised answer object
        reranked_documents: str  — JSON-serialised evidence list

    Output state keys (partial dict):
        rag_answer: str
        result:     str  (same as rag_answer — backbone convention)
        status:     str
        error_log:  list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        answer_obj: dict[str, Any] = from_json(state.get("generated_answer"), {})
        evidence: list[dict[str, Any]] = from_json(state.get("reranked_documents"), [])

        if not answer_obj:
            logger.error("OutputFormatNode: generated_answer missing in state")
            emit_trace_event(
                "output_format_failed",
                {"reason": "missing_generated_answer"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: generated_answer missing in state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("OutputFormatNode: generated_answer missing in state"),
            }

        uncited = _uncited(answer_obj, evidence)
        if uncited:
            logger.error("OutputFormatNode: %d citation(s) name no supplied passage", len(uncited))
            emit_trace_event(
                "output_format_grounding_violation",
                {"uncited_citation_count": len(uncited)},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: answer cites passages that are not in the evidence set"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("OutputFormatNode: answer cites passages that are not in the evidence set"),
            }

        document = _assemble(answer_obj, evidence)

        logger.info(
            "OutputFormatNode: answer document assembled (chars=%d, grounded=%s)",
            len(document),
            answer_obj.get("grounded", False),
        )
        # Say where the answer came from. This agent answers from a corpus defined inside
        # its own module; a reader seeing a citation has no way to tell that from a live query
        # against the system of record, and the review round rated that confusion its most
        # serious finding. Added before the gate below so it passes the same checks the answer
        # does.
        document = document + source_label(state)
        emit_trace_event(
            "output_format_complete",
            {
                "document_length": len(document),
                "citation_count": len(answer_obj.get("citations", []) or []),
                "grounded": bool(answer_obj.get("grounded", False)),
            },
            state,
        )

        return {
            "rag_answer": document,
            "result": document,
            "status": AgentStatus.SUCCESS.value,
        }
