"""AgentCore Platform v1.0"""

# INS-C2-016 — PostProcessNode
# Outer backbone post_process slot: the output boundary. It screens the
# assembled underwriting answer for credential material and, only if it is
# clean, exposes it as formatted_output and result.
#
# Two properties this node exists to hold:
#
#   1. The screen is a SUPERSET of the framework's own credential detector.
#      The framework re-scans every value of every node result and RAISES on a
#      match; a raise is caught by the node wrapper, which returns a bare error
#      partial and discards whatever this node decided. So a pattern the
#      framework knows and this node does not is not a smaller net — it is a
#      bypass of the containment below. detect_credentials() is called directly
#      here so the two sets can never drift apart.
#
#   2. On a violation the node CLEARS every output-bearing state field rather
#      than only marking the status. The framework's get_output() resolves the
#      caller's output as formatted_output or result, with no status check, so
#      leaving the ungated document in state would ship it inside the error
#      envelope. The withheld notice is deliberately non-empty: an empty
#      formatted_output is falsy and re-activates that same fallback.
#
# The domain screen is a module-level function (_security_gate_output) called
# from inside execute() — not an instance method on the node class, which the
# framework marks final and refuses to let a domain node override.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security.credential_detector import detect_credentials
from shared.utils.audit_logger import emit_trace_event
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

logger = logging.getLogger(__name__)

# Domain patterns ON TOP OF the framework detector — never instead of it.
# A `password = ...` assignment is not a credential SHAPE the framework
# recognises, but it is exactly what a mis-pasted operations note in a
# guideline document looks like.
_EXTRA_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
        "credential_assignment",
    ),
)

# Every state field that can carry the ungated answer. On a violation each one
# is cleared, so no representation of the answer survives in state for
# get_output() — or any later consumer — to fall back to. Inert provenance
# (retrieved_count) is not listed: it is an integer and carries no answer text.
_OUTPUT_BEARING_FIELDS = ("rag_answer", "generated_answer", "reranked_documents")

_WITHHELD_NOTICE = (
    "[UNDERWRITING ANSWER WITHHELD: the assembled answer did not clear the "
    "output boundary and has not been released. Route this request to a manual "
    "underwriter and quote its correlation id.]"
)

_EMPTY_ANSWER_NOTICE = (
    "[Underwriting] No grounded answer was produced. Check error_log for "
    "upstream failures; route to manual underwriter review."
)


def _security_gate_output(value: Any) -> Optional[str]:
    """Scan *value* for credential material; return the finding type or None.

    Walks dicts, lists and tuples depth-first so a credential riding inside a
    nested structure is seen — a scan of top-level strings only would report
    zero findings on exactly the payload shape that carries one. Every string
    leaf is screened with the framework's detector first, then with the domain
    patterns above.
    """
    return _scan(value, 0)


def _scan(value: Any, depth: int) -> Optional[str]:
    if depth > 8:
        return None
    if isinstance(value, str):
        findings = detect_credentials(value)
        if findings:
            return str(findings[0]["type"])
        for pattern, name in _EXTRA_PATTERNS:
            if pattern.search(value):
                return name
        return None
    if isinstance(value, dict):
        for nested in value.values():
            found = _scan(nested, depth + 1)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _scan(item, depth + 1)
            if found:
                return found
    return None


class PostProcessNode(FunctionNode):
    """Apply the output boundary and expose the final underwriting answer.

    Outer backbone post_process slot. Declared ANONYMOUS — trust was already
    enforced at PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        rag_answer: str  — the assembled answer document from OutputFormatNode
        result:     str  — the same content (backbone convention)

    Output state keys (partial dict):
        formatted_output: str
        result:           str
        status:           str
        error_log:        list[str]  (only on ERROR)
        rag_answer / generated_answer / reranked_documents — cleared on a violation
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        answer: str = state.get("rag_answer") or state.get("result") or ""

        if not answer.strip():
            logger.warning("PostProcessNode: no answer content — using the fallback notice")
            answer = _EMPTY_ANSWER_NOTICE

        # Screen the rendered document AND every other output-bearing field.
        # The document is the caller-visible representation, but the same
        # material also rides in the structured intermediates, and those are
        # what the framework's own re-scan sees.
        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=answer,
            domain="INS InsuranceUnderwritingRiskScoringAgent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if _review and isinstance(answer, str) and not _security_gate_output({"rag_answer": answer + _review}):
            answer = answer + _review

        candidate: dict[str, Any] = {"rag_answer": answer}
        for field in _OUTPUT_BEARING_FIELDS:
            if field != "rag_answer":
                candidate[field] = state.get(field)

        violation = _security_gate_output(candidate)
        if violation:
            logger.error("PostProcessNode: output boundary violation — %s", violation)
            emit_trace_event(
                "post_process_output_violation",
                {"violation": violation},
                state,
            )
            withheld: dict[str, Any] = {
                "formatted_output": _WITHHELD_NOTICE,
                "result": _WITHHELD_NOTICE,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output withheld — credential pattern detected ({violation})"],
            }
            # Clear every field that could still carry the ungated document.
            for field in _OUTPUT_BEARING_FIELDS:
                withheld[field] = None
            return withheld

        logger.info("PostProcessNode: output boundary passed — length=%d", len(answer))
        emit_trace_event(
            "post_process_complete",
            {"output_length": len(answer)},
            state,
        )

        return {
            "formatted_output": answer,
            "result": answer,
            "status": AgentStatus.SUCCESS.value,
        }
