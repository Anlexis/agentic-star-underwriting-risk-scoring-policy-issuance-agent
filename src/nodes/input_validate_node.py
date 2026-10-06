"""AgentCore Platform v1.0"""

# INS-C2-016 — InputValidateNode
# Inner domain node 1: domain-level validation of the underwriting question and
# construction of the normalised retrieval query.
#
# Distinct from PreProcessNode (trust boundary + caller-data contract): this
# node applies the domain rule that a retrieval query must carry enough
# substance to match a guideline passage at all.
#
# Inner node — ANONYMOUS trust. The outer PreProcessNode already enforced
# VERIFIED_EXTERNAL, and the inner nodes must be ANONYMOUS so the caller's
# invocation context passes through the subgraph boundary without rejection.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# A query must carry at least this many word characters of substance to be a
# meaningful underwriting question (guards near-empty / punctuation-only input).
_MIN_QUERY_WORDS = 2
_WORD_RE = re.compile(r"\w+", re.UNICODE)


class InputValidateNode(FunctionNode):
    """Domain validation of the underwriting question + retrieval-query build.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        validated_input: str  — normalised question from PreProcessNode.
                                Falls back to user_input, which is what the
                                inner graph is invoked with.

    Output state keys (partial dict):
        retrieval_query: str
        status:          str
        error_log:       list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "") or ""
        query = " ".join(str(raw).split())

        if not query:
            emit_trace_event(
                "input_validate_failed",
                {"reason": "empty_query"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: query is empty"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. " + ("InputValidateNode: query is empty"),
            }

        if len(_WORD_RE.findall(query)) < _MIN_QUERY_WORDS:
            emit_trace_event(
                "input_validate_failed",
                {"reason": "query_too_short", "query_len": len(query)},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: query lacks enough substance for retrieval"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("InputValidateNode: query lacks enough substance for retrieval"),
            }

        logger.info("InputValidateNode: retrieval_query built (chars=%d)", len(query))
        emit_trace_event(
            "input_validate_complete",
            {"query_chars": len(query), "word_count": len(_WORD_RE.findall(query))},
            state,
        )

        return {
            "retrieval_query": query,
            "status": AgentStatus.SUCCESS.value,
        }
