"""AgentCore Platform v1.0"""

# INS-C2-016 — PreProcessNode
# Outer backbone pre_process slot: trust enforcement + the caller-data contract.
#
# Responsibilities:
#   - Enforce VERIFIED_EXTERNAL trust (required_trust_level) — this node is the
#     external trust boundary for the whole agent; the inner domain nodes run
#     ANONYMOUS because trust is enforced once, here
#   - Reject empty or oversized input early (fail-fast)
#   - Refuse instruction-override content in the node that owns the caller
#     contract, so refusal does not depend on any gate in front of it
#   - Validate the structured input_context channel field by field, refusing
#     unknown keys rather than ignoring them
#   - Write validated_input + enriched_context to State
#   - Emit an audit event for every validation decision
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import is_inert_token, safe_field_name, screen_structure, screen_text


# The Marketplace runner seeds input_context with its own conversation history on every
# invocation (shared/bootstrap/marketplace_app.py); the caller neither sends that key nor can
# suppress it, and the build_input_context hook can only overwrite its value, never remove it.
# It is platform plumbing rather than caller data, so it is dropped here, before the caller
# contract runs: the unknown-field guard below stays strict for everything a caller can
# actually send, and no value screen is ever asked to judge a transcript that contains this
# agent's own earlier answers. The value may also be None, which this tolerates.
_PLATFORM_CONTEXT_KEYS = frozenset({"conversation_history"})


def _without_platform_context(raw: Any) -> Any:
    """The caller-supplied half of input_context, platform-injected keys removed."""
    if not isinstance(raw, dict):
        return raw
    return {k: v for k, v in raw.items() if k not in _PLATFORM_CONTEXT_KEYS}


logger = logging.getLogger(__name__)

# Structural cap on the free-text question. An oversized question is both a
# cost surface and an injection surface; it is refused rather than truncated,
# because silently answering a different question than the one asked is worse
# than saying no.
_MAX_QUERY_CHARS = 4000

# The structured invocation channel. Values are inert lowercase tokens only:
# the channel label is recorded in the request metadata and audit events, so
# free text there would be caller-controlled content on those paths. Unknown
# keys are refused, not silently ignored — an ignored key is still caller data
# travelling into the pipeline.
_ALLOWED_CONTEXT_KEYS = frozenset({"channel"})

_DEFAULT_CHANNEL = "unknown"


# Reason codes whose message names the screen that fired rather than a field the caller
# can correct. These refusals publish nothing; every other reason publishes its reason.
_SILENT_REASONS = frozenset({"instruction_override"})


class PreProcessNode(FunctionNode):
    """Caller-data contract and trust boundary for INS-C2-016.

    Input state keys:
        user_input:    str  — the caller's underwriting question (free text)
        input_context: dict — structured invocation parameters (inert tokens)

    Output state keys (partial dict):
        validated_input:  str        — normalised question
        enriched_context: str        — JSON-serialised request metadata
        status:           str        — AgentStatus.SUCCESS or ERROR
        error_log:        list[str]  — set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        raw_context = _without_platform_context(state.get("input_context")) or {}

        if not isinstance(user_input, str) or not user_input.strip():
            logger.warning("PreProcessNode: user_input is empty or missing")
            return self._reject("empty_input", "user_input is empty or missing", state)

        if len(user_input) > _MAX_QUERY_CHARS:
            return self._reject(
                "query_too_long",
                f"user_input exceeds the {_MAX_QUERY_CHARS}-character limit",
                state,
            )

        # Instruction-override screen on the question. The framework applies its
        # own input policy in front of this node, but it treats some
        # control-token forms as audit-only, so a template that relied on it
        # alone would forward those payloads into the answer document.
        finding = screen_text(user_input)
        if finding:
            return self._reject(
                "instruction_override",
                "user_input rejected: disallowed instruction content",
                state,
                detail=finding,
            )

        context_error = self._validate_context(raw_context)
        if context_error is not None:
            return self._reject("invalid_input_context", context_error, state)

        normalised = " ".join(user_input.split())
        channel = str(raw_context.get("channel", "")) or _DEFAULT_CHANNEL

        logger.info("PreProcessNode: question normalised (chars=%d)", len(normalised))
        emit_trace_event(
            "pre_process_complete",
            {"query_chars": len(normalised), "channel": channel},
            state,
        )

        return {
            "validated_input": normalised,
            "enriched_context": to_json(
                {
                    "source": "InsuranceUnderwritingRiskScoringAgent",
                    "channel": channel,
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_context(raw_context: Any) -> Optional[str]:
        """Validate the structured invocation channel; return an error or None."""
        if not isinstance(raw_context, dict):
            return "input_context must be an object"
        unknown = [
            safe_field_name(key, position)
            for position, key in enumerate(raw_context, start=1)
            if key not in _ALLOWED_CONTEXT_KEYS
        ]
        if unknown:
            return f"unsupported input_context fields: {sorted(unknown)}"
        finding = screen_structure(raw_context)
        if finding:
            return "input_context rejected: disallowed instruction content"
        if "channel" in raw_context and not is_inert_token(raw_context["channel"]):
            return "input_context.channel must be 1-32 characters of [a-z0-9_]"
        return None

    @staticmethod
    def _reject(reason: str, message: str, state: AgentState, detail: Any = None) -> dict[str, Any]:
        """Fail closed with an audit event.

        The message names the field and the reason, never the value: a rejected
        payload is exactly the payload that must not be echoed back into a log
        line or an error response.
        """
        event: dict[str, Any] = {"reason": reason}
        if detail is not None:
            event["detail"] = detail
        emit_trace_event("pre_process_validation_failed", event, state)
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {message}"],
            # The runner surfaces `formatted_output or result` as `output`. A reason left only in
            # error_log reaches no one: the terminal result carries just `status`, and get_output()
            # does not copy error_log out of the graph -- the caller sees a blank spinner.
            # A SCREENED refusal stays silent. Its message names the detector that caught the
            # payload, and handing that back lets an attacker probe the screen one try at a
            # time. A VALIDATION refusal says what to fix -- without it the caller cannot tell
            # a rejected request from a hung one.
            **(
                {}
                if reason in _SILENT_REASONS
                else {"formatted_output": "Request could not be completed. " + f"PreProcessNode: {message}"}
            ),
        }
