"""AgentCore Platform v1.0"""

# INS-C2-016 — MainNode (reference node for the flat single-layer sample)
#
# This template is a two-layer agent: the `main` backbone slot is filled by
# UnderwritingRAGGraphNode (src/graph/graph.py), which delegates the domain
# workflow to DomainWorkflowGraph. It is NOT filled by this node.
#
# MainNode is kept because src/examples/graph_cat1_sample.py — the flat,
# single-layer composition shown alongside the nested one — needs a concrete
# main-slot node to be a runnable illustration. It is the minimum a main-slot
# FunctionNode must do: declare its trust level, emit a domain audit event, and
# return a partial state dict. Nothing in the underwriting pipeline imports it.

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

_SAMPLE_RESULT = (
    "MainNode is the flat single-layer reference node. This agent's main slot "
    "is UnderwritingRAGGraphNode, which delegates to DomainWorkflowGraph."
)


class MainNode(FunctionNode):
    """Reference main-slot node for the flat composition sample."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # Every boundary node emits at least one domain audit event.
        emit_trace_event(
            "main_node_reference_invoked",
            {"note": "flat-composition reference node", "pipeline": "not the underwriting path"},
            state,
        )
        return {
            "status": AgentStatus.SUCCESS.value,
            "result": _SAMPLE_RESULT,
        }
