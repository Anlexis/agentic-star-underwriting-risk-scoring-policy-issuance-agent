"""AgentCore Platform v1.0"""

# INS-C2-016 — DomainWorkflowGraph (inner BaseGraph)
#
# The inner graph of the two-layer nested composition. It encapsulates the
# underwriting retrieval pipeline:
#
#   START
#     -> input_validate   (InputValidateNode)
#     -> retrieve         (RetrieveNode)
#     -> rerank_filter    (RerankFilterNode)
#     -> generate_answer  (GenerateAnswerNode)
#     -> output_format    (OutputFormatNode)
#     -> END
#
# Called by UnderwritingRAGGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Inherits BaseGraph for a fully custom topology; register_nodes() does not call
# super() (it is abstract on BaseGraph) and does not register initialize /
# finalize, which are outer-backbone concerns. Every inner node declares
# TrustLevel.ANONYMOUS and takes no constructor arguments.

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for INS-C2-016 (underwriting retrieval).

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)
          -> retrieve        (RetrieveNode)
          -> rerank_filter   (RerankFilterNode)
          -> generate_answer (GenerateAnswerNode)
          -> output_format   (OutputFormatNode)
          -> END
    """

    # -- Identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ins_c2_016_underwriting_retrieval_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # -- Config validation ----------------------------------------------------

    def _validate_config(self) -> None:
        """No key is mandatory: every declared setting was already validated.

        UnderwritingRAGGraphNode._parent_config() parses each declared value for
        type, finiteness and range and forwards only the ones that pass, so an
        absent key here means "not declared, or declared invalid" and the
        consuming node keeps its own documented default.
        """
        return None

    # -- Initial state --------------------------------------------------------

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the inner graph's initial state.

        Two things cross the outer -> inner boundary here, and neither can
        travel any other way:

        - ``input_context`` — the framework's GraphNode.execute() invokes this
          subgraph without forwarding it, so it is stashed by the outer node's
          extract_input() and picked up here (src/graph/context_bridge.py).
        - ``runtime_settings`` — the declared, already-validated retrieval
          settings. Node constructors take no arguments and ``execute(state)``
          takes no config argument, so state is the route by which a value in
          config/config.yaml reaches a domain node.
        """
        return {
            "input_context": get_caller_input_context(),
            "runtime_settings": to_json(dict(self.config)),
        }

    # -- Node registration ----------------------------------------------------

    def register_nodes(self) -> None:
        """Register the 5 domain nodes. Every key registered is used in add_edges()."""
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring ----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear retrieval topology.

        input_validate -> retrieve -> rerank_filter -> generate_answer
        -> output_format -> END. There is no conditional branching, so route()
        satisfies the abstract contract but is not wired into any edge.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing --------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by the BaseGraph contract.

        The topology is linear and add_conditional_edges() is not used, so this
        method is never called at runtime. It returns END on an error status so
        an unexpected invocation cannot re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ---------------------------------------------------------

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        Received by UnderwritingRAGGraphNode.merge_output() in graph.py. Both
        methods are designed together so the field names cannot drift:

            Inner get_output() emits:   "rag_answer", "generated_answer",
                                        "reranked_documents", "retrieved_count",
                                        "result", "status"
            Outer merge_output() reads: sub_result.get(...) for each key above.
        """
        return {
            "rag_answer": state.get("rag_answer"),
            "generated_answer": state.get("generated_answer"),
            "reranked_documents": state.get("reranked_documents"),
            "retrieved_count": state.get("retrieved_count", 0),
            "result": state.get("result"),
            "status": state.get("status"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
