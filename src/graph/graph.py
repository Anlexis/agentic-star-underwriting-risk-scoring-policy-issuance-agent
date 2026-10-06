"""AgentCore Platform v1.0"""

# INS-C2-016 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process
#              -> finalize -> END
#                                        | (retry, budget from config/config.yaml)
#                                        v
#                                    pre_process
#
#   The `main` slot is a GraphNode subclass (UnderwritingRAGGraphNode) that
#   delegates the full domain workflow to DomainWorkflowGraph (inner BaseGraph).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step retrieval topology)
#   src/graph/context_bridge.py        <- outer -> inner input_context hand-off

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.trust_level import TrustLevel
from framework.schemas.agent_state import AgentState
from framework.security.credential_detector import detect_credentials

from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State
from src.services.service import finite_in_range


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


if TYPE_CHECKING:
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Runtime-parameter file: src/graph/graph.py -> parents[2] is the repository
# root. config/agent.yaml holds only the registration identity (a flat manifest
# with no runtime block); every runtime parameter lives in config/config.yaml,
# which is the file the platform registry loads and passes as Graph(config=...).
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Declared-value bounds. A configuration file is not caller data, but it is
# still an input: an out-of-range or non-finite value here would silently
# disable retrieval or the relevance filter, so each one is parsed and bounded
# exactly like a caller field and an invalid entry falls back to the node
# default instead of propagating.
_TOP_K_BOUNDS = (1.0, 50.0)
_SCORE_THRESHOLD_BOUNDS = (0.0, 1.0)

# Domain patterns for the agent-level output redaction, applied ON TOP OF the
# framework's own credential detector — never instead of it. A `password = ...`
# assignment is not a credential SHAPE the detector recognises, but it is
# exactly what a mis-pasted operations note looks like.
_EXTRA_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
        re.IGNORECASE,
    ),
)

_REDACTED = "[REDACTED]"


def runtime_config() -> dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    The standalone server (src/api/server.py) calls this so a directly deployed
    agent and a registry-loaded agent see identical configuration. Returns an
    empty dict — never raises — when the file is absent, unreadable, not valid
    YAML, or not a mapping; the graph then runs on its built-in defaults.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return cast("dict[str, Any]", loaded)


class UnderwritingRAGGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the agent.

    Wraps DomainWorkflowGraph (the inner retrieval workflow).
    Called by the AgentBaseGraph backbone after pre_process and before
    post_process.

    Contracts:
      get_subgraph()  — instantiate and return DomainWorkflowGraph
      extract_input() — pull validated_input from outer state; bridge input_context
      merge_output()  — map sub_result fields into the outer state delta (changed keys only)
      error_strategy  — "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    # S-1 declared on the wrapper too: the CI gate only AST-scans FunctionNode
    # subclasses, so a GraphNode main slot passes the pipeline without one and is
    # flagged at review. Same level the nodes in this repo already declare.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time and to match the nested
        composition pattern.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates and normalises the raw user_input and writes
        the result to validated_input. Prefer that; fall back to user_input if
        validated_input is absent (e.g. in a unit test that drives this node
        directly).

        Also bridges the caller's input_context to the inner graph: the
        framework's GraphNode.execute() does not forward input_context on
        subgraph.invoke(), and this is the last hook in template code that sees
        the outer state before the inner invoke — see src/graph/context_bridge.py.
        """
        set_caller_input_context(
            cast("Optional[dict[str, Any]]", _without_platform_context(state.get("input_context")))
        )
        return cast(str, state.get("validated_input", state.get("user_input", "")))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map the inner graph's sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "rag_answer", "generated_answer",
                                       "reranked_documents", "retrieved_count",
                                       "result", "status"
          This merge_output() reads -> sub_result.get(...) for each of those keys.

        reranked_documents is forwarded because PostProcessNode screens it as
        part of the output-bearing set: the rendered answer is the
        caller-visible representation, but the same material also rides in the
        structured evidence list, and a boundary that screened only the rendered
        text would leave the structured copy in state.
        """
        return {
            "rag_answer": sub_result.get("rag_answer"),
            "generated_answer": sub_result.get("generated_answer"),
            "reranked_documents": sub_result.get("reranked_documents"),
            "retrieved_count": sub_result.get("retrieved_count", 0),
            "result": sub_result.get("result"),
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> dict[str, Any]:
        """Forward the declared retrieval settings to the inner graph.

        Reads config/config.yaml (see runtime_config) and returns a flat
        settings mapping. DomainWorkflowGraph seeds it into the inner graph's
        initial state, where the domain nodes read it — the node contract is
        ``execute(self, state) -> dict``, so a node cannot receive a
        per-invocation config argument and state is the only live route.

        Every value is validated here (type, finiteness, range). A key that is
        absent or invalid is simply not forwarded, and the node falls back to
        its own module default.
        """
        cfg = runtime_config()
        retrieval_raw = cfg.get("retrieval")
        retrieval: dict[str, Any] = retrieval_raw if isinstance(retrieval_raw, dict) else {}
        llm_raw = cfg.get("llm")
        llm: dict[str, Any] = llm_raw if isinstance(llm_raw, dict) else {}

        declared: dict[str, Any] = {}

        top_k = finite_in_range(retrieval.get("top_k"), *_TOP_K_BOUNDS)
        if top_k is not None:
            declared["top_k"] = int(top_k)

        score_threshold = finite_in_range(retrieval.get("score_threshold"), *_SCORE_THRESHOLD_BOUNDS)
        if score_threshold is not None:
            declared["score_threshold"] = score_threshold

        hybrid_search = retrieval.get("hybrid_search")
        if isinstance(hybrid_search, bool):
            declared["hybrid_search"] = hybrid_search

        template = llm.get("system_prompt_template")
        if isinstance(template, str) and template:
            declared["system_prompt_template"] = template

        return declared


class InsuranceUnderwritingRiskScoringAgent(AgentBaseGraph):
    """Outer graph for INS-C2-016 — grounded Q&A over the underwriting knowledge base.

    Inherits AgentBaseGraph directly. Domain logic is fully encapsulated in
    UnderwritingRAGGraphNode (main slot), which delegates to DomainWorkflowGraph.

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode           (VERIFIED_EXTERNAL — trust boundary)
      - main:         UnderwritingRAGGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode          (ANONYMOUS — output boundary)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.

    The class name must match the `class:` entry point in config/agent.yaml;
    src/api/server.py imports it as `Graph`.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the platform registry."""
        return "InsuranceUnderwritingRiskScoringAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (schema_version, session_id,
        trust_level) and FinalizeNode (response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = UnderwritingRAGGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.

    def _security_gate_output(self, output: Optional[str]) -> Optional[str]:
        """Redact credential material from a rendered output string.

        This is the agent-class output filter. Runtime enforcement for the
        deployed pipeline lives in PostProcessNode, which withholds rather than
        redacts; this method exists so a caller that renders an answer through
        the agent object directly still gets credential material removed.

        The framework's own detector supplies the pattern set, with the domain
        assignment pattern layered on top. Keeping a narrower local set here
        would mean a value the framework recognises passes this filter, which
        is a bypass rather than a smaller net.
        """
        if not output:
            return output
        redacted = output
        for finding in reversed(detect_credentials(redacted)):
            start, end = int(finding["start"]), int(finding["end"])
            redacted = redacted[:start] + _REDACTED + redacted[end:]
        for pattern in _EXTRA_REDACTION_PATTERNS:
            redacted = pattern.sub(_REDACTED, redacted)
        return redacted
