"""AgentCore Platform v1.0"""

# INS-C2-016 — RerankFilterNode
# Inner domain node 3: rerank the retrieved candidates and filter them to the
# grounded evidence set by the declared score_threshold.
#
# Reranking is relative: the raw retrieval scores are normalised against the
# top candidate (best = 1.0), then filtered so only passages within
# score_threshold of the top match survive. Swapping in a cross-encoder
# reranker does not change the contract (retrieved_documents in ->
# reranked_documents out).
#
# score_threshold is read from runtime_settings — the declared value from
# config/config.yaml, validated by the outer graph and seeded into this graph's
# initial state, because execute(state) takes no config argument.
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

from src.schemas.state import from_json, to_json
from src.services.service import finite_in_range

logger = logging.getLogger(__name__)

# Fallback used when score_threshold is absent from runtime_settings — not
# declared in config/config.yaml, or declared with a value that failed
# validation. It mirrors the shipped configuration.
_DEFAULT_SCORE_THRESHOLD = 0.75
_SCORE_THRESHOLD_BOUNDS = (0.0, 1.0)


class RerankFilterNode(FunctionNode):
    """Rerank retrieved candidates and filter by relative score_threshold.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        retrieved_documents: str  — JSON-serialised list[doc] from RetrieveNode
        runtime_settings:    str  — JSON-serialised declared settings; the route
                                    by which score_threshold reaches this node

    Output state keys (partial dict):
        reranked_documents: str  — JSON-serialised grounded evidence list
        retrieved_count:    int  — number of grounded passages kept
        status:             str
        error_log:          list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = from_json(state.get("retrieved_documents"), [])
        settings: dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}

        declared = finite_in_range(settings.get("score_threshold"), *_SCORE_THRESHOLD_BOUNDS)
        threshold = declared if declared is not None else _DEFAULT_SCORE_THRESHOLD

        if not candidates:
            # Not an error: an empty candidate set means "no grounded evidence".
            emit_trace_event(
                "rerank_filter_empty",
                {"reason": "no_candidates", "threshold": threshold},
                state,
            )
            return {
                "reranked_documents": to_json([]),
                "retrieved_count": 0,
                "status": AgentStatus.SUCCESS.value,
            }

        top_raw = max((float(c.get("score", 0.0)) for c in candidates), default=0.0)

        evidence: list[dict[str, Any]] = []
        for c in candidates:
            raw = float(c.get("score", 0.0))
            rerank_score = round(raw / top_raw, 4) if top_raw > 0 else 0.0
            if rerank_score >= threshold:
                item = dict(c)  # local copy — do not mutate the input list items
                item["score"] = rerank_score
                evidence.append(item)

        evidence.sort(key=lambda d: d["score"], reverse=True)

        logger.info(
            "RerankFilterNode: candidates=%d kept=%d (threshold=%.2f)",
            len(candidates),
            len(evidence),
            threshold,
        )
        emit_trace_event(
            "rerank_filter_complete",
            {"kept": len(evidence), "threshold": threshold, "candidates": len(candidates)},
            state,
        )

        return {
            "reranked_documents": to_json(evidence),
            "retrieved_count": len(evidence),
            "status": AgentStatus.SUCCESS.value,
        }
