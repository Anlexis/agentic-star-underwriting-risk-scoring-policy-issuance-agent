"""AgentCore Platform v1.0"""

# INS-C2-016 — GenerateAnswerNode
# Inner domain node 4: compose the underwriting answer from the reranked
# evidence ONLY.
#
# This step is DETERMINISTIC RENDERING, and the manifest says so
# (generation_mode: deterministic). No model is called: the node assembles the
# answer from the caller's question and the passages that survived reranking,
# and it abstains when nothing survived. Naming that plainly matters, because
# the two failure modes look identical from outside — an agent that renders
# deterministically and an agent whose generation step is broken both return
# text — and only one of them varies with its input. The unit and end-to-end
# tests pin exactly that: two different questions produce two different answers
# built from the passages each one retrieved.
#
# The declared prompt template (config/config.yaml -> llm.system_prompt_template)
# is resolved here and its resolution is recorded in the audit event, so the
# declared value is observable rather than merely present in a file. It is the
# grounding contract a model would be given if one were wired in: cite only the
# supplied passages, and never invent a risk score or premium. The rendering
# below holds that contract by construction — every citation is an evidence
# doc_id, and no other text is added.
#
# Inner node — ANONYMOUS trust.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from pathlib import Path
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# src/nodes/generate_answer_node.py -> parents[2] is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_INSUFFICIENT = (
    "The knowledge base does not contain sufficient grounded evidence to answer "
    "this underwriting question. Route to a manual underwriter for review; do not "
    "issue a risk score or premium guidance without a supporting guideline."
)


def _confidence(evidence_count: int, top_score: float) -> str:
    """Map evidence coverage to a confidence band (derived from the evidence)."""
    if evidence_count == 0:
        return "low"
    if evidence_count >= 3 and top_score >= 0.9:
        return "high"
    if evidence_count >= 2:
        return "medium"
    return "medium" if top_score >= 0.9 else "low"


def _resolve_template(settings: dict[str, Any]) -> bool:
    """True when the declared prompt template exists on disk.

    A declared path that does not resolve is a configuration defect worth
    surfacing in the audit trail; it does not stop the answer being composed,
    because the grounding contract is enforced by the rendering below rather
    than by the template text.
    """
    declared = settings.get("system_prompt_template")
    if not isinstance(declared, str) or not declared:
        return False
    candidate = (_REPO_ROOT / declared).resolve()
    try:
        candidate.relative_to(_REPO_ROOT)
    except ValueError:
        return False
    return candidate.is_file()


def _compose(query: str, evidence: list[dict[str, Any]]) -> str:
    """Build the answer body citing ONLY the supplied evidence."""
    lines = [f"Question: {query}", "", "Grounded determination:"]
    for item in evidence:
        doc_id = item.get("doc_id", "N/A")
        text = str(item.get("text", "")).strip()
        lines.append(f"  - [{doc_id}] {text}")
    lines.append("")
    lines.append(
        "The above passages are the sole basis for this determination. Any factor "
        "not covered by a cited passage requires actuarial / underwriter judgement."
    )
    return "\n".join(lines)


class GenerateAnswerNode(FunctionNode):
    """Compose the grounded underwriting answer from the reranked evidence.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        retrieval_query:    str  — the underwriting question
        reranked_documents: str  — JSON-serialised grounded evidence
        runtime_settings:   str  — JSON-serialised declared settings; the route
                                   by which system_prompt_template reaches here

    Output state keys (partial dict):
        generated_answer: str  — JSON-serialised answer object
        status:           str
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        query = state.get("retrieval_query") or state.get("validated_input", "") or ""
        evidence: list[dict[str, Any]] = from_json(state.get("reranked_documents"), [])
        settings: dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        template_resolved = _resolve_template(settings)

        citations = [item.get("doc_id", "N/A") for item in evidence]
        top_score = max((float(item.get("score", 0.0)) for item in evidence), default=0.0)

        if not evidence:
            answer_obj: dict[str, Any] = {
                "answer_text": _INSUFFICIENT,
                "citations": [],
                "confidence": "low",
                "grounded": False,
                "evidence_count": 0,
            }
            logger.info("GenerateAnswerNode: no evidence — abstaining")
            emit_trace_event(
                "generate_answer_insufficient",
                {
                    "query_chars": len(query),
                    "evidence_count": 0,
                    "prompt_template_resolved": template_resolved,
                },
                state,
            )
            return {
                "generated_answer": to_json(answer_obj),
                "status": AgentStatus.SUCCESS.value,
            }

        answer_obj = {
            "answer_text": _compose(query, evidence),
            "citations": citations,
            "confidence": _confidence(len(evidence), top_score),
            "grounded": True,
            "evidence_count": len(evidence),
        }

        logger.info(
            "GenerateAnswerNode: answer composed (evidence=%d, confidence=%s)",
            len(evidence),
            answer_obj["confidence"],
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "evidence_count": len(evidence),
                "citation_count": len(citations),
                "confidence": answer_obj["confidence"],
                "prompt_template_resolved": template_resolved,
            },
            state,
        )

        return {
            "generated_answer": to_json(answer_obj),
            "status": AgentStatus.SUCCESS.value,
        }
