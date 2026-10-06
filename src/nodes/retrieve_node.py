"""AgentCore Platform v1.0"""

# INS-C2-016 — RetrieveNode
# Inner domain node 2: retrieve candidate passages from the underwriting
# knowledge base for the query.
#
# Retrieval is deterministic keyword/title overlap over the in-repo underwriting
# knowledge base below. Replacing it with a vector or hybrid index does not
# change the node contract (retrieval_query in -> retrieved_documents out) or
# the settings it reads.
#
# top_k and hybrid_search are read from runtime_settings — the declared values
# from config/config.yaml, validated by the outer graph and seeded into this
# graph's initial state. A node cannot receive a per-invocation config
# argument: the framework calls execute(state) with one argument, so state is
# the only route a declared value can travel.
#
# Inner node — ANONYMOUS trust.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.service import finite_in_range

logger = logging.getLogger(__name__)

# Fallbacks used when the corresponding key is absent from runtime_settings —
# i.e. not declared in config/config.yaml, or declared with a value that failed
# validation. They mirror the shipped configuration.
_DEFAULT_TOP_K = 8
_DEFAULT_HYBRID_SEARCH = True
_TOP_K_BOUNDS = (1.0, 50.0)

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# The in-repo underwriting knowledge base (read-only). Each entry:
# {doc_id, title, source, keywords, text}.
# Treated as immutable — never mutated inside execute(), because a mutation
# would leak across invocations in a long-lived process. Build local copies.
_UNDERWRITING_KB: list[dict[str, Any]] = [
    {
        "doc_id": "UW-GL-001",
        "title": "Life Underwriting — BMI and Build Chart",
        "source": "underwriting_guidelines",
        "keywords": ["bmi", "build", "weight", "height", "obesity", "life", "rating"],
        "text": "Standard life build limits: BMI 18.5-30.0 rates standard. BMI "
        "30.1-35.0 adds +50% mortality debit. BMI above 35 requires "
        "medical underwriter review.",
    },
    {
        "doc_id": "UW-GL-014",
        "title": "Blood Pressure Rating Guidelines",
        "source": "underwriting_guidelines",
        "keywords": ["blood", "pressure", "hypertension", "systolic", "diastolic", "cardiac"],
        "text": "Treated hypertension controlled below 140/90 rates standard. "
        "Readings 140/90-160/100 add a table-2 debit. Uncontrolled "
        "readings require an attending physician statement.",
    },
    {
        "doc_id": "UW-ACT-101",
        "title": "Actuarial Mortality Table — Smoker Differential",
        "source": "actuarial_tables",
        "keywords": ["smoker", "tobacco", "nicotine", "mortality", "premium", "differential"],
        "text": "Smoker mortality is priced at 1.8x the non-smoker base rate. A "
        "12-month cessation with a negative cotinine test reclassifies to "
        "non-smoker at the next anniversary.",
    },
    {
        "doc_id": "UW-PRD-207",
        "title": "SME Group Insurance — Eligibility & Participation",
        "source": "product_terms",
        "keywords": ["sme", "group", "participation", "eligibility", "employees", "census"],
        "text": "Group life for SMEs requires >=75% eligible-employee participation "
        "for non-contributory plans and >=50% for contributory plans. "
        "Minimum group size is 10 lives.",
    },
    {
        "doc_id": "UW-REG-042",
        "title": "Disclosure Duty and Anti-Selection Controls",
        "source": "regulatory_notices",
        "keywords": ["disclosure", "duty", "anti", "selection", "misrepresentation", "consent"],
        "text": "Applicants owe a duty of disclosure of material facts. Non-disclosure "
        "discovered within the contestability period permits rescission per "
        "the Insurance Business Act.",
    },
    {
        "doc_id": "UW-GL-058",
        "title": "Occupational Risk Classes",
        "source": "underwriting_guidelines",
        "keywords": ["occupation", "hazard", "class", "manual", "risk", "aviation"],
        "text": "Occupation class 1-2 rates standard. Hazardous occupations "
        "(aviation crew, offshore, mining) map to class 4-5 with a flat "
        "extra premium per 1,000 sum assured.",
    },
]


def _score(query_terms: set[str], doc: dict[str, Any], hybrid: bool) -> float:
    """Deterministic relevance score in [0, 1] from keyword + title overlap."""
    kw = {str(k).lower() for k in doc.get("keywords", [])}
    title_terms = {t.lower() for t in _WORD_RE.findall(doc.get("title", ""))}
    haystack = kw | (title_terms if hybrid else set())
    if not haystack or not query_terms:
        return 0.0
    overlap = len(query_terms & haystack)
    return round(overlap / max(len(query_terms), 1), 4)


class RetrieveNode(FunctionNode):
    """Retrieve candidate underwriting documents for the query.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        retrieval_query:  str  — from InputValidateNode
        runtime_settings: str  — JSON-serialised declared settings; the route by
                                 which top_k / hybrid_search reach this node

    Output state keys (partial dict):
        retrieved_documents: str  — JSON-serialised list[doc]
        retrieved_count:     int
        status:              str
        error_log:           list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        query = state.get("retrieval_query") or state.get("validated_input", "") or ""
        settings: dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}

        declared_top_k = finite_in_range(settings.get("top_k"), *_TOP_K_BOUNDS)
        top_k = int(declared_top_k) if declared_top_k is not None else _DEFAULT_TOP_K
        declared_hybrid = settings.get("hybrid_search")
        hybrid = declared_hybrid if isinstance(declared_hybrid, bool) else _DEFAULT_HYBRID_SEARCH

        if not query.strip():
            emit_trace_event(
                "retrieve_failed",
                {"reason": "empty_retrieval_query"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: retrieval_query missing"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. " + ("RetrieveNode: retrieval_query missing"),
            }

        query_terms = {t.lower() for t in _WORD_RE.findall(query)}

        # Build a NEW list of scored candidates — never mutate _UNDERWRITING_KB.
        scored: list[dict[str, Any]] = []
        for doc in _UNDERWRITING_KB:
            s = _score(query_terms, doc, hybrid)
            if s <= 0.0:
                continue
            scored.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "text": doc["text"],
                    "score": s,
                }
            )

        scored.sort(key=lambda d: d["score"], reverse=True)
        candidates = scored[:top_k]

        logger.info(
            "RetrieveNode: query_terms=%d candidates=%d (top_k=%d, hybrid=%s)",
            len(query_terms),
            len(candidates),
            top_k,
            hybrid,
        )
        emit_trace_event(
            "retrieve_complete",
            {"candidate_count": len(candidates), "top_k": top_k, "hybrid_search": hybrid},
            state,
        )

        return {
            "retrieved_documents": to_json(candidates),
            "retrieved_count": len(candidates),
            "status": AgentStatus.SUCCESS.value,
        }
