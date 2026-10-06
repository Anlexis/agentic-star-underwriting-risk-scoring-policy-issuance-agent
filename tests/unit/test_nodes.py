# INS-C2-016 — Unit Tests: domain nodes + graph wiring
#
# Real, non-stub unit tests against the shipped modules: knowledge-base
# retrieval, rerank/filter, answer composition, insufficient-evidence
# ABSTENTION, the declared trust levels, the output boundary, and the two-layer
# graph composition.
#
# Audit events are patched at the node MODULE level (not via a sys.modules
# stub, which would break the real `shared` package the framework loads at
# import time). Patch pattern per node:
#     monkeypatch.setattr("src.nodes.<mod>.emit_trace_event", lambda *a, **k: None)

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.schemas.state import from_json, to_json


# ── Shared fixtures / helpers ─────────────────────────────────────────────────

# A grounded-yielding underwriting question (terms overlap the in-repo KB).
_VALID_QUERY = (
    "What are the BMI build chart limits and treated hypertension blood pressure "
    "rating guidelines for a smoker life insurance applicant?"
)


def _candidates() -> list:
    """A retrieved-candidate list (RetrieveNode output shape) with known raw scores.

    Relative rerank against top=0.30 → 1.0 / 0.833 / 0.167; with the default
    score_threshold=0.75 only the first two survive."""
    return [
        {
            "doc_id": "UW-GL-001",
            "title": "BMI and Build Chart",
            "source": "underwriting_guidelines",
            "text": "Standard build limits by BMI band.",
            "score": 0.30,
        },
        {
            "doc_id": "UW-GL-014",
            "title": "Blood Pressure Rating",
            "source": "underwriting_guidelines",
            "text": "Treated hypertension below 140/90 rates standard.",
            "score": 0.25,
        },
        {
            "doc_id": "UW-ACT-101",
            "title": "Smoker Differential",
            "source": "actuarial_tables",
            "text": "Smoker mortality priced at 1.8x base.",
            "score": 0.05,
        },
    ]


def _evidence() -> list:
    """A reranked/grounded evidence list (RerankFilterNode output shape)."""
    return [
        {
            "doc_id": "UW-GL-001",
            "title": "BMI and Build Chart",
            "source": "underwriting_guidelines",
            "text": "Standard build limits by BMI band.",
            "score": 1.0,
        },
        {
            "doc_id": "UW-GL-014",
            "title": "Blood Pressure Rating",
            "source": "underwriting_guidelines",
            "text": "Treated hypertension below 140/90 rates standard.",
            "score": 0.83,
        },
    ]


def _answer_obj(grounded: bool = True) -> dict:
    """A generated-answer object (GenerateAnswerNode output shape)."""
    if grounded:
        return {
            "answer_text": "Grounded determination:\n  - [UW-GL-001] BMI band.\n  - [UW-GL-014] BP.",
            "citations": ["UW-GL-001", "UW-GL-014"],
            "confidence": "medium",
            "grounded": True,
            "evidence_count": 2,
        }
    return {
        "answer_text": "The knowledge base does not contain sufficient grounded evidence. "
        "Route to a manual underwriter for review.",
        "citations": [],
        "confidence": "low",
        "grounded": False,
        "evidence_count": 0,
    }


# ── PreProcessNode (outer pre_process, VERIFIED_EXTERNAL trust boundary) ──────


class TestPreProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_valid_query_returns_success(self):
        result = self.node.execute({"user_input": _VALID_QUERY, "input_context": {}})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None
        assert "smoker life insurance" in result["validated_input"]

    def test_whitespace_is_normalised(self):
        result = self.node.execute({"user_input": "  life   insurance\n\tunderwriting  ", "input_context": {}})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == "life insurance underwriting"

    def test_enriched_context_carries_channel_and_source(self):
        result = self.node.execute({"user_input": _VALID_QUERY, "input_context": {"channel": "broker_portal"}})
        ctx = from_json(result["enriched_context"])
        assert ctx["channel"] == "broker_portal"
        assert ctx["source"] == "InsuranceUnderwritingRiskScoringAgent"

    def test_empty_input_returns_error(self):
        result = self.node.execute({"user_input": "", "input_context": {}})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("empty" in e for e in result["error_log"])

    def test_trust_level_is_verified_external(self):
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_execute_takes_state_only(self):
        """The node contract is execute(self, state) — see TestNodeExecuteSignatures."""
        import inspect

        from src.nodes.pre_process_node import PreProcessNode

        assert list(inspect.signature(PreProcessNode.execute).parameters) == ["self", "state"]


# ── InputValidateNode (inner domain node 1, ANONYMOUS) ─────────────────────────


class TestInputValidateNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.input_validate_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.input_validate_node import InputValidateNode

        self.node = InputValidateNode()

    def test_valid_input_builds_retrieval_query(self):
        result = self.node.execute({"validated_input": _VALID_QUERY})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieval_query"] == _VALID_QUERY

    def test_collapses_whitespace(self):
        result = self.node.execute({"validated_input": "life   insurance   underwriting"})
        assert result["retrieval_query"] == "life insurance underwriting"

    def test_falls_back_to_user_input(self):
        result = self.node.execute({"user_input": _VALID_QUERY})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieval_query"] == _VALID_QUERY

    def test_empty_query_returns_error(self):
        result = self.node.execute({"validated_input": ""})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("empty" in e for e in result["error_log"])

    def test_too_short_query_returns_error(self):
        result = self.node.execute({"validated_input": "bmi"})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("substance" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RetrieveNode (inner domain node 2, ANONYMOUS) ──────────────────────────────


class TestRetrieveNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.retrieve_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.retrieve_node import RetrieveNode

        self.node = RetrieveNode()

    def test_retrieves_grounded_candidates(self):
        result = self.node.execute({"retrieval_query": _VALID_QUERY})
        assert result["status"] == AgentStatus.SUCCESS.value
        docs = from_json(result["retrieved_documents"])
        assert result["retrieved_count"] == len(docs)
        assert len(docs) >= 2
        # scores are sorted descending and every candidate has a positive score
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(d["score"] > 0.0 for d in docs)
        assert any(d["doc_id"] == "UW-GL-001" for d in docs)

    def test_no_keyword_overlap_returns_empty(self):
        result = self.node.execute({"retrieval_query": "zzz qqq foobar nomatch"})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieved_count"] == 0
        assert from_json(result["retrieved_documents"]) == []

    def test_empty_query_returns_error(self):
        result = self.node.execute({"retrieval_query": ""})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("retrieval_query" in e for e in result["error_log"])

    def test_declared_top_k_caps_results(self):
        """A declared top_k arrives in runtime_settings and bounds the result set."""
        result = self.node.execute({"retrieval_query": _VALID_QUERY, "runtime_settings": to_json({"top_k": 1})})
        docs = from_json(result["retrieved_documents"])
        assert len(docs) == 1

    @pytest.mark.parametrize(
        "bad",
        [float("nan"), float("inf"), float("-inf"), 0, -3, 999, "8", True, None, [8]],
        ids=["nan", "inf", "-inf", "zero", "negative", "over-range", "string", "bool", "none", "list"],
    )
    def test_non_finite_or_out_of_range_top_k_falls_back(self, bad):
        """NaN parses through float() and compares False against every bound, so an
        unchecked value would silently disable the cap. It must fall back instead."""
        baseline = from_json(self.node.execute({"retrieval_query": _VALID_QUERY})["retrieved_documents"])
        result = self.node.execute({"retrieval_query": _VALID_QUERY, "runtime_settings": to_json({"top_k": bad})})
        assert from_json(result["retrieved_documents"]) == baseline, f"top_k={bad!r} was not rejected"

    def test_declared_hybrid_search_changes_scoring(self):
        """hybrid_search widens the match surface to titles as well as keywords."""
        query = "blood pressure rating"
        on = self.node.execute({"retrieval_query": query, "runtime_settings": to_json({"hybrid_search": True})})
        off = self.node.execute({"retrieval_query": query, "runtime_settings": to_json({"hybrid_search": False})})
        assert from_json(on["retrieved_documents"]) != from_json(off["retrieved_documents"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RerankFilterNode (inner domain node 3, ANONYMOUS) ──────────────────────────


class TestRerankFilterNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.rerank_filter_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.rerank_filter_node import RerankFilterNode

        self.node = RerankFilterNode()

    def test_filters_by_default_threshold(self):
        result = self.node.execute({"retrieved_documents": to_json(_candidates())})
        assert result["status"] == AgentStatus.SUCCESS.value
        evidence = from_json(result["reranked_documents"])
        # top=0.30 → 1.0/0.833/0.167; threshold 0.75 keeps the first two only
        assert result["retrieved_count"] == 2
        kept_ids = {e["doc_id"] for e in evidence}
        assert kept_ids == {"UW-GL-001", "UW-GL-014"}
        assert evidence[0]["score"] == 1.0  # normalised to the top candidate

    def test_stricter_declared_threshold(self):
        """A declared score_threshold arrives in runtime_settings and tightens the filter."""
        result = self.node.execute(
            {
                "retrieved_documents": to_json(_candidates()),
                "runtime_settings": to_json({"score_threshold": 0.9}),
            }
        )
        evidence = from_json(result["reranked_documents"])
        assert result["retrieved_count"] == 1
        assert evidence[0]["doc_id"] == "UW-GL-001"

    @pytest.mark.parametrize(
        "bad",
        [float("nan"), float("inf"), float("-inf"), 1.5, -0.2, "0.9", True, None, {}],
        ids=["nan", "inf", "-inf", "over-range", "negative", "string", "bool", "none", "dict"],
    )
    def test_non_finite_or_out_of_range_threshold_falls_back(self, bad):
        """An unchecked NaN threshold compares False against every score and would
        drop the whole evidence set without any error surfacing."""
        result = self.node.execute(
            {
                "retrieved_documents": to_json(_candidates()),
                "runtime_settings": to_json({"score_threshold": bad}),
            }
        )
        assert result["retrieved_count"] == 2, f"score_threshold={bad!r} was not rejected"

    def test_empty_candidates_is_success_with_no_evidence(self):
        result = self.node.execute({"retrieved_documents": to_json([])})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieved_count"] == 0
        assert from_json(result["reranked_documents"]) == []

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── GenerateAnswerNode (inner domain node 4, ANONYMOUS) — grounding + abstention


class TestGenerateAnswerNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.generate_answer_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.generate_answer_node import GenerateAnswerNode

        self.node = GenerateAnswerNode()

    def test_grounded_answer_cites_only_supplied_evidence(self):
        state = {"retrieval_query": _VALID_QUERY, "reranked_documents": to_json(_evidence())}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        ans = from_json(result["generated_answer"])
        assert ans["grounded"] is True
        assert ans["evidence_count"] == 2
        assert ans["citations"] == ["UW-GL-001", "UW-GL-014"]
        # every cited doc_id appears in the answer body ...
        for doc_id in ("UW-GL-001", "UW-GL-014"):
            assert doc_id in ans["answer_text"]
        # ... and a doc NOT in the evidence is never cited or invented (grounding).
        assert "UW-ACT-101" not in ans["citations"]
        assert "UW-ACT-101" not in ans["answer_text"]

    def test_abstains_when_no_evidence(self):
        state = {"retrieval_query": _VALID_QUERY, "reranked_documents": to_json([])}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        ans = from_json(result["generated_answer"])
        assert ans["grounded"] is False
        assert ans["citations"] == []
        assert ans["evidence_count"] == 0
        assert ans["confidence"] == "low"
        assert "manual underwriter" in ans["answer_text"].lower()

    def test_missing_reranked_documents_abstains(self):
        result = self.node.execute({"retrieval_query": _VALID_QUERY})
        ans = from_json(result["generated_answer"])
        assert ans["grounded"] is False

    def test_confidence_bands(self):
        from src.nodes.generate_answer_node import _confidence

        assert _confidence(0, 0.0) == "low"
        assert _confidence(3, 0.95) == "high"
        assert _confidence(2, 0.60) == "medium"
        assert _confidence(1, 0.50) == "low"

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── OutputFormatNode (inner domain node 5, ANONYMOUS) ──────────────────────────


class TestOutputFormatNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.output_format_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.output_format_node import OutputFormatNode

        self.node = OutputFormatNode()

    def test_assembles_grounded_answer_document(self):
        state = {
            "generated_answer": to_json(_answer_obj(grounded=True)),
            "reranked_documents": to_json(_evidence()),
        }
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        doc = result["rag_answer"]
        assert result["result"] == doc
        assert "INSURANCE UNDERWRITING" in doc
        assert "Grounded: YES" in doc
        for doc_id in ("UW-GL-001", "UW-GL-014"):
            assert f"[{doc_id}]" in doc

    def test_ungrounded_answer_document_marks_no_evidence(self):
        state = {
            "generated_answer": to_json(_answer_obj(grounded=False)),
            "reranked_documents": to_json([]),
        }
        doc = self.node.execute(state)["rag_answer"]
        assert "Grounded: NO" in doc
        assert "none" in doc.lower()

    def test_uncited_passage_is_refused(self):
        """The grounding invariant: every citation names a passage in the evidence set.

        This template renders no monetary aggregates, so the rounding grid other
        templates enforce does not apply. Grounding is the invariant it claims,
        so grounding is the one enforced at the boundary — separately from
        composition, so a future change to composition cannot quietly start
        emitting an uncited determination.
        """
        answer = _answer_obj(grounded=True)
        answer["citations"] = answer["citations"] + ["UW-NOT-RETRIEVED"]
        result = self.node.execute({"generated_answer": to_json(answer), "reranked_documents": to_json(_evidence())})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("evidence set" in e for e in result["error_log"])
        assert "rag_answer" not in result

    def test_grounded_answer_passes_the_invariant(self):
        result = self.node.execute(
            {
                "generated_answer": to_json(_answer_obj(grounded=True)),
                "reranked_documents": to_json(_evidence()),
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_missing_generated_answer_returns_error(self):
        result = self.node.execute({"reranked_documents": to_json(_evidence())})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("generated_answer" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── PostProcessNode (outer post_process, the output boundary, ANONYMOUS) ──────


class TestPostProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.post_process_node import PostProcessNode

        self.node = PostProcessNode()

    def test_clean_answer_passes_gate(self):
        answer = "INSURANCE UNDERWRITING — GROUNDED ANSWER\nStandard build limits apply."
        result = self.node.execute({"rag_answer": answer, "result": answer})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == answer
        assert result["result"] == answer

    def test_empty_answer_uses_fallback(self):
        result = self.node.execute({"rag_answer": "", "result": ""})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "No grounded answer" in result["formatted_output"]

    def test_violation_withholds_and_clears_every_output_bearing_field(self):
        """A violation must CLEAR the output-bearing fields, not merely flag the status.

        The framework resolves the caller's output as `formatted_output or result`
        with no status check, so a gate that returns ERROR while leaving the
        document in state still ships it inside the error envelope. Each cleared
        key is asserted PRESENT in the returned delta as well as empty: a delta
        that simply omits the key leaves the previous value in state, and an
        `assert not result.get(field)` would pass on exactly that defect.
        """
        from src.nodes.post_process_node import _OUTPUT_BEARING_FIELDS

        leaky = "UNDERWRITING ANSWER\ntoken=sk-abcdefghij0123456789ABCDEF"
        result = self.node.execute(
            {
                "rag_answer": leaky,
                "generated_answer": to_json({"answer_text": leaky}),
                "reranked_documents": to_json([{"doc_id": "UW-GL-001", "text": leaky}]),
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        # The notice is TRUTHY: an empty formatted_output re-activates the fallback.
        assert result["formatted_output"]
        assert result["formatted_output"] == result["result"]
        assert "WITHHELD" in result["formatted_output"]
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in result, f"{field} must be present in the delta, not omitted"
            assert not result[field], f"{field} must be cleared"
        # Neither the released text nor the matched value may appear anywhere.
        assert "sk-abcdefghij0123456789ABCDEF" not in str(result)
        assert "UNDERWRITING ANSWER\n" not in str(result)
        assert any("withheld" in e for e in result["error_log"])

    def test_violation_message_names_a_reason_not_the_value(self):
        result = self.node.execute({"rag_answer": "note password = supersecret123"})
        joined = " ".join(result["error_log"])
        assert "credential_assignment" in joined
        assert "supersecret123" not in joined
        assert "supersecret123" not in str(result)

    def test_screen_is_a_superset_of_the_framework_detector(self):
        """Every pattern the framework recognises must be caught here too.

        A value the framework catches and this node misses is not a smaller net:
        the framework raises inside the node wrapper, which then returns a bare
        error partial and discards the clearing above — a containment bypass.
        """
        from framework.security.credential_detector import detect_credentials
        from src.nodes.post_process_node import _security_gate_output

        samples = [
            "note sk_live_" + "abcdefghijklmnop1234",
            "note AKIAIOSFODNN7EXAMPLE",
            "note postgresql://db.internal:5432/underwriting",
            "note eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "note sk-abcdefghij0123456789ABCDEF",
            "note Bearer abcdefghijklmnop1234",
        ]
        for sample in samples:
            assert detect_credentials(sample), f"probe {sample!r} does not trip the framework"
            assert _security_gate_output(sample) is not None, f"{sample!r} escaped the screen"

    def test_screen_walks_nested_structures(self):
        from src.nodes.post_process_node import _security_gate_output

        nested = {"evidence": [{"passage": {"text": "AKIAIOSFODNN7EXAMPLE"}}]}
        assert _security_gate_output(nested) == "aws_key"
        assert _security_gate_output({"evidence": [{"passage": {"text": "clean text"}}]}) is None

    def test_domain_pattern_extends_rather_than_replaces(self):
        from src.nodes.post_process_node import _security_gate_output

        assert _security_gate_output("password = supersecret123") == "credential_assignment"
        assert _security_gate_output("A perfectly clean underwriting answer.") is None

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── Graph wiring: outer AgentBaseGraph + inner BaseGraph (nested composition) ──


class TestOuterGraphComposition:
    def test_registers_five_backbone_slots(self):
        from src.graph.graph import (
            InsuranceUnderwritingRiskScoringAgent,
            UnderwritingRAGGraphNode,
        )
        from src.nodes.post_process_node import PostProcessNode
        from src.nodes.pre_process_node import PreProcessNode

        agent = InsuranceUnderwritingRiskScoringAgent()
        agent.compile()
        assert set(agent._nodes.keys()) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], UnderwritingRAGGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.graph import InsuranceUnderwritingRiskScoringAgent

        agent = InsuranceUnderwritingRiskScoringAgent()
        assert agent.name == "InsuranceUnderwritingRiskScoringAgent"
        assert agent.state_schema is State

    def test_main_slot_graphnode_contracts(self):
        from src.graph.graph import UnderwritingRAGGraphNode

        node = UnderwritingRAGGraphNode()
        assert node.error_strategy == "propagate"
        assert node.propagate_hitl is False
        # extract_input prefers validated_input, falls back to user_input
        assert node.extract_input({"validated_input": "V", "user_input": "U"}) == "V"
        assert node.extract_input({"user_input": "U"}) == "U"

    def test_merge_output_maps_subresult_keys(self):
        from src.graph.graph import UnderwritingRAGGraphNode

        node = UnderwritingRAGGraphNode()
        sub_result = {
            "rag_answer": "ANSWER",
            "generated_answer": "{}",
            "reranked_documents": "[]",
            "retrieved_count": 2,
            "result": "ANSWER",
            "status": AgentStatus.SUCCESS.value,
            "node_history": ["x"],  # not forwarded by merge_output
        }
        delta = node.merge_output({}, sub_result)
        assert delta["rag_answer"] == "ANSWER"
        assert delta["retrieved_count"] == 2
        assert delta["result"] == "ANSWER"
        assert delta["status"] == AgentStatus.SUCCESS.value
        # reranked_documents IS forwarded: the output boundary screens it as
        # part of the output-bearing set, so it has to exist at that level.
        assert delta["reranked_documents"] == "[]"
        assert set(delta.keys()) == {
            "rag_answer",
            "generated_answer",
            "reranked_documents",
            "retrieved_count",
            "result",
            "status",
        }

    def test_agent_level_output_filter_redacts(self):
        from src.graph.graph import InsuranceUnderwritingRiskScoringAgent

        agent = InsuranceUnderwritingRiskScoringAgent()
        redacted = agent._security_gate_output("leak sk-abcdefghij0123456789ABCDEF end")
        assert "REDACTED" in redacted
        assert "sk-abcdefghij0123456789ABCDEF" not in redacted
        assert agent._security_gate_output(None) is None
        assert agent._security_gate_output("clean text") == "clean text"

    def test_agent_level_output_filter_uses_the_framework_pattern_set(self):
        """The same superset rule as the node boundary: no narrower local set."""
        from src.graph.graph import InsuranceUnderwritingRiskScoringAgent

        agent = InsuranceUnderwritingRiskScoringAgent()
        for secret in (
            "sk_live_" + "abcdefghijklmnop1234",
            "AKIAIOSFODNN7EXAMPLE",
            "postgresql://db.internal:5432/underwriting",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        ):
            filtered = agent._security_gate_output(f"answer {secret} end")
            assert secret not in filtered, f"{secret!r} survived the agent-level filter"
            assert "REDACTED" in filtered


class TestInnerDomainGraph:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        for mod in (
            "input_validate_node",
            "retrieve_node",
            "rerank_filter_node",
            "generate_answer_node",
            "output_format_node",
        ):
            monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)

    def test_registers_five_domain_nodes(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.register_nodes()
        assert set(g._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        assert g.name == "ins_c2_016_underwriting_retrieval_workflow"
        assert g.state_schema is State

    def test_inner_graph_invoke_produces_grounded_answer(self):
        """Standalone inner-graph invoke (ANONYMOUS caller) runs the linear RAG
        pipeline and shapes the get_output() dict consumed by outer merge_output()."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(_VALID_QUERY, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["rag_answer"] is not None
        assert result["retrieved_count"] >= 2
        assert "INSURANCE UNDERWRITING" in result["rag_answer"]
        assert "Grounded: YES" in result["rag_answer"]
        # get_output shape couples with the outer merge_output keys.
        for key in ("rag_answer", "generated_answer", "retrieved_count", "result", "status"):
            assert key in result

    def test_inner_graph_abstains_when_no_grounding(self):
        """A query with no KB overlap must ABSTAIN — no grounded evidence, no invented
        determination — end to end through the inner pipeline."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke("hello world foobar nomatch topic", ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieved_count"] == 0
        assert "Grounded: NO" in result["rag_answer"]
        assert "manual underwriter" in result["rag_answer"].lower()
