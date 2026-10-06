# Test Specification — INS-C2-016

## 1. Test strategy

- **Agent**: INS-C2-016 — retrieval-grounded underwriting question answering
  (two-layer nested composition: outer `AgentBaseGraph` backbone + inner
  `DomainWorkflowGraph`).
- **Test types**: unit (per node, per helper, plus graph wiring) ·
  proof-of-boundary (framework contracts) · end-to-end through the real HTTP
  entry point.
- **Framework provisioning**: `agenticstar-agentcore` is installed from the
  package registry by CI. The tests import the shipped modules; there are no
  stub nodes.
- **Audit events**: `emit_trace_event` is patched at the node module level in
  unit tests so no audit backend is called. It is never patched through
  `sys.modules`, which would break the real `shared` package the framework
  imports at load time.

### Test file map

| File | Scope |
|------|-------|
| `tests/unit/test_nodes.py` | all 7 domain/backbone nodes + outer and inner graph wiring |
| `tests/unit/test_caller_contract.py` | the caller-data contract, the instruction-override screen (both directions), and the validation helpers |
| `tests/unit/test_main_node.py` | the flat-composition reference node, the `execute(self, state)` signature rule, and the trust gate |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | TC-06/TC-07 — the framework gates cannot be overridden |
| `tests/proof_of_boundary/test_invoke_e2e.py` | PB-8 — real `/invoke`: serving, configuration liveness, input dependence, the context channel, refusals, output-boundary containment |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6 — per-node and backbone invoke order, the trust gate, payload alignment |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4 — import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2/PB-5 — State serialisation and credential safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7 — human-in-the-loop interrupt propagation; skipped, this template does not enable it |

### The pipeline under test

```
input_validate -> retrieve -> rerank_filter -> generate_answer -> output_format
```

- **Grounding contract**: the answer is composed from the reranked evidence
  ONLY. Every citation is an evidence `doc_id`, checked in `output_format`
  before the document is released; no determination is invented for a factor no
  cited passage covers.
- **Abstention contract**: when the evidence set is empty — no knowledge-base
  overlap, or nothing cleared the relative threshold — the agent ABSTAINS:
  `grounded=False`, `citations=[]`, `confidence=low`, and the answer routes the
  case to a manual underwriter instead of emitting a risk score.

### Canonical question

The input is a plain-text underwriting question, not a JSON payload. The
question used by the backbone invoke test and by `deploy/invoke_payload.json`
(the two must stay identical — asserted by
`test_invoke_payload_matches_the_canonical_question`):

```
What are the BMI build chart limits and treated hypertension blood pressure
rating guidelines for a smoker life insurance applicant?
```

Its terms overlap the knowledge base (build/BMI chart `UW-GL-001`,
blood-pressure rating `UW-GL-014`), so retrieval returns candidates, the top two
clear the relative rerank at the shipped `score_threshold` of 0.75, and the
answer is grounded with citations `[UW-GL-001, UW-GL-014]`.

## 2. Framework compliance

| TC-ID | Test | Expected result | Where |
|-------|------|----------------|-------|
| TC-01 | State is a flat `TypedDict`, domain fields `NotRequired`, no Pydantic/dataclass | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Empty, oversized or non-string question refused at PreProcessNode; empty / too-short query refused at InputValidateNode | `status=error`, error_log populated, no output key | `TestQueryBounds`, `TestInputValidateNode` |
| TC-03 | No credential material in State | AST scan + CI credential gate: 0 violations | `test_state_safety.py` + CI |
| TC-04 | Every node's `execute` signature is `(self, state)` | a `config` parameter is never populated, so a node reading one runs on defaults while its configuration looks live | `TestNodeExecuteSignatures` |
| TC-05 | One domain audit event inside every node `execute()` | ≥1 domain event per node, positional form | CI audit-trace gate |
| TC-06 | The framework input gate cannot be overridden | `TypeError` at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-07 | The framework output gate cannot be overridden | `TypeError` at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | `required_trust_level` enforced in `__call__` before `execute()` | ANONYMOUS refused, VERIFIED_EXTERNAL admitted | `TestTrustGate` |
| TC-08a | Outer `PreProcessNode` is VERIFIED_EXTERNAL; inner nodes and post_process are ANONYMOUS | trust level asserted per node | `test_trust_level_*` |
| TC-11 | The output boundary withholds AND clears on a violation | truthy notice, every output-bearing key present in the delta and empty, reason code not the value | `TestPostProcessNode`, `TestOutputBoundaryContainment` |
| TC-12 | The domain screen is a superset of the framework detector | every framework pattern also trips the local screen | `test_screen_is_a_superset_of_the_framework_detector` |

## 3. Proof of boundary

| PB-ID | Boundary | Expected result | Where |
|-------|----------|----------------|-------|
| PB-2 | State serialisation | primitives only; no Pydantic/dataclass | `test_state_safety.py` |
| PB-4 | Import isolation | 0 platform-internal imports under `src/` | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named field or prohibited type in State | `test_state_safety.py` |
| PB-6 | Per-node invoke order | node_start → trust gate → input gate → `execute()` → output gate → node_complete, for every node | `TestInvokeOrder` |
| PB-6b | Backbone invoke order | `status=success`; node_history = `[Initialize, PreProcess, UnderwritingRAGGraphNode, PostProcess, Finalize]` | `TestBackboneInvokeOrder` |
| PB-6c | Real external caller | VERIFIED_EXTERNAL context, never an internal one; inner ANONYMOUS nodes accept the pass-through | `TestBackboneInvokeOrder` |
| PB-6d | Payload alignment | `deploy/invoke_payload.json["input"]` equals the canonical question | `test_invoke_payload_matches_the_canonical_question` |
| PB-7 | Human-in-the-loop interrupt propagation | skipped — `hitl.enabled` is not set, `propagate_hitl=False`, no `interrupt()` checkpoint | `test_pb7_hitl_interrupt_propagation.py` |
| PB-8 | Real HTTP entry point | see section 5 | `test_invoke_e2e.py` |

## 4. Business logic

| BL-ID | Test | Input | Expected result |
|-------|------|-------|----------------|
| BL-01 | Grounded happy path | canonical question | cited answer document, `Grounded: YES` |
| BL-02 | Knowledge-base retrieval | overlapping query | ≥2 candidates, scores sorted descending, `UW-GL-001` present |
| BL-03 | No overlap | non-matching query | 0 candidates, `status=success` |
| BL-04 | Declared `top_k` | `runtime_settings.top_k=1` | at most 1 candidate |
| BL-04a | Invalid `top_k` | NaN, ±Infinity, 0, negative, out of range, string, bool, None, list | falls back to the default; the cap is never silently disabled |
| BL-05 | Relative rerank + filter | candidates 0.30 / 0.25 / 0.05 | top → 1.0; threshold 0.75 keeps 2; 0.9 keeps 1 |
| BL-05a | Invalid `score_threshold` | NaN, ±Infinity, out of range, string, bool, None, dict | falls back to the default |
| BL-06 | Grounded composition cites only the evidence | 2-passage evidence | `grounded=True`, citations exactly the supplied ids, no uncited passage |
| BL-07 | Abstention on empty evidence | `reranked_documents=[]` | `grounded=False`, `citations=[]`, `confidence=low`, "manual underwriter" |
| BL-08 | Confidence bands | evidence/score matrix | high / medium / low by coverage |
| BL-09 | Document assembly | grounded and ungrounded answer objects | `Grounded: YES`/`NO`, citation markers, evidence block |
| BL-10 | End-to-end abstention | non-matching question | `retrieved_count=0`, `Grounded: NO` |
| BL-11 | Graph key coupling | inner `get_output` ↔ outer `merge_output` | 6 coupled keys mapped; changed keys only |

## 5. End to end, through the real entry point

| E2E-ID | Question it answers | Expected result |
|--------|--------------------|-----------------|
| E2E-01 | Can the agent serve a request at its declared trust level? | bearer-authenticated `/invoke` returns `status=success` with a grounded answer and the full backbone node_history |
| E2E-02 | Is the entry point actually gated? | no bearer token → 401 |
| E2E-03 | Does the answer depend on the question? | two questions produce two different answers, each citing what its own question retrieved |
| E2E-04 | Does the agent abstain end to end? | an unrelated question returns `Grounded: NO` |
| E2E-05 | Does a declared value reach the pipeline? | changing `retrieval.score_threshold` and `retrieval.top_k` in `config/config.yaml` changes the answer the caller receives |
| E2E-06 | Does an invalid declared value fail safe? | an out-of-range threshold produces the same answer as the shipped default |
| E2E-07 | Is the context channel closed and inert? | an inert channel is accepted; unknown keys are dropped before invoke |
| E2E-08 | Is a credential-shaped context value refused readably? | HTTP 400 naming `input_context.channel`, never echoing the value; ordinary domain text on the same field still passes |
| E2E-09 | Does the refusal set match the framework's block set? | refused exactly when `detect_credentials_in_value` fires — the anti-drift property |
| E2E-10 | Are instruction-override payloads refused? | control tokens and directives → `status=error`, no output, attack text never echoed |
| E2E-11 | Does the boundary contain a violation? | a knowledge-base passage carrying credential material → `status=error`, truthy withheld notice, no released text, no passage, no traceback, no source path, and `PostProcessNode` in node_history |
| E2E-12 | Clean-path control | the same request without the poisoned passage still returns its real answer, so a refuse-everything boundary cannot pass |
| E2E-13 | Framework-known material | terminates inside the retrieval node with no output at all — the difference between the two paths recorded as a tested fact |

The containment fault (E2E-11) is injected on the **data** path — a
knowledge-base passage carrying credential material, which is what a mis-pasted
operations note in a guideline document looks like — never on the boundary
itself. Patching the boundary to force a violation would test the patch rather
than the pipeline.

## 6. Negative and boundary cases

| Case | Node | Expected |
|------|------|----------|
| empty / whitespace `user_input` | PreProcessNode | `status=error`, "empty" |
| non-string `user_input` | PreProcessNode | `status=error` |
| question over 4000 characters | PreProcessNode | `status=error`, refused not truncated |
| chat-template control token or directive | PreProcessNode | `status=error`, value never echoed |
| real underwriting language containing screen keywords | PreProcessNode | accepted — the screen must not refuse genuine work |
| `input_context` not a mapping, unknown key, or non-inert channel | PreProcessNode | `status=error`, hostile key names reported positionally |
| escaped directive behind JSON `\u` escapes | screen_structure | caught after parsing |
| nesting deeper than 8 levels | screen_structure | refused |
| empty query | InputValidateNode | `status=error`, "empty" |
| fewer than 2 words | InputValidateNode | `status=error`, "substance" |
| missing `retrieval_query` | RetrieveNode | `status=error` |
| empty candidate set | RerankFilterNode | `status=success`, empty evidence (not an error) |
| empty / missing evidence | GenerateAnswerNode | `status=success`, abstains |
| citation naming no supplied passage | OutputFormatNode | `status=error`, grounding violation event |
| missing `generated_answer` | OutputFormatNode | `status=error` |
| empty answer | PostProcessNode | fallback notice, `status=success` |
| credential material in the answer | PostProcessNode | withheld, output-bearing fields cleared, `status=error` |
