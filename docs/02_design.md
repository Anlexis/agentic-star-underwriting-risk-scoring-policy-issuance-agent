# Template Design Specification — INS-C2-016

## Position in the framework

- **Agent class**: `InsuranceUnderwritingRiskScoringAgent`

| Layer | Choice |
|---|---|
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| State | flat `TypedDict` composition (never a Pydantic model — checkpoints are msgpack-serialised; dict/list fields are stored JSON-serialised) |
| Node | framework inheritance; a node overrides `execute(self, state) -> dict` and nothing else |
| Graph | composition — `register_nodes()` for slot substitution, plus a nested `GraphNode` delegating to an inner `BaseGraph` |

## Pattern

Retrieval-grounded question answering over an insurance-underwriting knowledge
base (underwriting guidelines, actuarial tables, product terms, regulatory
notices). Two-layer **nested** composition: the fixed 5-slot outer backbone
delegates the domain workflow to an inner `DomainWorkflowGraph` through a
`GraphNode` in the `main` slot. Single responsibility — question answering only,
with no document-generation pipeline mixed in.

## Architecture overview

### Outer backbone

| Node | Responsibility | Input state | Output state | Class |
|------|---------------|-------------|--------------|-------|
| initialize | schema version, session id, caller trust | — | (framework) | InitializeNode (default) |
| pre_process | trust boundary (VERIFIED_EXTERNAL) + the caller-data contract | user_input, input_context | validated_input, enriched_context | PreProcessNode |
| main | delegate to the inner workflow | validated_input, input_context | rag_answer, generated_answer, reranked_documents, retrieved_count, result | UnderwritingRAGGraphNode |
| post_process | output boundary — screen, then surface or withhold | rag_answer / result | formatted_output, result | PostProcessNode |
| finalize | response metadata, total time | — | (framework) | FinalizeNode (default) |

### Inner pipeline (`DomainWorkflowGraph`)

| Node | Responsibility | Input | Output |
|------|---------------|-------|--------|
| input_validate | domain validation; build the retrieval query | validated_input | retrieval_query |
| retrieve | fetch candidate passages (`top_k`, `hybrid_search`) | retrieval_query, runtime_settings | retrieved_documents, retrieved_count |
| rerank_filter | relative rerank + filter (`score_threshold`) | retrieved_documents, runtime_settings | reranked_documents, retrieved_count |
| generate_answer | compose the grounded answer from the evidence | reranked_documents, runtime_settings | generated_answer |
| output_format | check the grounding invariant, assemble the document | generated_answer, reranked_documents | rag_answer, result |

All inner nodes declare `required_trust_level = TrustLevel.ANONYMOUS`: trust is
enforced once, at the backbone `pre_process` boundary, and the inner nodes must
admit the caller's context as it passes through the subgraph boundary.

### Data flow

```
START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
                                        |  (retry, budget from config/config.yaml)
                                        v
  [inner] input_validate -> retrieve -> rerank_filter -> generate_answer -> output_format -> END
```

## How configuration reaches a node

This is the part of the design most easily got wrong, so it is stated
explicitly.

- `config/agent.yaml` is a **flat registration manifest**: identity, entry point,
  declared trust level, and the compile-time `requires` block. It has no
  runtime section, and a runtime value placed there is read by nothing.
- `config/config.yaml` holds the runtime parameters. The platform registry loads
  it and passes it as `Graph(config=...)`; `src/api/server.py` does the same via
  `runtime_config()`, so a directly deployed agent and a registry-loaded agent
  see identical configuration.
- The framework calls a node as `execute(state)` — **one argument**. A node
  therefore cannot receive a per-invocation config object, and a node that
  declared a `config` parameter would always see `None` and silently run on its
  own defaults while the declared configuration looked live.
- The live route is state. `UnderwritingRAGGraphNode._parent_config()` reads
  `config/config.yaml`, **validates every value** (type, finiteness, range) and
  forwards the ones that pass; `DomainWorkflowGraph._extra_initial_state()`
  seeds them into the inner graph's initial state as `runtime_settings`; each
  node reads its own key from there. A key that is absent or invalid is not
  forwarded, and the node keeps its documented default.

| Declared key | Bounds | Consumed by | Default if absent/invalid |
|---|---|---|---|
| `max_retry` | framework contract | AgentBaseGraph route | 3 |
| `timeout_s` | framework contract | framework | — (no external call is made) |
| `retrieval.top_k` | 1–50 | RetrieveNode | 8 |
| `retrieval.score_threshold` | 0.0–1.0 | RerankFilterNode | 0.75 |
| `retrieval.hybrid_search` | boolean | RetrieveNode | true |
| `llm.system_prompt_template` | path inside the repository | GenerateAnswerNode | not resolved (recorded in the audit event) |

`generation_mode: deterministic` in the manifest is a statement about the code:
no model is invoked. The answer is composed from the caller's question and the
passages that survived reranking, so it varies with its input — which is the
property the tests pin, because a broken generation step and a deterministic one
both return text.

## The caller-data contract

The caller sends a free-text underwriting question, and optionally a structured
`input_context` whose only supported key is `channel`.

| Rule | Where | Behaviour |
|---|---|---|
| Question is a non-empty string, at most 4000 characters | PreProcessNode | refused, not truncated — answering a shortened question is worse than refusing |
| Question carries no chat-template control token (`<|…|>`, `[INST]`, `<<SYS>>`) or instruction-override directive | PreProcessNode via `src/services/service.py` | refused; the template enforces this itself rather than relying on the framework's input policy, which treats some of these forms as audit-only |
| `input_context` keys are a closed set | HTTP adapter drops unknown keys; PreProcessNode refuses them | dropping at the adapter is what confers immunity — an ignored key still travels |
| `input_context.channel` is an inert `[a-z0-9_]{1,32}` token | PreProcessNode | refused otherwise; the label reaches audit events and request metadata |
| No credential-shaped value in `input_context` | HTTP adapter, using the framework's own detector | 400 naming the field, never echoing the value |
| Every declared number is finite and in range | `finite_in_range` | invalid values fall back to the node default; NaN compares False against every bound, so an unchecked one would silently disable the check it feeds |

The contract admits **no caller-supplied number at all** — the payload is a
question string — so there is no caller-controlled numeric anywhere in the
pipeline. The finite-and-bounded rule is applied to the declared configuration
values instead, which are the only numbers the agent acts on.

## The output boundary

Two invariants are enforced, each in its own place and each with its own audit
event:

1. **Grounding** (`OutputFormatNode`): every citation in the rendered document
   names a passage in the evidence set. This template renders no monetary
   aggregates, so the rounding grid some templates enforce does not apply;
   grounding is the invariant this one claims, and it is the one enforced.
2. **Credential containment** (`PostProcessNode`): the assembled document and
   every other output-bearing field are screened before release. The screen is a
   **superset** of the framework's own credential detector — it calls
   `detect_credentials()` directly and layers a domain assignment pattern on top.
   A narrower local set would not be a smaller net: the framework re-scans every
   node result and raises on a match, the node wrapper turns that raise into a
   bare error partial, and the containment below would be discarded.

On a violation the node returns ERROR **and clears** `rag_answer`,
`generated_answer` and `reranked_documents`, and sets `formatted_output` and
`result` to a **truthy** withheld notice. Both halves matter: the framework
resolves the caller's output as `formatted_output or result` with no status
check, so an ERROR that left the document in state would ship it inside the
error envelope, and an empty `formatted_output` would fall through to `result`
for the same reason. The violation message carries a reason code, never the
matched value.

## State definition (key fields)

| Field | Type | Purpose |
|-------|------|---------|
| validated_input | NotRequired[Optional[str]] | normalised question |
| enriched_context | NotRequired[Optional[str]] | JSON request metadata |
| runtime_settings | NotRequired[Optional[str]] | JSON declared settings — how configuration reaches a node |
| retrieval_query | NotRequired[Optional[str]] | knowledge-base query text |
| retrieved_documents | NotRequired[Optional[str]] | JSON list of candidates |
| reranked_documents | NotRequired[Optional[str]] | JSON list of grounded evidence |
| generated_answer | NotRequired[Optional[str]] | JSON answer object |
| rag_answer | NotRequired[Optional[str]] | final cited answer document |
| retrieved_count | NotRequired[Optional[int]] | evidence passages used |
| result | NotRequired[Optional[str]] | caller-facing result |

**State constraints (mandatory):**
- Flat `TypedDict` only — primitives plus JSON-serialised strings
- No credential, token or connection string in State (it reaches the checkpoint store)
- The invocation context travels through the framework, not through State
- No Pydantic models, dataclasses or arbitrary objects (msgpack-incompatible)

## Framework facilities used

- `InvocationContext` — correlation id, session id, caller trust level
- `required_trust_level = VERIFIED_EXTERNAL` on `pre_process` — the external boundary
- `detect_credentials()` / `detect_credentials_in_value()` — the shared pattern set,
  used at the output boundary and at the HTTP adapter so neither can drift from it
- `emit_trace_event(event, payload, state)` — one positional domain event inside
  every `execute()`; the framework emits the node lifecycle events itself and a
  template must not duplicate them

> **Gate behaviour by node type:**
> - `FunctionNode` subclass — the framework's final input/output gates run
>   automatically. A domain node extends them only through
>   `_extra_security_gate_input` / `_extra_security_gate_output`; the domain
>   screen in `PostProcessNode` is a module-level function called from
>   `execute()`, not an override.
> - `GraphNode` (main slot) — deliberate no-op gates at the subgraph boundary;
>   the inner nodes already applied theirs.

## Composition

- **Pattern**: `GraphNode` wrapping an inner `BaseGraph`
- **Target**: `src/graph/domain_workflow_graph.py::DomainWorkflowGraph`
- **Error propagation**: propagate — an inner error fails the request fast
- **Context hand-off**: `src/graph/context_bridge.py`. The framework's
  `GraphNode.execute()` invokes the subgraph without forwarding `input_context`,
  so the outer `extract_input()` stashes it in a ContextVar and the inner
  `_extra_initial_state()` picks it up. Without this the inner graph would see
  `{}` on every real invocation.

## Import isolation

- [x] No platform-internal package is imported
- [x] Import targets are `framework/` and `shared/` only

## Design decisions

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base class | AgentBaseGraph | AutonomousBaseGraph | AgentBaseGraph | fixed multi-step workflow, no autonomous loop |
| Composition | flat single layer | nested subgraph | nested subgraph | keeps the domain pipeline behind the fixed backbone |
| Retrieval | dense only | hybrid | hybrid | the knowledge base carries codes and product identifiers that exact matching finds and embeddings blur |
| Oversized question | truncate | refuse | refuse | a truncated question is a different question |
| Violating output | redact in place | withhold entirely | withhold | redaction leaves the reader unable to tell what was removed, and a partial answer to an underwriting question is a liability |
