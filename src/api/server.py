"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, the agent gateway calls agent.invoke() directly.

import json
import os
import secrets
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.security.credential_detector import detect_credentials_in_value
from pydantic import BaseModel
from shared.secrets import factory as secrets_factory

from src.graph.graph import InsuranceUnderwritingRiskScoringAgent as Graph
from src.graph.graph import runtime_config

app = FastAPI(title="Agent")

# The platform registry loads config/config.yaml and passes it as
# Graph(config=...); the standalone server mirrors that exactly, so the
# declared runtime parameters (the retry budget, the retrieval tuning) are live
# in both deployments rather than only in the file.
agent = Graph(config=runtime_config())
agent.compile()
# Namespace / agent_name match the agent's manifest values.
agent.provision_secrets(secrets_factory(namespace="ins-c2-016", agent_name="InsuranceUnderwritingRiskScoringAgent"))

# Coarse adapter guard on the structured channel. Per-field bounds (the inert
# token alphabet, the closed key set) are enforced inside the graph; this only
# stops an oversized body from reaching it at all.
_MAX_INPUT_CONTEXT_BYTES = 262_144

# The structured channel's closed key set. Unknown keys are DROPPED here rather
# than forwarded: the framework's first node copies input_context verbatim into
# its own result, and the output gate then scans every value of that result — so
# an undeclared key is not merely ignored downstream, it is carried all the way
# into a gate that can fail the whole request. Validators ignore undeclared
# keys; ignoring is not stripping, and only stripping is immunity.
_ALLOWED_CONTEXT_KEYS = ("channel",)


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Structured invocation parameters. The only supported key is `channel`, an
    # inert lowercase token; it is validated field by field inside the graph
    # (PreProcessNode).
    input_context: dict[str, Any] | None = None


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Any:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone-deployment caller auth: when INVOKE_AUTH_TOKEN is set on the
    # server environment, callers that no upstream middleware vouched for
    # (still ANONYMOUS) must present it as a Bearer token and then run at
    # VERIFIED_EXTERNAL. Middleware-established trust is never demoted. This
    # adapter is the entry-point auth boundary (the standalone equivalent of
    # the platform's auth middleware) — a deployment-level caller credential,
    # not an agent secret, so ctx.secrets does not apply: no InvocationContext
    # exists before authentication.
    #
    # Required here specifically: PreProcessNode occupies the pre_process
    # backbone slot and declares required_trust_level = VERIFIED_EXTERNAL.
    # Nothing else sets request.state.trust_level in a standalone deployment,
    # so without this boundary every deployed invoke arrives ANONYMOUS, the
    # trust gate denies it, and the agent returns status="error".
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    raw_context = req.input_context or {}
    if raw_context and len(json.dumps(raw_context, default=str)) > _MAX_INPUT_CONTEXT_BYTES:
        raise HTTPException(status_code=413, detail="input_context exceeds the maximum allowed size.")
    input_context = {k: raw_context[k] for k in _ALLOWED_CONTEXT_KEYS if k in raw_context}

    # A credential-shaped value anywhere in input_context makes the framework's
    # very first node fail with an opaque error, before any template code runs.
    # The request cannot succeed either way, so refuse it here with something
    # the caller can act on. The framework's own detector is used so this
    # refusal set is exactly its block set — a local approximation would drift.
    # Fields are scanned one at a time purely so the offending one can be named;
    # detect_credentials_in_value over a mapping is defined as the union over
    # its values, so per-field scanning blocks exactly the same set.
    for position, (name, value) in enumerate(input_context.items(), start=1):
        if detect_credentials_in_value(value):
            raise HTTPException(
                status_code=400,
                detail=f"input_context.{_safe_name(name, position)} looks like a credential and was refused.",
            )

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return agent.invoke(req.input, ctx=ctx, input_context=input_context)


def _safe_name(name: str, position: int) -> str:
    """Name a rejected field only when the name itself is safe to echo."""
    if name.isidentifier() and len(name) <= 32 and not detect_credentials_in_value(name):
        return name
    return f"field #{position}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "InsuranceUnderwritingRiskScoringAgent"}
