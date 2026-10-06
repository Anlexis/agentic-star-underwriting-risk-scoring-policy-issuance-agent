# Underwriting Risk Scoring & Policy Issuance Agent

AI agent for scoring underwriting risk and producing policy issuance reports, built with Agentic Star.

> **Category**: Cat 2 (a domain pipeline — a fixed sequence of steps that answers one kind of question)
> **Industry**: Insurance
> **Template ID**: INS-C2-016

## Overview

Answers insurance-underwriting questions from a knowledge base of underwriting guidelines,
actuarial tables, product terms and regulatory notices — and says so plainly when the knowledge
base does not cover the question.

Given a question such as *"what are the build-chart limits and blood-pressure rating guidelines
for a smoker applicant?"*, the agent retrieves the relevant guideline passages, keeps only the
ones that clear a relevance threshold, and returns a plain-text determination that cites the
passages it used, with a confidence band derived from how much evidence it found.

It is deliberately a **retrieval-grounded answering** agent, not a decision engine: it never
issues a binding risk score or premium. When no passage clears the threshold it abstains —
returning an explicit "insufficient grounded evidence, route to a manual underwriter" answer
rather than an unsupported opinion. Every citation in the answer is checked against the evidence
set before the document is released, and the answer passes an output boundary that withholds it
entirely if credential-shaped material is found in it.

The bundled knowledge base is a small worked example (build/BMI limits, blood-pressure rating,
smoker mortality, group eligibility, disclosure duty, occupational classes). Replace it with your
own corpus and retriever — the pipeline and its contracts do not change.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >= 3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode: if the platform is unreachable or the installed framework version does not match, start-up
fails rather than bringing up a partially working agent. This is intentional — a half-running
agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/02_design.md` for the architecture and `docs/03_test_spec.md` for what the test suite
covers.

## Customising

1. `config/agent.yaml` is the registration manifest (identity and entry point);
   `config/config.yaml` holds the runtime parameters — the retry budget and the retrieval
   tuning (`retrieval.top_k`, `retrieval.score_threshold`, `retrieval.hybrid_search`).
   Every declared value is validated for type, finiteness and range before it is used; an
   invalid entry is dropped and the node falls back to its documented default.
2. Replace the knowledge base in `src/nodes/retrieve_node.py` with your own corpus and
   retriever. The node contract (a query in, scored passages out) does not change.
3. `prompts/insurance_underwriting.j2` states the grounding contract for the answer step.
4. Review the caller-data contract in `src/nodes/pre_process_node.py` and the output boundary
   in `src/nodes/post_process_node.py` before you widen either.
5. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
