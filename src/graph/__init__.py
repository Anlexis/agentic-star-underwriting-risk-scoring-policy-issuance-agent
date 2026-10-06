"""AgentCore Platform v1.0"""

# Expose the agent class at the package level so a package-level import
# resolves it. config/agent.yaml declares the entry point as the single dotted
# path src.graph.graph.InsuranceUnderwritingRiskScoringAgent; this re-export
# means `from src.graph import InsuranceUnderwritingRiskScoringAgent` also
# works, which is what an embedding application usually reaches for.
from src.graph.graph import InsuranceUnderwritingRiskScoringAgent

__all__ = ["InsuranceUnderwritingRiskScoringAgent"]
