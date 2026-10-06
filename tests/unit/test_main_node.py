# INS-C2-016 — Unit Tests: the flat-composition reference node + the trust gate

import inspect

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.main_node import MainNode
from src.nodes.pre_process_node import PreProcessNode


class TestMainNode:
    """Unit tests for the flat-composition reference node.

    Node instances are invoked as ``node(state)`` (i.e. through
    ``BaseNode.__call__``) rather than by calling ``node.execute(state)``
    directly, so the trust gate and the input/output gates run on every
    invocation exactly as they do in a deployed agent.
    """

    def setup_method(self):
        self.node = MainNode()

    def test_success_path(self):
        """The reference node returns SUCCESS and a non-empty result."""
        state = {
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "validated_input": "test input",
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)  # __call__ -> trust gate -> execute()
        # status is the lowercase .value string, never the bare enum member.
        assert result["status"] == AgentStatus.SUCCESS.value == "success"
        assert result["result"]

    def test_empty_input(self):
        """The reference node handles empty input gracefully."""
        state = {
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "validated_input": "",
            "node_history": [],
            "error_log": [],
        }
        assert self.node(state)["status"] == AgentStatus.SUCCESS.value

    def test_execute_method_signature(self):
        """The node contract is execute(self, state) — one state argument.

        The framework calls ``self.execute(state)`` with a single positional
        argument, so a node that declares an extra ``config`` parameter can
        never receive one: any value it reads from there is dead. The signature
        is asserted so that shape cannot come back.
        """
        assert hasattr(MainNode, "execute")
        params = list(inspect.signature(MainNode.execute).parameters)
        assert params == ["self", "state"], f"execute() must be (self, state), got {params}"


class TestNodeExecuteSignatures:
    """Every domain node takes exactly (self, state).

    A ``config=None`` parameter on execute() is never populated — the framework
    invokes nodes with the state alone — so a node reading its settings from
    there silently runs on its defaults while the declared configuration looks
    live. Settings reach a node through state (``runtime_settings``) instead.
    """

    def test_every_node_execute_takes_state_only(self):
        import importlib
        import pkgutil

        from framework.nodes.base_node import BaseNode

        pkg = importlib.import_module("src.nodes")
        offenders = []
        for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="src.nodes."):
            module = importlib.import_module(modname)
            for attr in vars(module).values():
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseNode)
                    and attr.__module__ == modname
                    and not inspect.isabstract(attr)
                ):
                    params = list(inspect.signature(attr.execute).parameters)
                    if params != ["self", "state"]:
                        offenders.append(f"{attr.__name__}: {params}")
        assert not offenders, "execute() must be (self, state): " + "; ".join(offenders)


class TestTrustGate:
    """The trust gate runs inside ``BaseNode.__call__``, before ``execute()``.

    PreProcessNode is the agent's external trust boundary
    (required_trust_level = VERIFIED_EXTERNAL); the inner domain nodes run
    ANONYMOUS. A caller presenting no vouched trust must be denied before
    execute() runs, and the gate must admit a properly vouched caller.

    On insufficient trust the gate RETURNS an error dict (it does not raise):
    status == AgentStatus.ERROR.value, error_log carries the denial, and
    execute()'s output keys are absent.
    """

    def setup_method(self):
        self.node = PreProcessNode()

    # Payload chosen so the framework's input gate admits it in the positive
    # control: lowercase, no '@', no digit groups, no two consecutive
    # Title-Case tokens.
    _PAYLOAD = "please assess the flood risk exposure for a small riverside warehouse"

    def test_gate_rejects_untrusted_caller_before_execute(self):
        state = {
            "user_input": self._PAYLOAD,
            "input_context": {"channel": "web"},
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)  # via __call__ — the trust gate runs first

        assert result["status"] == AgentStatus.ERROR.value
        assert any(
            "trust gate denied" in entry for entry in result.get("error_log", [])
        ), f"expected a trust-gate denial in error_log, got: {result.get('error_log')}"
        # execute() never ran, so its output key must be absent.
        assert "validated_input" not in result

    def test_gate_admits_trusted_caller(self):
        state = {
            "user_input": self._PAYLOAD,
            "input_context": {"channel": "web"},
            "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)

        assert result["status"] == AgentStatus.SUCCESS.value
        assert "validated_input" in result
