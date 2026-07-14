"""PDAM: a reproducible testbed for Persistent-State and Delayed-Activation
Attacks on Memory-Augmented LLM Agents.

Implements the experimental platform of the research design document (§7, §11): a scenario
engine (M1), pluggable memory adapters (M2), an attack generator for the A1-A8
taxonomy (M3), a mock tool sandbox (M4), a policy monitor with the §10 defenses
(M5), and an evaluator for the §9.1 metrics (M6).
"""
from .attacks.generator import AttackGenerator, AttackSpec
from .broker import ToolBroker
from .evaluator import aggregate, compare, holm, metrics
from .memory.store import MemoryStore
from .orchestrator import Orchestrator, RunResult
from .planner import Planner, RuleBasedPlanner
from .policy import DefenseConfig, PolicyMonitor
from .scenario import Scenario, all_scenarios, build_scenario, load_scenario
from .schema import AttackType, PolicyViolation, State, TrustLevel

__version__ = "1.0.0"

__all__ = [
    "Orchestrator", "RunResult",
    "Scenario", "build_scenario", "all_scenarios", "load_scenario",
    "AttackGenerator", "AttackSpec",
    "MemoryStore", "ToolBroker",
    "Planner", "RuleBasedPlanner",
    "DefenseConfig", "PolicyMonitor",
    "metrics", "aggregate", "compare", "holm",
    "AttackType", "PolicyViolation", "TrustLevel", "State",
    "__version__",
]


def run_scenario(scenario: "Scenario", planner: "Planner | None" = None) -> "RunResult":
    """Convenience: execute one scenario and return its RunResult."""
    return Orchestrator(scenario, planner).run()
