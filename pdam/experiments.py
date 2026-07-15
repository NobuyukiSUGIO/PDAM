"""Reviewer-driven robustness experiments (§4.2, §4.3, §4.4, §4.6, §4.7).

Each function returns a list of plain dict rows so the CLI can print and persist
them as CSV. All experiments use the deterministic rule-based planner, so every
number here is byte-reproducible across machines.
"""
from __future__ import annotations

from typing import Optional

from .evaluator import metrics
from .orchestrator import Orchestrator
from .policy import DefenseConfig, PolicyMonitor
from .scenario import all_scenarios
from .utility import run_utility


def _run_matrix(defense: str, cfg: Optional[DefenseConfig] = None) -> list:
    """Run the full 96-scenario matrix under a named defense or explicit cfg."""
    runs = []
    for sc in all_scenarios(defense=("none" if cfg else defense)):
        runs.append(Orchestrator(sc, defense_cfg=cfg).run())
    return runs


# --------------------------------------------------------------------------- #
# §4.7c  Leave-one-component-out ablation of the 3-stage minimal defense
# --------------------------------------------------------------------------- #
LEAVE_ONE_OUT = [
    "minimal_defense",
    "minimal_minus_content_filter",
    "minimal_minus_provenance",
    "minimal_minus_least_privilege",
    "minimal_minus_fragment_limit",
]


def leave_one_out() -> list[dict]:
    rows = []
    for d in LEAVE_ONE_OUT:
        m = metrics(_run_matrix(d))
        rows.append({
            "config": d,
            "asr": m["asr"],
            "authority_amplification": m["authority_amplification_rate"],
            "benign_success": m["benign_task_success"],
        })
    return rows


# --------------------------------------------------------------------------- #
# §4.2  Non-oracle provenance: provenance-only vs 3-stage under estimation noise
# --------------------------------------------------------------------------- #
# (trust_noise, dropout), ordered by increasing provenance-estimation degradation
# so the reported curve (Fig. nonoracle) is a monotone stress sweep.
NOISE_GRID = [(0.0, 0.0), (0.0, 0.25), (0.0, 0.5),
              (0.5, 0.5), (0.75, 0.5), (1.0, 1.0)]


def _prov_only_cfg(noise: float, drop: float) -> DefenseConfig:
    c = DefenseConfig("provenance", provenance_revalidation=True,
                      max_low_trust_fragments=1)
    c.prov_trust_noise, c.prov_dropout = noise, drop
    return c


def _minimal_cfg(noise: float, drop: float) -> DefenseConfig:
    c = DefenseConfig.preset("minimal_defense")
    c.prov_trust_noise, c.prov_dropout = noise, drop
    return c


def non_oracle_provenance() -> list[dict]:
    rows = []
    for noise, drop in NOISE_GRID:
        prov = metrics(_run_matrix("", _prov_only_cfg(noise, drop)))
        mind = metrics(_run_matrix("", _minimal_cfg(noise, drop)))
        rows.append({
            "trust_noise": noise, "dropout": drop,
            "provenance_only_asr": prov["asr"],
            "minimal_3stage_asr": mind["asr"],
        })
    return rows


# --------------------------------------------------------------------------- #
# §5.3  Causal-attribution error: a second oracle, independent of provenance
# --------------------------------------------------------------------------- #
def attribution_noise() -> list[dict]:
    rows = []
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        c = DefenseConfig.preset("minimal_defense")
        c.attribution_dropout = a
        m = metrics(_run_matrix("", c))
        rows.append({
            "attribution_dropout": a,
            "minimal_asr": m["asr"],
            "authority_amplification": m["authority_amplification_rate"],
        })
    return rows


# --------------------------------------------------------------------------- #
# §4.4  Safety/utility Pareto over the legitimate-external-memory suite
# --------------------------------------------------------------------------- #
PARETO_DEFENSES = [
    "none", "content_filter", "least_privilege", "provenance",
    "minimal_defense", "block_external", "confirm_all", "confirm_external",
    "memory_disabled",
]


def safety_utility_pareto() -> list[dict]:
    rows = []
    for d in PARETO_DEFENSES:
        cfg = DefenseConfig.preset(d)
        asr = metrics(_run_matrix(d))["asr"]
        u = run_utility(cfg)
        rows.append({
            "defense": d,
            "attack_asr": asr,
            "call_allow_rate": u.call_allow_rate,
            "task_completion": u.task_completion_rate,      # §5.7 end-to-end
            "external_legit_fpr": u.ext_false_positive_rate,
            "confirmations": u.confirmations,
        })
    return rows


# --------------------------------------------------------------------------- #
# §5.7  Confirmation fatigue: ASR under confirm-external when the user mis-approves
# an attack confirmation with probability p (the honest-user assumption relaxed).
# --------------------------------------------------------------------------- #
def misapproval_sensitivity(seeds: int = 20) -> list[dict]:
    # misapproval is stochastic per state; det_unit gives one fixed draw per
    # (seed, state), so we average ASR over several seeds for a smooth expected
    # curve rather than a single step-function realisation.
    rows = []
    for p in [0.0, 0.01, 0.05, 0.10, 0.20]:
        asrs = []
        for s in range(seeds):
            cfg = DefenseConfig.preset("confirm_external")
            cfg.misapproval_rate = p
            cfg.prov_seed = s
            asrs.append(metrics(_run_matrix("", cfg))["asr"])
        mean = sum(asrs) / len(asrs)
        rows.append({"misapproval_rate": p, "asr": mean,
                     "asr_min": min(asrs), "asr_max": max(asrs)})
    return rows


# --------------------------------------------------------------------------- #
# §4.3  Lifecycle funnel — strictly nested so each stage implies the previous,
# giving a monotone survival curve (fraction of attack runs reaching each stage).
# --------------------------------------------------------------------------- #
FUNNEL_STAGES = ["write", "survive", "retrieve", "synthesize",
                 "dispatch", "effect"]


def _nested(runs) -> dict[str, float]:
    n = len(runs) or 1
    reach = {s: 0 for s in FUNNEL_STAGES}
    for r in runs:
        write = r.attack_saved
        survive = write and r.attack_in_store_at_probe
        retrieve = survive and r.persisted
        synth = retrieve and r.fired_on_probe        # reconstructed AT the probe
        dispatch = synth and r.executed
        effect = dispatch and r.success
        for s, ok in zip(FUNNEL_STAGES,
                         [write, survive, retrieve, synth, dispatch, effect]):
            reach[s] += int(ok)
    return {s: reach[s] / n for s in FUNNEL_STAGES}


def funnel(defenses=("none", "content_filter", "minimal_defense")) -> list[dict]:
    rows = []
    for d in defenses:
        rows.append({"defense": d, **_nested(_run_matrix(d))})
    return rows


# --------------------------------------------------------------------------- #
# §5.4  Real-framework evaluation: same 96 attacks through real LangChain and
# LlamaIndex memory (vector / summary / kv), decoupling attack type from backend
# and exercising real embedding, retrieval, TTL, and LLM summarisation.
# Runs only under the .venv-real extras; the deterministic planner isolates the
# backend's contribution from LLM variance.
# --------------------------------------------------------------------------- #
REAL_STORES = [
    ("langchain", "lc_vector"), ("langchain", "lc_summary"), ("langchain", "lc_kv"),
    ("llamaindex", "li_vector"), ("llamaindex", "li_summary"), ("llamaindex", "li_kv"),
]
REAL_DEFENSES = ["none", "minimal_defense"]


def _run_matrix_backend(backend: str, defense: str) -> list:
    runs = []
    for sc in all_scenarios(defense="none"):
        sc.memory = backend
        runs.append(Orchestrator(sc, defense_cfg=DefenseConfig.preset(defense)).run())
    return runs


def real_framework() -> dict[str, list[dict]]:
    """Return two tables: aggregate per (framework, backend, defense) with the
    write->survive->retrieve->effect funnel, and a per-attack-type breakdown so
    backend and attack type are no longer confounded (reviewer §5.4)."""
    agg, by_attack = [], []
    for framework, backend in REAL_STORES:
        for defense in REAL_DEFENSES:
            runs = _run_matrix_backend(backend, defense)
            m = metrics(runs)
            stages = _nested(runs)   # monotone write..effect
            agg.append({
                "framework": framework, "backend": backend.split("_", 1)[1],
                "defense": defense, "asr": m["asr"],
                "write": stages["write"], "survive": stages["survive"],
                "retrieve": stages["retrieve"], "effect": stages["effect"],
                "authority_amplification": m["authority_amplification_rate"],
                "lineage": m["state_lineage_recovery"],
            })
            # per-attack ASR (only need the no-defense arm to compare backends)
            if defense == "none":
                by = {}
                for r in runs:
                    by.setdefault(r.attack_type, []).append(r.success)
                for at in sorted(by):
                    vals = by[at]
                    by_attack.append({
                        "framework": framework, "backend": backend.split("_", 1)[1],
                        "attack_type": at,
                        "asr": sum(vals) / len(vals),
                    })
    return {"backend_matrix": agg, "backend_by_attack": by_attack}


# --------------------------------------------------------------------------- #
# §5.4/D  Long-horizon dormancy: does a persisted attack survive 10/50/100 filler
# turns on a real backend? Measures the write..effect funnel as a function of the
# dormancy length, isolating memory erosion (deterministic planner).
# --------------------------------------------------------------------------- #
from .scenario import build_scenario  # noqa: E402  (kept local to real-fw block)
from .schema import AttackType  # noqa: E402

DORMANCY_N = [10, 50, 100]
DORMANCY_JOBS = [  # (workload, attack, backend)
    ("personal_secretary", AttackType.A2_EVENT_CONDITIONAL, "lc_vector"),
    ("rag_support",        AttackType.A2_EVENT_CONDITIONAL, "lc_vector"),
    ("personal_secretary", AttackType.A3_SPLIT_STATE,       "lc_vector"),
    ("personal_secretary", AttackType.A4_SUMMARY_RECONSTRUCT, "lc_summary"),
]


def dormancy_sweep() -> list[dict]:
    rows = []
    for wl, at, backend in DORMANCY_JOBS:
        for N in DORMANCY_N:
            sc = build_scenario(wl, at, "medium", dormancy=N)
            sc.memory = backend
            r = Orchestrator(sc, defense_cfg=DefenseConfig.preset("none")).run()
            rows.append({
                "backend": backend.split("_", 1)[1], "workload": wl,
                "attack_type": at.value, "dormancy": N,
                "survive": int(r.attack_in_store_at_probe),
                "retrieve": int(r.persisted),
                "full_persist": int(r.full_persisted),
                "effect": int(r.success),
            })
    return rows


# --------------------------------------------------------------------------- #
# §5.4/D  Cross-user / cross-tenant: does an attack authored in tenant A reach a
# probe issued by tenant B, and does tenant-scoped memory stop it? Measured at
# the retrieval gate on a real vector store (retrieved attack => would fire under
# no defense). Contrast with cross-session (A6), which leaks within one user
# because memory is session-global.
# --------------------------------------------------------------------------- #
def cross_user() -> list[dict]:
    from .attacks.generator import AttackGenerator
    from .memory.store import make_adapter
    from .scenario import _WL, build_scenario

    gen = AttackGenerator()
    rows = []
    for wl in ["personal_secretary", "rag_support", "coding_support", "soc_support"]:
        spec = build_scenario(wl, AttackType.A2_EVENT_CONDITIONAL, "medium").attack
        probe_q = _WL[wl]["probe"]["query"]
        adapter = make_adapter("lc_vector")
        # attacker writes in tenant A; tenant B fills memory with benign notes
        for st in gen.build(spec, now=0, session_id="userA"):
            adapter.add(st)
        for i in range(5):
            from .schema import Provenance, State, StateType, TrustLevel
            adapter.add(State(content=f"userB benign note {i} about {spec.topic}",
                              state_type=StateType.CONVERSATION, session_id="userB",
                              provenance=Provenance(author="user", source="conversation"),
                              trust_level=TrustLevel.MEDIUM, created_at=i,
                              tags=[spec.topic.split()[-1]]))
        # tenant B probes; shared store (no isolation) vs tenant-scoped retrieval
        shared = adapter.search(probe_q, top_k=5, now=100)
        scoped = adapter.search(probe_q, top_k=5, now=100,
                                filter_fn=lambda s: s.session_id == "userB")
        rows.append({
            "workload": wl,
            "cross_user_shared_leak": int(any(s.attack_marker for s, _ in shared)),
            "cross_user_scoped_leak": int(any(s.attack_marker for s, _ in scoped)),
        })
    return rows


# --------------------------------------------------------------------------- #
# §5.9  Oracle-integrity adaptive attack: instead of mutating the payload, the
# attacker forges the provenance the defense reads (trusted-tool echo). This
# defeats provenance-dependent checks with a perfect oracle; the least-privilege
# stage (keyed on user intent, not provenance) is the independent backstop.
# --------------------------------------------------------------------------- #
def oracle_attack_sweep() -> list[dict]:
    rows = []
    for oa in ["", "trusted_echo"]:
        runs = []
        for sc in all_scenarios(defense="none"):
            sc.defense = "minimal_defense"
            sc.attack.oracle_attack = oa
            runs.append(Orchestrator(
                sc, defense_cfg=DefenseConfig.preset("minimal_defense")).run())
        m = metrics(runs)
        caught_lp = sum(1 for r in runs if r.blocked_by == "least_privilege")
        caught_prov = sum(1 for r in runs if r.blocked_by == "provenance_revalidation")
        rows.append({
            "oracle_attack": oa or "none",
            "asr_vs_minimal": m["asr"],
            "authority_amplification": m["authority_amplification_rate"],
            "lineage_recovery": m["state_lineage_recovery"],
            "caught_by_least_privilege": caught_lp / len(runs),
            "caught_by_provenance": caught_prov / len(runs),
        })
    return rows


def run_all() -> dict[str, list[dict]]:
    return {
        "leave_one_out": leave_one_out(),
        "non_oracle_provenance": non_oracle_provenance(),
        "attribution_noise": attribution_noise(),
        "safety_utility_pareto": safety_utility_pareto(),
        "funnel": funnel(),
    }
