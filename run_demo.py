#!/usr/bin/env python3
"""End-to-end demonstration of the PDAM testbed.

Runs the full attack lifecycle (Injection -> Persistence -> Retrieval ->
Activation -> Privileged Action -> Trace Suppression, §5) for a few
representative scenarios and prints how each defense stage fares — reproducing
the core claim of §14.3 that save-time content inspection alone is insufficient.

    python3 run_demo.py
"""
from pdam import Orchestrator, build_scenario
from pdam.evaluator import aggregate, compare, format_table, holm, count_success
from pdam.schema import AttackType
from pdam.scenario import all_scenarios

SEP = "=" * 72


def show_lifecycle():
    print(SEP)
    print("1) Attack lifecycle — split-state (A3), Hard, no defense")
    print(SEP)
    sc = build_scenario("personal_secretary", AttackType.A3_SPLIT_STATE, "hard",
                        defense="none")
    res = Orchestrator(sc).run()
    for ev in res.events.events:
        print(f"  t{ev.tick:>2}  {ev.event_type:<15} {ev.payload}")
    print(f"\n  => success={res.success}  violation={res.violation}  "
          f"delayed={res.delayed}  lineage_recoverable={res.lineage_recoverable}")


def show_defense_gradient():
    print("\n" + SEP)
    print("2) Same attack across defenses (save vs retrieval vs execution stage)")
    print(SEP)
    for defense in ["none", "content_filter", "least_privilege",
                    "provenance", "minimal_defense", "full"]:
        sc = build_scenario("personal_secretary", AttackType.A3_SPLIT_STATE,
                            "hard", defense=defense)
        res = Orchestrator(sc).run()
        verdict = ("SUCCESS" if res.success
                   else f"blocked@{res.blocked_stage or 'retrieval'}"
                        f":{res.blocked_by or 'fragment_limit'}")
        print(f"  {defense:<18} -> {verdict}")


def show_matrix():
    print("\n" + SEP)
    print("3) Full matrix (4 workloads × 8 attacks × 3 difficulties) by defense")
    print(SEP)
    results = []
    for defense in ["none", "content_filter", "least_privilege",
                    "provenance", "minimal_defense"]:
        for sc in all_scenarios(defense=defense):
            results.append(Orchestrator(sc).run())
    print(format_table(aggregate(results, by=["defense"]), by=["defense"]))

    print("\nASR vs no-defense baseline (Fisher exact, Holm-adjusted):")
    by_def = {}
    for r in results:
        by_def.setdefault(r.defense, []).append(r)
    b_succ, b_n = count_success(by_def["none"])
    comps = [compare(f"{d} vs none", *count_success(rs), b_succ, b_n)
             for d, rs in by_def.items() if d != "none"]
    holm(comps)
    for c in sorted(comps, key=lambda x: x.p_adjusted):
        sig = "*" if c.p_adjusted < 0.05 else " "
        print(f" {sig} {c.label:<26} ASR={c.a_rate:.3f} OR={c.odds_ratio:.4f} "
              f"RD={c.risk_diff:+.3f} p_adj={c.p_adjusted:.3g}")


if __name__ == "__main__":
    show_lifecycle()
    show_defense_gradient()
    show_matrix()
    print("\nDone. See README.md and DESIGN_MAP.md for how this maps to the design.")
