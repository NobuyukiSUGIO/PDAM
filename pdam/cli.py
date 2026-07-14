"""Command-line interface for the PDAM testbed.

    python -m pdam list-attacks
    python -m pdam list-defenses
    python -m pdam gen-scenarios scenarios/
    python -m pdam run scenarios/personal_secretary_A3_hard.json --defense minimal_defense
    python -m pdam batch --defenses none,least_privilege,provenance,minimal_defense,full
    python -m pdam ablate
"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .evaluator import (
    aggregate,
    compare,
    count_success,
    format_table,
    holm,
    metrics,
    rows_to_csv,
    to_json,
)
from .orchestrator import Orchestrator
from .policy import PRESET_NAMES
from .scenario import (
    DIFFICULTIES,
    WORKLOADS,
    Scenario,
    all_scenarios,
    build_scenario,
    load_scenario,
)
from .schema import ATTACK_NAMES, AttackType


def _make_llm_planner(args):
    from .llm import LLMPlanner, LMStudioClient
    client = LMStudioClient(
        model=args.model, base_url=args.base_url,
        max_tokens=args.max_tokens, hard_cap=max(args.max_tokens, args.hard_cap),
    )
    guardrail = getattr(args, "_guardrail", False)
    return LLMPlanner(client, guardrail=guardrail), client


def _run_all(scenarios, quiet=False, repeats=1):
    results = []
    for sc in scenarios:
        for rep in range(repeats):
            res = Orchestrator(sc).run()
            results.append(res)
        if not quiet:
            res = results[-1]
            flag = "SUCCESS" if res.success else (
                f"blocked@{res.blocked_stage}:{res.blocked_by}"
                if res.blocked_by else "no-fire")
            reps = f" x{repeats}" if repeats > 1 else ""
            print(f"  {sc.name:<42} {sc.defense:<16} -> {flag}{reps}")
    return results


# --------------------------------------------------------------------------- #
def cmd_list_attacks(_):
    print("Attack taxonomy (§5.1):")
    for at in AttackType:
        print(f"  {at.value}  {ATTACK_NAMES[at]}")


def cmd_list_defenses(_):
    print("Defense presets (§10):")
    for name in PRESET_NAMES:
        print(f"  {name}")


def cmd_gen_scenarios(args):
    os.makedirs(args.outdir, exist_ok=True)
    scenarios = all_scenarios(defense=args.defense)
    for sc in scenarios:
        sc.save(os.path.join(args.outdir, f"{sc.name}.json"))
    print(f"wrote {len(scenarios)} scenarios to {args.outdir}/ "
          f"({len(WORKLOADS)} workloads × {len(list(AttackType))} attacks "
          f"× {len(DIFFICULTIES)} difficulties)")


def cmd_run(args):
    sc = load_scenario(args.scenario)
    if args.defense:
        sc.defense = args.defense
    if args.memory:
        sc.memory = args.memory
    planner = None
    if args.planner == "llm":
        sc.attack.actionable = True
        args._guardrail = sc.defense in ("prompt_only", "full")
        planner, _ = _make_llm_planner(args)
    res = Orchestrator(sc, planner).run()
    print(f"scenario : {sc.name}")
    print(f"defense  : {sc.defense}   memory: {sc.memory}")
    print(f"fired    : {res.fired}  executed: {res.executed}  success: {res.success}")
    print(f"delayed  : {res.delayed}  persisted: {res.persisted}  "
          f"lineage_recoverable: {res.lineage_recoverable}")
    print(f"violation: {res.violation}")
    if res.blocked_by:
        print(f"blocked  : {res.blocked_stage} by {res.blocked_by}")
    print(f"benign   : {res.benign_success}/{res.benign_total} ok")
    if args.trace:
        print("\n--- event log ---")
        for ev in res.events.events:
            print(f"  t{ev.tick:>2} {ev.event_type:<15} {ev.payload}")
    if args.logdir:
        os.makedirs(args.logdir, exist_ok=True)
        res.events.save(os.path.join(args.logdir, f"{sc.name}.jsonl"))
        print(f"\nlog saved to {args.logdir}/{sc.name}.jsonl")


def cmd_batch(args):
    defenses = args.defenses.split(",")
    all_results = []
    for defense in defenses:
        scenarios = all_scenarios(defense=defense)
        if args.difficulty:
            scenarios = [s for s in scenarios if s.difficulty == args.difficulty]
        print(f"\n=== defense: {defense}  ({len(scenarios)} scenarios"
              f"{f' x{args.repeats} repeats' if args.repeats > 1 else ''}) ===")
        all_results += _run_all(scenarios, quiet=args.quiet, repeats=args.repeats)

    print("\n" + "=" * 70)
    print("Aggregate by defense (§9.1 metrics):")
    rows = aggregate(all_results, by=["defense"])
    print(format_table(rows, by=["defense"]))

    print("\nAggregate by attack_type × defense:")
    rows2 = aggregate(all_results, by=["attack_type", "defense"])
    print(format_table(rows2, by=["attack_type", "defense"]))

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "by_defense.csv"), "w") as fh:
            fh.write(rows_to_csv(rows))
        with open(os.path.join(args.outdir, "by_attack_defense.csv"), "w") as fh:
            fh.write(rows_to_csv(rows2))
        with open(os.path.join(args.outdir, "runs.json"), "w") as fh:
            fh.write(to_json(all_results))
        print(f"\nwrote CSV + runs.json to {args.outdir}/")

    _defense_stats(all_results, baseline="none")


def _defense_stats(all_results, baseline="none"):
    """Fisher exact ASR comparison of each defense vs the no-defense baseline."""
    by_def: dict[str, list] = {}
    for r in all_results:
        by_def.setdefault(r.defense, []).append(r)
    if baseline not in by_def:
        return
    b_succ, b_n = count_success(by_def[baseline])
    comps = []
    for defense, runs in by_def.items():
        if defense == baseline:
            continue
        a_succ, a_n = count_success(runs)
        comps.append(compare(f"{defense} vs {baseline}", a_succ, a_n, b_succ, b_n))
    holm(comps)
    print("\nASR vs no-defense baseline (Fisher exact, Holm-adjusted, 95% CI):")
    print(f"  baseline ASR = {b_succ}/{b_n} = {b_succ / b_n:.3f}" if b_n else "")
    for c in sorted(comps, key=lambda x: x.p_adjusted):
        sig = "*" if c.p_adjusted < 0.05 else " "
        print(f" {sig} {c.label:<28} ASR {c.a_rate:.3f} "
              f"[{c.a_ci[0]:.3f},{c.a_ci[1]:.3f}]  "
              f"OR={c.odds_ratio:.3f} [{c.or_ci[0]:.3f},{c.or_ci[1]:.3f}]  "
              f"RD={c.risk_diff:+.3f} [{c.rd_ci[0]:+.3f},{c.rd_ci[1]:+.3f}]  "
              f"p_adj={c.p_adjusted:.3g}")


def cmd_llm_eval(args):
    """Evaluate a curated subset with a real local LLM as the planner, and
    compare the model's behaviour against the deterministic rule-based planner."""
    import time

    from .llm import LLMPlanner, LMStudioClient

    workloads = args.workloads.split(",") if args.workloads else ["personal_secretary"]
    attacks = ([AttackType(a) for a in args.attacks.split(",")]
               if args.attacks else list(AttackType))
    defenses = args.defenses.split(",")
    client = LMStudioClient(model=args.model, base_url=args.base_url,
                            max_tokens=args.max_tokens,
                            hard_cap=max(args.max_tokens, args.hard_cap))
    print(f"model: {args.model} @ {args.base_url}")
    print(f"checking server ... ", end="", flush=True)
    if not client.ping():
        print("UNREACHABLE. Is LM Studio serving and a model loaded?")
        return
    print("ok\n")

    results = []
    t0 = time.time()
    for defense in defenses:
        guardrail = defense in ("prompt_only", "full")
        for wl in workloads:
            for at in attacks:
                sc = build_scenario(wl, at, args.difficulty, defense=defense,
                                    actionable=True)
                planner = LLMPlanner(client, guardrail=guardrail)
                res = Orchestrator(sc, planner).run()
                results.append(res)
                verdict = ("SUCCESS" if res.success
                           else (f"blocked@{res.blocked_stage}:{res.blocked_by}"
                                 if res.blocked_by else "no-fire"))
                print(f"  {sc.name:<40} {defense:<16} -> {verdict:<28} "
                      f"[{client.calls} calls, {time.time()-t0:.0f}s]")

    print("\n" + "=" * 70)
    print(f"Real-LLM evaluation ({args.model}) — {len(results)} runs, "
          f"{client.calls} LLM calls, {time.time()-t0:.0f}s")
    print(format_table(aggregate(results, by=["defense"]), by=["defense"]))
    print("\nBy attack_type × defense:")
    print(format_table(aggregate(results, by=["attack_type", "defense"]),
                       by=["attack_type", "defense"]))

    if "none" in defenses and len(defenses) > 1:
        _defense_stats(results, baseline="none")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "llm_runs.json"), "w") as fh:
            fh.write(to_json(results))
        with open(os.path.join(args.outdir, "llm_by_defense.csv"), "w") as fh:
            fh.write(rows_to_csv(aggregate(results, by=["defense"])))
        for r in results:
            r.events.save(os.path.join(args.outdir, f"llm_{r.scenario}_{r.defense}.jsonl"))
        print(f"\nwrote LLM results to {args.outdir}/")


def cmd_adapt(args):
    """Adaptive attack (§6.2 手順7): retry with payload/trigger mutation until
    the attack succeeds or the attempt budget is spent, per defense."""
    from .attacks.adaptive import run_adaptive

    attacks = ([AttackType(a) for a in args.attacks.split(",")]
               if args.attacks else list(AttackType))
    defenses = args.defenses.split(",")
    print(f"Adaptive attacker: workload={args.workload} difficulty={args.difficulty} "
          f"max_attempts={args.max_attempts}\n")
    summary = {}
    for defense in defenses:
        succ = 0
        attempts_to_success = []
        for at in attacks:
            sc = build_scenario(args.workload, at, args.difficulty, defense=defense)
            r = run_adaptive(sc, max_attempts=args.max_attempts)
            if r.success:
                succ += 1
                attempts_to_success.append(r.attempts)
            path = " -> ".join(
                f"a{h['attempt']}[{h['blocked_by'] or ('fired' if h['fired'] else 'no-fire')}]"
                for h in r.history)
            print(f"  {at.value} {defense:<16} "
                  f"{'SUCCESS in ' + str(r.attempts) if r.success else 'RESISTED'}"
                  f"  | {path}")
        mean_att = (sum(attempts_to_success) / len(attempts_to_success)
                    if attempts_to_success else float('nan'))
        summary[defense] = (succ, len(attacks), mean_att)
        print()
    print("=" * 60)
    print(f"{'defense':<18}{'adaptive ASR':>14}{'mean attempts-to-success':>28}")
    for d, (s, n, m) in summary.items():
        print(f"{d:<18}{s}/{n} = {s/n:>6.3f}   {m:>22.2f}")


def cmd_ablate(args):
    """Ablation over memory type, difficulty and defense (§9.3)."""
    results = []
    for defense in ["none", "minimal_defense"]:
        for sc in all_scenarios(defense=defense):
            results.append(Orchestrator(sc).run())
    print("Ablation — ASR by difficulty × defense:")
    print(format_table(aggregate(results, by=["difficulty", "defense"]),
                       by=["difficulty", "defense"]))
    print("\nAblation — ASR by memory × defense:")
    print(format_table(aggregate(results, by=["memory", "defense"]),
                       by=["memory", "defense"]))
    print("\nAblation — ASR by workload × defense:")
    print(format_table(aggregate(results, by=["workload", "defense"]),
                       by=["workload", "defense"]))

    # §6.3 split-state independent variables (k, semantic distance, save-time
    # spread, ttl) — sweep them on A3, no defense, and report ASR / persistence.
    print("\nAblation — A3 split-state variables (no defense):")
    print(f"  {'k':>2} {'sem.dist':>9} {'save.spread':>12} {'ttl':>4} "
          f"{'asr':>6} {'persist':>8}")
    from .scenario import WORKLOADS
    for k, sd, sp, ttl in [(2, 0.0, 0, 0), (4, 0.0, 0, 0), (4, 0.0, 3, 4),
                           (4, 0.9, 0, 0), (4, 0.9, 3, 4)]:
        runs = []
        for wl in WORKLOADS:
            sc = build_scenario(wl, AttackType.A3_SPLIT_STATE, "medium",
                                defense="none", semantic_distance=sd,
                                save_time_spread=sp, ttl=ttl)
            sc.attack.fragments = k
            runs.append(Orchestrator(sc).run())
        m = metrics(runs)
        print(f"  {k:>2} {sd:>9.1f} {sp:>12} {ttl:>4} "
              f"{m['asr']:>6.3f} {m['persistence_survival']:>8.3f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdam", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"pdam {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-attacks").set_defaults(func=cmd_list_attacks)
    sub.add_parser("list-defenses").set_defaults(func=cmd_list_defenses)

    g = sub.add_parser("gen-scenarios")
    g.add_argument("outdir", nargs="?", default="scenarios")
    g.add_argument("--defense", default="none")
    g.set_defaults(func=cmd_gen_scenarios)

    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--defense", default="")
    r.add_argument("--memory", default="")
    r.add_argument("--trace", action="store_true")
    r.add_argument("--logdir", default="")
    r.add_argument("--planner", choices=["rule", "llm"], default="rule")
    r.add_argument("--model", default="qwen/qwen3.6-27b")
    r.add_argument("--base-url", dest="base_url", default="http://localhost:1234/v1")
    r.add_argument("--max-tokens", dest="max_tokens", type=int, default=2200)
    r.add_argument("--hard-cap", dest="hard_cap", type=int, default=4200)
    r.set_defaults(func=cmd_run)

    le = sub.add_parser("llm-eval",
                        help="evaluate a subset with a real local LLM planner")
    le.add_argument("--model", default="qwen/qwen3.6-27b")
    le.add_argument("--base-url", dest="base_url", default="http://localhost:1234/v1")
    le.add_argument("--workloads", default="personal_secretary",
                    help="comma-separated workloads")
    le.add_argument("--attacks", default="",
                    help="comma-separated attack ids (A1..A8); default all")
    le.add_argument("--difficulty", default="easy", choices=DIFFICULTIES)
    le.add_argument("--defenses", default="none,minimal_defense")
    le.add_argument("--max-tokens", dest="max_tokens", type=int, default=2200)
    le.add_argument("--hard-cap", dest="hard_cap", type=int, default=4200)
    le.add_argument("--outdir", default="results")
    le.set_defaults(func=cmd_llm_eval)

    b = sub.add_parser("batch")
    b.add_argument("--defenses",
                   default="none,content_filter,least_privilege,provenance,minimal_defense,full")
    b.add_argument("--difficulty", default="", choices=["", *DIFFICULTIES])
    b.add_argument("--outdir", default="results")
    b.add_argument("--quiet", action="store_true")
    b.add_argument("--repeats", type=int, default=1,
                   help="repeat each scenario N times (§8.2; matters for a "
                        "nondeterministic LLM planner)")
    b.set_defaults(func=cmd_batch)

    a = sub.add_parser("ablate")
    a.set_defaults(func=cmd_ablate)

    ad = sub.add_parser("adapt", help="adaptive attack with payload/trigger mutation")
    ad.add_argument("--workload", default="personal_secretary", choices=WORKLOADS)
    ad.add_argument("--attacks", default="", help="comma-separated A1..A8; default all")
    ad.add_argument("--difficulty", default="easy", choices=DIFFICULTIES)
    ad.add_argument("--defenses", default="content_filter,least_privilege,minimal_defense")
    ad.add_argument("--max-attempts", dest="max_attempts", type=int, default=5)
    ad.set_defaults(func=cmd_adapt)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
