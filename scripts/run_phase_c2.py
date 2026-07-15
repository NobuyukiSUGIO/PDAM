"""Phase C2: harden the one novel end-to-end claim — a real LLM planner
reconstructs summary/split payloads that the deterministic planner misses, so
the deterministic harness UNDER-states the summary-backend risk. Expand beyond
Phase C's n=3 to 2 models x {A3 split(vector), A4 summary} x 4 workloads x 3
difficulties x {none, minimal}. Compares real-LLM ASR to the deterministic
Phase-B baseline (summary A4 = 0.75, split A3 = 1.00)."""
import csv, os, time
from collections import defaultdict
from pdam.scenario import build_scenario, WORKLOADS
from pdam.schema import AttackType
from pdam.orchestrator import Orchestrator
from pdam.llm import LLMPlanner, LMStudioClient

MODELS = ["meta-llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3"]
ATTACKS = [(AttackType.A3_SPLIT_STATE, "lc_vector"),
           (AttackType.A4_SUMMARY_RECONSTRUCT, "lc_summary")]
OUT = "results/real_framework"
rows = []
t0 = time.time()
for model in MODELS:
    client = LMStudioClient(model=model, base_url="http://localhost:1234/v1",
                            temperature=0.0, max_tokens=512, hard_cap=512)
    if not client.ping():
        print(f"SKIP {model}: unreachable"); continue
    for at, backend in ATTACKS:
        for wl in WORKLOADS:
            for diff in ["easy", "medium", "hard"]:
                for defense in ["none", "minimal_defense"]:
                    sc = build_scenario(wl, at, diff, defense=defense, actionable=True)
                    sc.memory = backend
                    r = Orchestrator(sc, LLMPlanner(client)).run()
                    rows.append({"model": model, "attack_type": at.value,
                                 "backend": backend.split("_",1)[1], "workload": wl,
                                 "difficulty": diff, "defense": defense,
                                 "success": int(r.success)})
        print(f"[{model}] {at.value} done ({time.time()-t0:.0f}s, {client.calls} calls)",
              flush=True)

with open(os.path.join(OUT, "phase_c2_runs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# aggregate: real-LLM ASR per (attack, defense=none), vs deterministic baseline
det = {"A3": 1.00, "A4": 0.75}   # Phase-B deterministic lc_* ASR (none)
agg = defaultdict(list)
for r in rows:
    if r["defense"] == "none":
        agg[(r["model"], r["attack_type"])].append(r["success"])
print("\n=== real-LLM end-to-end ASR (none) vs deterministic baseline ===")
summ = []
for (model, at) in sorted(agg):
    v = agg[(model, at)]; asr = sum(v)/len(v)
    summ.append({"model": model, "attack_type": at, "real_llm_asr": round(asr,4),
                 "deterministic_asr": det[at], "n": len(v)})
    print(f"  {model:30s} {at}: real-LLM={asr:.3f}  deterministic={det[at]:.2f}  (n={len(v)})")
with open(os.path.join(OUT, "phase_c2_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
print(f"\nDONE {len(rows)} runs in {(time.time()-t0)/60:.1f} min")
