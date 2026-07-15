"""Phase C: real LLM planner x real framework memory, end-to-end (§5.4/C).
Sweeps difficulty explicitly (easy/medium/hard) -> answers reviewer Q#5.
Records per-run outcomes and aggregates ASR by difficulty and defense."""
import csv, os, time
from pdam.scenario import build_scenario
from pdam.schema import AttackType
from pdam.orchestrator import Orchestrator
from pdam.llm import LLMPlanner, LMStudioClient

MODEL = "meta-llama-3.1-8b-instruct"
OUT = "results/real_framework"
os.makedirs(OUT, exist_ok=True)

client = LMStudioClient(model=MODEL, base_url="http://localhost:1234/v1",
                        temperature=0.0, max_tokens=512, hard_cap=512)
assert client.ping(), "LM Studio unreachable"

# main sweep: each workload's canonical event-conditional attack across difficulty,
# on real LangChain vector memory; plus split (A3, vector) and summary (A4, summary)
# on one workload to cover fragmentation/laundering end-to-end.
JOBS = []
WL = ["personal_secretary", "rag_support", "coding_support", "soc_support"]
for wl in WL:
    for diff in ["easy", "medium", "hard"]:
        for defense in ["none", "minimal_defense"]:
            JOBS.append((wl, AttackType.A2_EVENT_CONDITIONAL, diff, defense, "lc_vector"))
for at, backend in [(AttackType.A3_SPLIT_STATE, "lc_vector"),
                    (AttackType.A4_SUMMARY_RECONSTRUCT, "lc_summary")]:
    for diff in ["easy", "medium", "hard"]:
        for defense in ["none", "minimal_defense"]:
            JOBS.append(("personal_secretary", at, diff, defense, backend))

rows = []
t0 = time.time()
for i, (wl, at, diff, defense, backend) in enumerate(JOBS):
    sc = build_scenario(wl, at, diff, defense=defense, actionable=True)
    sc.memory = backend
    res = Orchestrator(sc, LLMPlanner(client)).run()
    rows.append({
        "model": MODEL, "framework": "langchain", "backend": backend.split("_",1)[1],
        "workload": wl, "attack_type": at.value, "difficulty": diff, "defense": defense,
        "success": int(res.success), "persisted": int(res.persisted),
        "fired_on_probe": int(res.fired_on_probe), "fired_on_noise": int(res.fired_on_noise),
        "authority_amplified": int(res.authority_amplified),
        "violation": res.violation, "blocked_by": res.blocked_by or "",
    })
    print(f"[{i+1}/{len(JOBS)}] {wl[:12]:12s} {at.value} {diff:6s} {defense:15s} {backend:10s}"
          f" -> success={res.success} viol={res.violation} ({time.time()-t0:.0f}s, {client.calls} calls)",
          flush=True)

with open(os.path.join(OUT, "phase_c_runs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# aggregate ASR by difficulty x defense
agg = {}
for r in rows:
    k = (r["difficulty"], r["defense"]); agg.setdefault(k, []).append(r["success"])
with open(os.path.join(OUT, "phase_c_by_difficulty.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["difficulty","defense","asr","n"])
    for (diff, defe) in sorted(agg):
        v = agg[(diff, defe)]; w.writerow([diff, defe, f"{sum(v)/len(v):.4f}", len(v)])
print(f"DONE {len(rows)} runs in {(time.time()-t0)/60:.1f} min, {client.calls} LLM calls")
