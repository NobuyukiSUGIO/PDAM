"""Phase E1: per-stage cost of the defense on real memory (§6.5 cheap/practical).
Measures (a) the defense's marginal latency vs its embedding/summariser cost,
(b) storage at probe, (c) token volume, (d) the real summariser latency that the
agent pays regardless of defense."""
import csv, time, statistics
import tiktoken
from pdam.scenario import build_scenario
from pdam.schema import AttackType
from pdam.orchestrator import Orchestrator
from pdam.policy import DefenseConfig, PolicyMonitor
from pdam.memory.real_langchain import _summarise

enc = tiktoken.get_encoding("cl100k_base")
REPS = 8
rows = []

def run_latency(mem, attack, defense, reps=REPS):
    lat = []
    last = None
    for _ in range(reps):
        sc = build_scenario("personal_secretary", attack, "medium"); sc.memory = mem
        t0 = time.time()
        last = Orchestrator(sc, defense_cfg=DefenseConfig.preset(defense)).run()
        lat.append((time.time()-t0)*1000)
    return statistics.mean(lat), statistics.pstdev(lat), last

# (a) defense marginal latency on real vector (embedding cost is identical in both)
for defense in ["none", "minimal_defense"]:
    m, sd, res = run_latency("lc_vector", AttackType.A2_EVENT_CONDITIONAL, defense)
    # storage at probe
    rows.append({"backend":"vector","defense":defense,"attack":"A2",
                 "latency_ms_mean":f"{m:.1f}","latency_ms_sd":f"{sd:.1f}"})
    print(f"vector A2 {defense:15s} latency={m:.0f}±{sd:.0f}ms")

# (b) per-check microbenchmark: pure-Python defense compute in isolation
mon = PolicyMonitor(DefenseConfig.preset("minimal_defense"))
sc = build_scenario("personal_secretary", AttackType.A2_EVENT_CONDITIONAL, "medium")
from pdam.attacks.generator import AttackGenerator
st = AttackGenerator().build(sc.attack, 0, "s1")[0]
t0=time.time()
for _ in range(10000): mon.screen_state(st)
screen_us = (time.time()-t0)/10000*1e6
print(f"screen_state (save-time filter): {screen_us:.1f} us/call  (pure Python)")

# (c) real summariser latency — the dominant real cost, paid regardless of defense
txt = "\n".join(f"- note {i} about weekly account update and routine sync" for i in range(6))
t0=time.time(); s=_summarise(txt); summ_ms=(time.time()-t0)*1000
summ_out_tok = len(enc.encode(s))
print(f"real summariser: {summ_ms:.0f}ms, {summ_out_tok} output tokens (one compaction)")

# (d) storage/token at probe (real vector, none)
sc = build_scenario("personal_secretary", AttackType.A2_EVENT_CONDITIONAL, "medium"); sc.memory="lc_vector"
orch = Orchestrator(sc, defense_cfg=DefenseConfig.preset("none")); res = orch.run()
states = orch.store.all()
bytes_ = sum(len(s.content) for s in states)
toks = sum(len(enc.encode(s.content)) for s in states)
print(f"storage at end: {len(states)} states, {bytes_} bytes, {toks} tokens")

rows.append({"backend":"-","defense":"screen_state_us","attack":"-",
             "latency_ms_mean":f"{screen_us/1000:.4f}","latency_ms_sd":"0"})
rows.append({"backend":"summary","defense":"summariser_call","attack":"-",
             "latency_ms_mean":f"{summ_ms:.1f}","latency_ms_sd":"0"})
rows.append({"backend":"vector","defense":"storage_states","attack":"-",
             "latency_ms_mean":str(len(states)),"latency_ms_sd":str(toks)})
with open("results/real_framework/cost.csv","w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote cost.csv")
