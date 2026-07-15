# PDAM — Persistent-State and Delayed-Activation Attacks on Memory-Augmented LLM Agents

A reproducible **experimental testbed** implementing the research design
*"Persistent-State and Delayed-Activation Attacks on Memory-Augmented LLM
Agents"* (Version 1.0). It realizes the module structure M1–M6 of the design
(§7 Experimental System Design, §11 Implementation Plan) using **only the
Python standard library — no external dependencies** — so `python3 -m pdam`
runs the full attack / defense / evaluation pipeline out of the box.

> The testbed is a **fully closed simulation** (design §12: reproducibility,
> ethics and safety). It never contacts any real service; all tools are
> in-memory mocks and all data is synthetic. Attack payloads are kept to the
> minimum needed to verify the causal structure — no directly weaponizable
> high-capability payloads are included.

---

## What is implemented

The central claim of the design (§14.3) —

> "The persistent memory of an LLM agent is not mere data storage but an
> *execution state* that alters future control flow and privileged actions.
> Save-time content inspection alone is therefore insufficient; the state's
> **provenance**, the **synthesis** at retrieval time, the **activation
> condition**, and the **execution-time privileges** must be evaluated and
> controlled jointly."

— is reproduced and measured in a runnable form.

| Design module | Implementation | Contents |
|---|---|---|
| M1 Scenario Engine | `scenario.py`, `orchestrator.py` | Benign/attack scenario definitions (JSON/YAML) and a runner. Generates the 4×8×3 matrix of Appendix A |
| M2 Memory Adapter | `memory/` | Vector / summary / KV memory backends behind one API |
| M3 Attack Generator | `attacks/generator.py` | Attack taxonomy A1–A8; fragmentation, delay, summarization, self-re-save |
| M4 Tool Sandbox | `tools/sandbox.py` | Mock email / file / code / ticket / SOC / log tools |
| M5 Policy Monitor | `policy.py`, `broker.py` | The six §10 defenses + the minimal 3-stage defense; mechanical judgement (§9.2) |
| M6 Evaluator | `evaluator.py` | All §9.1 metrics; Fisher exact test, Holm correction, odds ratio, 95% CIs |

The system components of §7.1 also map 1:1: Agent Orchestrator
(`orchestrator.py`), Memory Store (`memory/store.py`), Retriever/RAG
(`retriever.py`), LLM Planner (`planner.py`), Tool Broker (`broker.py`),
Logger/Monitor (`logging_.py` + `policy.py`). See `DESIGN_MAP.md` for the full
section-by-section mapping.

---

## Quick start

```bash
# No installation required (Python 3.10+ only)
python3 -m pdam list-attacks          # attack taxonomy A1–A8
python3 -m pdam list-defenses         # defense presets

# Generate the 96 scenarios of Appendix A
python3 -m pdam gen-scenarios scenarios/

# Run one scenario (with an event-log trace)
python3 -m pdam run scenarios/personal_secretary_A3_hard.json \
        --defense minimal_defense --trace

# Run all scenarios × all defenses; write the summary tables and CSV/JSON
python3 -m pdam batch --outdir results

# Ablation (by difficulty / memory / workload + §6.3 split-state variables)
python3 -m pdam ablate

# Repetitions (§8.2; useful for a nondeterministic LLM planner)
python3 -m pdam batch --repeats 5

# Adaptive attack (§6.2 step 7: observe the blocking stage, mutate the
# payload/trigger, and retry)
python3 -m pdam adapt --workload personal_secretary --difficulty easy

# Real-LLM planner (any OpenAI-compatible local model, e.g. LM Studio)
python3 -m pdam llm-eval --model mistral-7b-instruct-v0.3

# End-to-end demo (attack-lifecycle visualization + statistics)
python3 run_demo.py

# Tests
python3 -m unittest discover -s tests
# or:  make test / make demo / make batch
```

---

## Attack model (§6.1)

One agent step follows the design's equations exactly:

```
C_t = R(x_t, M_t)      retrieve   Retriever.retrieve
a_t = P(x_t, C_t)      plan / state synthesis   Planner.plan
y_t = E(a_t)           execute    ToolBroker.execute
M_{t+1} = U(...)       state update   Orchestrator._update_memory
```

The attacker injects a state `m*` at time `t0` and, once the activation
predicate `g(x_t1, C_t1, s_t1)=1` holds at `t1 > t0`, causes the agent to
select the target action `a*`. Crucially, `m*` alone is designed **not** to be
judged a policy violation (via fragmentation, natural phrasing, summarization,
or trust escalation).

### Attack taxonomy A1–A8 (§5.1)

| ID | Type | Activation condition | Implementation note |
|---|---|---|---|
| A1 | time-delayed | elapsed time `after` / absolute tick `at` | `trigger={"type":"time"}` |
| A2 | event-conditional | keyword match on the target task | `trigger={"type":"event"}` |
| A3 | split-state | reconstructed when all fragments are co-retrieved | k fragments; `Planner._reconstruct` reassembles |
| A4 | summary-reconstruction | the instruction is consolidated during summarization | `SummaryMemoryAdapter.compact` stamps `reconstruct` in the provenance |
| A5 | trust-escalation | fires after repeated use raises trust to *high* | `Orchestrator._escalate` |
| A6 | cross-session | fires in a session other than the injection one | `trigger={"type":"cross_session"}` |
| A7 | self-re-save | re-saves the outcome after firing to persist | `Orchestrator._self_resave` |
| A8 | trace-suppression | tampers with the audit log after firing | `Orchestrator._suppress_trace` |

---

## Defenses (§10)

Selected with `--defense`. Individual baselines plus the three-stage "minimal
defense" of §10.2 are provided.

| Preset | Stage | Contents |
|---|---|---|
| `none` | — | No defense (baseline) |
| `prompt_only` | inference | System prompt only (weakest) |
| `content_filter` | save-time | Content inspection. **Catches an explicit single-payload instruction, but is bypassed by split (A3), summarization (A4), and natural phrasing** |
| `embedding_filter` | retrieval | Excludes anomalous records from retrieval |
| `llm_judge` | pre/post exec | Inspects tool arguments for danger |
| `least_privilege` | execution | Restricts high-privilege calls not tied to user intent (**catches ops that differ from the authorized one; misses reuse of an authorized op**) |
| `provenance` | retrieval + exec | Provenance-aware revalidation. **Blocks low-trust / external / derived (summarized or promoted) states from driving privileged actions**; also limits co-retrieval of low-trust fragments |
| `minimal_defense` | save+retrieval+exec | The §10.2 proposal (content_filter + provenance + least_privilege + fragment co-retrieval limit) |
| `full` | all stages | All defenses |

---

## Metrics (§9.1) and statistics (§9.2)

`Evaluator` computes: Attack Success Rate, Delayed ASR, Persistence Survival
Rate, Trigger Precision, Stealth Score, Unauthorized Action Rate, Data
Exfiltration Rate, Benign Task Success, False Positive Rate, and State Lineage
Recovery, an **authority-amplification rate**, and a nested **lifecycle funnel**.
Because each scenario is run under both arms, defense comparisons use the
**paired exact McNemar test** (primary) and the **Fisher exact test** (reference);
multiple comparisons use the **Holm correction**, and effect sizes are reported
as the **odds ratio, absolute risk difference, and 95% confidence intervals**
(Wilson for proportions; all in the standard library, no scipy).

### Representative results (96 scenarios × each defense, fixed seed)

```
defense          asr    delayed  persist  trig.P  stealth  unauth  exfil  benign  FPR   lineage
none             1.000  1.000    1.000    0.906   1.000    1.000   0.500  1.000   0.000 0.500
content_filter   0.760  0.760    0.771    0.667   0.760    0.760   0.354  1.000   0.000 0.562
least_privilege  0.500  0.500    0.990    0.917   0.500    0.500   0.250  1.000   0.000 0.938
provenance       0.000  0.000    0.490    0.406   0.000    0.000   0.000  1.000   0.000 1.000
minimal_defense  0.000  0.000    0.260    0.188   0.000    0.000   0.000  1.000   0.000 1.000
```

This table directly exhibits the design's hypotheses:

- **H1** (delayed activation is harder to detect): `content_filter` misses the
  natural-phrasing / delayed variants.
- **H2** (split-state bypasses content filtering): A3 passes `content_filter`
  even at the Easy level.
- **H3** (summarization can strengthen the attack): A4 launders trust through
  summarization, bypassing `content_filter` and `least_privilege` — only
  `provenance` stops it.
- **H4** (provenance-aware revalidation + execution-time least privilege): the
  `minimal_defense` reaches no observed attack success (Wilson 95% upper bound
  0.038) while keeping Benign Task Success = 1.0 on the **internal** benign tasks.
  ⚠️ This is not "no utility cost": on legitimate tasks that must act on
  **external / derived** memory, hard provenance blocking blocks *all* of them
  (external-task FPR = 1.0). The `confirm_external` variant escalates such actions
  to user confirmation, recovering full legitimate-task success at a bounded
  confirmation cost while keeping attack ASR = 0. See `pdam robustness`.
- **State Lineage Recovery**: 0.500 under `none` (A8 trace suppression and A4
  summarization destroy the lineage); 1.0 under `provenance` / `minimal_defense`.

> **Note (statistics).** A tail-tolerance bug in `fisher_exact` (an absolute
> `1e-12` floor) understated near-perfect-separation p-values by ~40 orders of
> magnitude; it is fixed (relative tolerance, regression-tested). Because each
> scenario is run under both arms, the **paired McNemar exact test** is now the
> primary test (Fisher is reported for reference), and ASR = 0 is reported as
> "no observed success" with a Wilson upper bound rather than a literal zero.

---

## Filled-in design details

Beyond the core, the following design items are also implemented:

- **§6.2 step 7 — adaptive attack** (`attacks/adaptive.py`, `pdam adapt`): the
  attacker observes which stage blocked it and mutates the payload/trigger,
  retrying up to a budget. Weak single-stage defenses are adapted around;
  the minimal defense resists.
- **§6.3 split-state independent variables** (`AttackSpec.semantic_distance /
  save_time_spread / ttl`, swept in `pdam ablate`): distractor padding, staggered
  injection, and state expiry all lower co-retrieval / reconstruction.
- **§6.5 self-re-save axes**: re-save generations, an influence-decay rate, and
  a **post-deletion survival** check (does the attack still fire after the
  attacker's original states are deleted?).
- **§9.2 95% confidence intervals**: Wilson (proportion), Wald (risk
  difference), log-OR (odds ratio).
- **§8.2 repetitions**: `pdam batch --repeats N`.

---

## Reviewer-driven robustness experiments (`pdam robustness`)

`python -m pdam robustness` runs four experiments that stress the defense
evaluation beyond the idealized benchmark (byte-reproducible; CSVs written to
`results/robustness/`):

- **Leave-one-component-out** of the 3-stage minimal defense. On the *oracle*
  benchmark, removing provenance revalidation is the only single removal that
  readmits attacks (ASR 0.333); the other stages are individually redundant.
- **Non-oracle provenance.** The defense sees an *estimated* provenance
  (`prov_trust_noise`, `prov_dropout`, seeded via SHA-1). Under provenance
  dropout, provenance-only degrades to ASR 0.250 while the 3-stage defense holds
  at 0.083 — a 3× reduction that justifies the redundant stages once provenance
  is imperfect. This separates the **oracle ceiling** from realistic performance.
- **Safety/utility Pareto** over a legitimate-task suite (`pdam/utility.py`) in
  which 4 of 7 tasks must act on external / derived memory. Shows the external-FPR
  cost of hard provenance blocking and how `confirm_external` resolves it.
- **Lifecycle funnel** (write → survive → retrieve → synthesize → dispatch →
  effect), showing the three defenses act at three different stages.

New defense presets added for this analysis: `memory_disabled`, `block_external`,
`confirm_all`, `confirm_external`, and `minimal_minus_{content_filter,
provenance, least_privilege, fragment_limit}`.

---

## Real-LLM planner (§7.2)

The default `RuleBasedPlanner` is a deterministic, reproducible stand-in for a
susceptible LLM. To evaluate a **real model**, implement `Planner.plan` (or use
the provided `pdam/llm.py`) so it parses the natural-language `State.content`
instead of the structured directive. `LLMPlanner` + `LMStudioClient` drive any
**OpenAI-compatible local server** (e.g. LM Studio):

```bash
python3 -m pdam llm-eval --model mistral-7b-instruct-v0.3 \
        --base-url http://localhost:1234/v1 \
        --workloads personal_secretary --defenses none,minimal_defense
```

With a real LLM, susceptibility, state synthesis (reassembling A3 fragments,
reading A4 summaries) and delayed/conditional activation become *emergent*
behaviours of the model, so the testbed measures what an actual agent would do.
The client tolerates template quirks (folds a `system` role into the first
`user` turn for templates that reject it) and, for reasoning models whose
chain-of-thought exhausts the budget, retries once with a larger token cap.

### Empirical multi-model results

Four open-weight models across three families and two size classes were run over
all 4 workloads × 8 attacks × {none, minimal_defense} × 3 repetitions
(`scripts/run_llm_eval.sh`; served locally by LM Studio; full data under
[`results/llm/`](results/llm/RESULTS.md)). Every emitted tool call is judged
mechanically (data-flow / policy), not by keywords.

| model | no-defense ASR | minimal_defense ASR | benign task success | Fisher exact |
|---|--:|--:|--:|--:|
| meta-llama-3.1-8b-instruct | 0.948 | **0.000** | 0.946 | p = 1.3e-12 |
| mistral-7b-instruct-v0.3 | 1.000 | **0.000** | 0.925 | p = 4.5e-13 |
| mistral-small-24b-instruct-2501 | 1.000 | **0.000** | 0.953 | p = 4.5e-13 |
| gemma-2-27b-it | 1.000 | **0.000** | 0.933 | p = 4.5e-13 |

Across all four models the persistent, delayed-activation injections succeed at
**95–100%** with no defense, and the §10.2 three-stage minimal defense drives ASR
to **0** while keeping benign-task completion at **92–95%** (false-positive rate
5–8%) — an empirical confirmation of H4 on real models (risk difference ≈ −1.0,
p < 1.3e-12 for every model). This reproduces, on actual LLMs, the pattern the
deterministic rule-based planner shows.

---

## Directory layout

```
pdam/
  schema.py        Appendix B data schema (State/Provenance/Directive/ToolCall/LogEvent)
  embedding.py     dependency-free bag-of-words embedding + cosine similarity
  memory/          M2: base / vector / summary / kv / store
  attacks/         M3: attack generator (A1–A8) + adaptive attacker
  tools/           M4: tool sandbox
  policy.py        M5: 6 defenses + minimal defense + mechanical judgement
  broker.py        Tool Broker (mediation / authorization)
  retriever.py     Retriever/RAG (retrieval-time defenses)
  planner.py       LLM Planner (state synthesis / activation) — rule-based
  llm.py           real-LLM planner via an OpenAI-compatible server
  orchestrator.py  Agent Orchestrator (the R/P/E/U loop)
  logging_.py      event log
  evaluator.py     M6: metrics + statistics
  cli.py           command line
scenarios/         generated scenarios (JSON)
tests/             unit + integration tests (33)
run_demo.py        end-to-end demo
DESIGN_MAP.md      mapping of each design section → code
```

---

## License / positioning

A defense-research testbed for research and education (design §16.2: "an
experimental environment and teaching material usable for safe agent-development
education"). Following responsible-disclosure principles, it is not intended to
help exploit specific vulnerabilities in production agents.
