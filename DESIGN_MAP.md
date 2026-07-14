# Design document → implementation mapping

How each section of the research design (v1.0) is realized in code.

## 3. Target system and terminology
- Target agent functions (memory write / retrieval-RAG / planning / tool use /
  state update) → the R/P/E/U loop in `orchestrator.py`, plus `memory/`,
  `retriever.py`, `planner.py`, `broker.py`.
- Key terms (persistent state, state poisoning, delayed activation, dormancy
  period, state synthesis, provenance) → `schema.py` (`State`, `Provenance`,
  `Directive`), `planner._reconstruct` (state synthesis), `memory/store.lineage`
  (lineage tracing).

## 4. Threat model
- Attacker capabilities (query the agent as a normal user; place content in
  shared documents/email; no access to internal prompts; black-box observation;
  target another user/session) → `AttackGenerator` (injection of external,
  low-trust states), A6 cross-session.
- Attacker goals (unauthorized action / exfiltration / decision manipulation /
  continued control / trace suppression) → the `PolicyViolation` enum + A1–A8.

## 5. Attack taxonomy and lifecycle
- Lifecycle (Injection → Persistence → Retrieval → Activation → Privileged
  Action → Trace Suppression) → the event sequence in `orchestrator.py`
  (`inject` / `plan` / `retrieve` / `attack_fire` / `trace_suppress`).
  Visualized in `run_demo.py` part (1).
- Taxonomy A1–A8 → `AttackType`, the per-type triggers in
  `scenario.build_scenario`, and `AttackGenerator`.

## 6. Attack algorithm details
- 6.1 Basic attack model (C_t, a_t, y_t, M_{t+1}, activation predicate g)
  → `orchestrator._act`, `planner.trigger_fires`.
- 6.2 Attack generation procedure (steps 1–6) → `AttackGenerator.build`,
  `scenario.build_scenario`.
- 6.2 step 7 **adaptive attack** (update the payload / trigger from observed
  output) → `attacks/adaptive.py` (`run_adaptive` / `_mutate`), CLI `pdam adapt`.
- 6.3 Split-state attack (**independent variables: split count k, semantic
  distance, save-time difference, top-k**) → A3, `planner._reconstruct`,
  `policy.filter_candidates` (fragment co-retrieval limit). The independent
  variables are `AttackSpec.fragments/semantic_distance/save_time_spread/ttl`
  and `Scenario.top_k`; the sweep is the A3 section of `pdam ablate`.
- 6.4 Summary-reconstruction attack → `memory/summary.SummaryMemoryAdapter.compact`.
- 6.5 Self-re-save attack (**re-save generations, influence-decay rate,
  post-deletion survival rate**) → `orchestrator._self_resave`
  (generations/decay), `_post_deletion_survival` (survival after deletion).
  Metrics: `post_deletion_survival_rate` / `mean_resave_generations` /
  `mean_influence_decay`.

## 7. Experimental system design
- 7.1 Components (Orchestrator / Memory Store / Retriever / Planner / Tool
  Broker / Logger) → same-named modules, 1:1 (see the README table). The
  recorded data (session_id, content, timestamp, owner, trust, lineage, score,
  top-k, decision, event chain, ground truth) are fields on `State` / `ToolCall`
  / `LogEvent`.
- 7.2 Implementation candidates (multiple memory backends, multiple models,
  mock tools, fixed seed, versioning) → `memory/` (3 backends), the `Planner`
  swap point (`llm.py`), `tools/sandbox.py`, deterministic ids
  (`schema.new_id`/`reset_ids`).

## 8. Datasets and scenario design
- 8.1 Workloads (personal secretary / RAG support / coding support / SOC
  support) → `scenario._WL`. Each workload defines a benign task, an attack
  goal, and the main persistent states.
- 8.2 Scenario scale (4×8×difficulty×**repetitions**) → `scenario.all_scenarios`
  (4×8×3=96), `pdam batch --repeats N` (for a nondeterministic LLM planner).
- 8.3 Difficulty levels (Easy/Medium/Hard) → `content_style` / `delay` / `trust`
  / fragment count in `build_scenario`; Hard layers trace suppression + self-re-save.

## 9. Evaluation methodology and statistics
- 9.1 Primary metrics (ASR, Delayed ASR, Persistence Survival, Trigger
  Precision, Stealth, Unauthorized Action, Data Exfiltration, Benign Task
  Success, FPR, State Lineage Recovery) → `evaluator.metrics`.
- 9.2 Statistical analysis (Fisher exact, Holm correction, odds ratio, **95%
  CI**, absolute risk difference, mechanical judgement) →
  `evaluator.fisher_exact/holm/odds_ratio/compare`; the 95% CIs are
  `wilson_ci` (proportion) / `risk_diff_ci` (risk difference) / `odds_ratio_ci`
  (OR); judgement is `policy.judge`. Logistic regression is substituted by
  Fisher.
- 9.3 Ablation (memory backend, top-k, dormancy period, split count, single-
  defense effect) → `pdam ablate`, `aggregate(by=...)`.

## 10. Defenses and comparison baselines
- 10.1 Comparison baselines (Prompt-only / Content Filter / Embedding-Anomaly /
  LLM Judge / Least-Privilege / Provenance-aware Revalidation) → the flags of
  `DefenseConfig` and the three staged methods of `PolicyMonitor`
  (`screen_state` / `filter_candidates` / `authorize`).
- 10.2 Proposed minimal defense (save-time = provenance tagging, retrieval-time
  = revalidation + co-retrieval limit, execution-time = least privilege +
  approval) → `DefenseConfig.preset("minimal_defense")`.

## 11. Implementation plan
- 11.1 Module structure M1–M6 → the README table.
- 11.2 Log design (id on every event, time-ordered storage, synthetic-only
  secrets, recording of nondeterminism) → `logging_.EventLog`, `schema.LogEvent`,
  deterministic execution.

## 12. Reproducibility, ethics, safety
- Closed simulation / no contact with real services / synthetic data / published
  mechanical judgement / misuse suppression → all tools are in-memory mocks
  (`tools/sandbox.py`), no external communication, minimal payloads, and the
  `policy.judge` mechanical judgement is public.

## Appendix A/B
- A. Experiment scenario list → `scenario.all_scenarios` generates the same matrix.
- B. Data schema → mapped field-by-field in `schema.py`
  (run_id, session_id, state_id/parent_state_id, state_type, provenance,
   trust_level, created_at/expires_at, retrieval_score, trigger_condition,
   tool_call, attack_success, policy_violation).

## Explicit simplifications (to be replaced when writing the paper)
- The default `Planner` is a deterministic "susceptibility model". It can be
  swapped for an OpenAI-compatible local/commercial model via `pdam/llm.py`
  `LLMPlanner` (supporting the multi-model evaluation of §7.2).
- A logical clock (tick) is used so the testbed is wall-clock independent and
  reproducible (§12.1).
- The detection heuristics (`policy._looks_malicious` / `_suspicious_stored`,
  etc.) and the rule-based defenses are idealized (they identify provenance and
  trust precisely). Consequently `provenance` alone reaches ASR≈0 and FPR≈0,
  whereas an imperfect real-world classifier would produce false negatives and
  positives. Driving the testbed with the real-LLM planner (`LLMPlanner`) yields
  more realistic detection variance.
- Stealth Score is a binary approximation (reached execution = 1 / blocked = 0),
  a simplification of the §9.1 three-detector composite.

## Design gaps closed in this revision
- §6.2 step 7 adaptive attack → `attacks/adaptive.py` + `pdam adapt`.
- §6.3 independent variables (semantic distance, save-time difference, TTL) →
  `AttackSpec` + generator + `pdam ablate`.
- §6.5 A7 axes (generations, decay, post-deletion survival) →
  `orchestrator._self_resave` / `_post_deletion_survival` + 3 metrics.
- §9.2 95% confidence intervals → `evaluator.wilson_ci/risk_diff_ci/odds_ratio_ci`.
- §8.2 repetitions → `pdam batch --repeats N`.
