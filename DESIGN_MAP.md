# 設計書 → 実装 対応表

研究詳細設計書 v1.0 の各節が、コードのどこで実現されているかの対応です。

## 3. 対象システムと用語定義
- 対象エージェント機能（メモリ書込/検索・RAG/計画/ツール利用/状態更新）
  → `orchestrator.py` の R/P/E/U ループ、`memory/`, `retriever.py`, `planner.py`,
    `broker.py`。
- 主要用語（永続状態・状態汚染・遅延発火・潜伏期間・状態合成・来歴）
  → `schema.py`（`State`, `Provenance`, `Directive`）、`planner._reconstruct`
    （状態合成）、`memory/store.lineage`（来歴追跡）。

## 4. 脅威モデル
- 攻撃者能力（通常ユーザーとして複数問合せ、共有文書/メール等へ配置、内部プロンプト
  非アクセス、ブラックボックス観測、別ユーザー/別セッション標的）
  → `AttackGenerator`（外部由来・低信頼状態の注入）、A6 跨セッション。
- 攻撃目標（不正行動/情報漏えい/意思決定操作/継続的支配/痕跡抑制）
  → `PolicyViolation` 列挙 + 各 A1–A8。

## 5. 攻撃分類と攻撃ライフサイクル
- ライフサイクル（Injection→Persistence→Retrieval→Activation→Privileged Action
  →Trace Suppression）
  → `orchestrator.py` のイベント列（`inject`/`plan`/`retrieve`/`attack_fire`/
    `trace_suppress`）。`run_demo.py` の (1) で可視化。
- 攻撃分類 A1–A8 → `AttackType`、`scenario.build_scenario` の型別トリガー、
  `AttackGenerator`。

## 6. 攻撃アルゴリズム詳細
- 6.1 基本攻撃モデル（C_t, a_t, y_t, M_{t+1}、発火述語 g） → `orchestrator._act`,
  `planner.trigger_fires`。
- 6.2 攻撃生成手順（手順1-6） → `AttackGenerator.build`, `scenario.build_scenario`。
- 6.2 手順7 **適応攻撃**（出力観測に基づくペイロード/発火条件更新）
  → `attacks/adaptive.py`（`run_adaptive`/`_mutate`）、CLI `pdam adapt`。
- 6.3 分割状態攻撃（**独立変数：分割数 k・意味的距離・保存時刻差・top-k**）
  → A3、`planner._reconstruct`、`policy.filter_candidates`（断片同時取得制限）。
    独立変数は `AttackSpec.fragments/semantic_distance/save_time_spread/ttl` と
    `Scenario.top_k`。スイープは `pdam ablate` の A3 セクション。
- 6.4 要約再構成攻撃 → `memory/summary.SummaryMemoryAdapter.compact`。
- 6.5 自己再保存攻撃（**再保存世代・影響減衰率・状態削除後の残存率**）
  → `orchestrator._self_resave`（世代/減衰）、`_post_deletion_survival`（削除後残存）。
    指標：`post_deletion_survival_rate` / `mean_resave_generations` /
    `mean_influence_decay`。

## 7. 実験システム設計
- 7.1 構成要素（Orchestrator/Memory Store/Retriever/Planner/Tool Broker/Logger）
  → 同名モジュールに 1:1 対応（README の表）。記録データ（session_id, content,
    timestamp, owner, trust, lineage, score, top-k, decision, event chain,
    ground truth）は `State` / `ToolCall` / `LogEvent` フィールド。
- 7.2 実装候補（複数メモリ方式・複数モデル・模擬ツール・固定シード・版管理）
  → `memory/`（3 方式）、`Planner` 差し替え点、`tools/sandbox.py`、決定論的 id
    (`schema.new_id`/`reset_ids`)。

## 8. データセットとシナリオ設計
- 8.1 ワークロード（個人秘書/RAG 業務支援/コーディング支援/SOC 支援）
  → `scenario._WL`。正常タスク・攻撃目標・主な永続状態を各ワークロードに定義。
- 8.2 シナリオ規模（4×8×難易度×**反復**） → `scenario.all_scenarios`（4×8×3=96）、
  `cli batch --repeats N`（非決定性 LLM 計画器での反復に対応）。
- 8.3 難易度レベル（Easy/Medium/Hard） → `build_scenario` の `content_style`/
  `delay`/`trust`/断片数、Hard は痕跡抑制+自己再保存を上乗せ。

## 9. 評価方法と統計計画
- 9.1 主要評価指標（ASR, Delayed ASR, Persistence Survival, Trigger Precision,
  Stealth, Unauthorized Action, Data Exfiltration, Benign Task Success, FPR,
  State Lineage Recovery） → `evaluator.metrics`。
- 9.2 統計解析（Fisher 正確確率、Holm 補正、オッズ比・**95%CI**・絶対リスク差、
  機械判定） → `evaluator.fisher_exact/holm/odds_ratio/compare`、95%CI は
  `wilson_ci`（比率）/`risk_diff_ci`（リスク差）/`odds_ratio_ci`（OR）、
  判定は `policy.judge`。ロジスティック回帰は Fisher で代替。
- 9.3 アブレーション（メモリ方式・top-k・潜伏期間・分割数・防御の単独効果）
  → `cli ablate`, `aggregate(by=...)`。

## 10. 防御方式と比較対象
- 10.1 比較ベースライン（Prompt-only/Content Filter/Embedding-Anomaly/LLM Judge/
  Least-Privilege/Provenance-aware Revalidation） → `DefenseConfig` の各フラグと
  `PolicyMonitor` の 3 段階メソッド（`screen_state`/`filter_candidates`/`authorize`）。
- 10.2 提案する最小防御（保存時=来歴付与、検索時=再検証+同時取得制限、実行時=
  最小権限+承認） → `DefenseConfig.preset("minimal_defense")`。

## 11. 実装計画
- 11.1 モジュール構成 M1–M6 → README の対応表。
- 11.2 ログ設計（全イベントに id 付与、時系列保存、機密は合成のみ、非決定性の記録）
  → `logging_.EventLog`, `schema.LogEvent`, 決定論的実行。

## 12. 再現性・倫理・安全管理
- 閉じた模擬環境・実サービス非通信・合成データ・機械判定公開・悪用抑制
  → 全ツールがインメモリのモック（`tools/sandbox.py`）、外部通信なし、
    ペイロードは最小限、`policy.judge` の機械判定を公開。

## 付録 A/B
- A 実験シナリオ一覧 → `scenario.all_scenarios` が同一マトリクスを生成。
- B データスキーマ → `schema.py` のフィールドに逐一対応
  （run_id, session_id, state_id/parent_state_id, state_type, provenance,
   trust_level, created_at/expires_at, retrieval_score, trigger_condition,
   tool_call, attack_success, policy_violation）。

## 実装上の明示的な単純化（論文化時に差し替える箇所）
- 既定 Planner は決定論的な「感受性モデル」。`pdam/llm.py` の `LLMPlanner` で
  OpenAI 互換ローカル/商用モデルへ差し替え可能（§7.2 の複数モデル評価に対応）。
- 論理時刻（tick）を採用し wall-clock 非依存で再現性を担保（§12.1）。
- 検知ヒューリスティック（`policy._looks_malicious`/`_suspicious_stored` 等）と
  ルールベース版の防御は理想化されている（来歴・信頼度を正確に識別）。このため
  provenance 単独で ASR≈0・FPR≈0 となり、実運用の不完全な分類器では偽陰性・偽陽性が
  生じる。実 LLM 計画器（`LLMPlanner`）で駆動すると、より現実的な検知揺らぎを評価できる。
- Stealth Score は「実行到達=1/遮断=0」の二値近似（§9.1 の3検出器合成値の簡略化）。

## 補完済みギャップ（本改訂で実装）
- §6.2 手順7 適応攻撃 → `attacks/adaptive.py` + `pdam adapt`。
- §6.3 独立変数（意味的距離・保存時刻差・TTL）→ `AttackSpec` + 生成器 + `pdam ablate`。
- §6.5 A7 評価軸（世代・減衰・削除後残存）→ `orchestrator._self_resave` /
  `_post_deletion_survival` + 指標3種。
- §9.2 95%信頼区間 → `evaluator.wilson_ci/risk_diff_ci/odds_ratio_ci`。
- §8.2 反復 → `pdam batch --repeats N`。
