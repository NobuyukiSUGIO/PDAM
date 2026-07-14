# PDAM — Persistent-State and Delayed-Activation Attacks on Memory-Augmented LLM Agents

研究詳細設計書「メモリ拡張型 LLM エージェントに対する永続状態汚染と遅延発火攻撃」
(Version 1.0) の**実験基盤（テストベンチ）実装**です。設計書 §7（実験システム設計）
と §11（実装計画）のモジュール構成 M1–M6 を、**外部依存ライブラリ無し・Python 標準
ライブラリのみ**で再現し、`python3 -m pdam` だけで攻撃・防御・評価が一通り走ります。

> 本実装は設計書 §12（倫理・安全管理）に従い、**完全に閉じた模擬環境**です。実サービス
> への通信は一切行わず、ツールはすべてインメモリのモック、データは合成のみです。攻撃
> ペイロードは因果構造の検証に必要な最小限に留め、直接悪用可能な高性能ペイロードは
> 含みません。

---

## 何を実装したか

設計書の中心仮説（§14.3）——
「LLM エージェントの永続メモリは単なるデータ保管ではなく、将来の制御フローと権限行動を
変化させる**実行状態**である。保存時の内容検査だけでは不十分であり、状態の**来歴**・
検索時の**合成**・**発火条件**・実行時**権限**を横断して評価・制御する必要がある」——
を、動く形で再現・計測します。

| 設計書のモジュール | 実装 | 内容 |
|---|---|---|
| M1 Scenario Engine | `pdam/scenario.py`, `pdam/orchestrator.py` | 正常/攻撃シナリオ定義（JSON/YAML）と実行器。付録 A の 4×8×3 マトリクス生成 |
| M2 Memory Adapter | `pdam/memory/` | vector / summary / KV の 3 メモリ方式を共通 API で提供 |
| M3 Attack Generator | `pdam/attacks/generator.py` | 攻撃分類 A1–A8、分割・遅延・要約・自己再保存の生成 |
| M4 Tool Sandbox | `pdam/tools/sandbox.py` | 模擬メール/ファイル/コード/チケット/SOC/ログツール |
| M5 Policy Monitor | `pdam/policy.py`, `pdam/broker.py` | §10 の 6 防御と最小防御、機械判定（§9.2） |
| M6 Evaluator | `pdam/evaluator.py` | §9.1 の全指標、Fisher 正確検定・Holm 補正・オッズ比 |

システム構成（§7.1）の各要素も 1:1 で対応します：Agent Orchestrator
(`orchestrator.py`)、Memory Store (`memory/store.py`)、Retriever/RAG
(`retriever.py`)、LLM Planner (`planner.py`)、Tool Broker (`broker.py`)、
Logger/Monitor (`logging_.py` + `policy.py`)。詳細は `DESIGN_MAP.md` を参照。

---

## クイックスタート

```bash
# 依存インストール不要（Python 3.10+ のみ）
python3 -m pdam list-attacks          # 攻撃分類 A1–A8
python3 -m pdam list-defenses         # 防御プリセット

# 付録 A の 96 シナリオを生成
python3 -m pdam gen-scenarios scenarios/

# 1 シナリオを実行（イベントログのトレース付き）
python3 -m pdam run scenarios/personal_secretary_A3_hard.json \
        --defense minimal_defense --trace

# 全シナリオ × 全防御を実行し、集計表と CSV/JSON を results/ に出力
python3 -m pdam batch --outdir results

# アブレーション（難易度 / メモリ / ワークロード別 + §6.3 分割状態の独立変数）
python3 -m pdam ablate

# 反復実行（§8.2、非決定性の LLM 計画器で有効）
python3 -m pdam batch --repeats 5

# 適応攻撃（§6.2 手順7：遮断段階を観測してペイロード/発火条件を更新し再試行）
python3 -m pdam adapt --workload personal_secretary --difficulty easy

# 実 LLM 計画器（LM Studio 等の OpenAI 互換ローカルモデル）
python3 -m pdam llm-eval --model mistral-7b-instruct-v0.3

# エンドツーエンドのデモ（攻撃ライフサイクル可視化 + 統計）
python3 run_demo.py

# テスト
python3 -m unittest discover -s tests
# もしくは  make test / make demo / make batch
```

---

## 攻撃モデル（§6.1）

エージェントの 1 ステップは設計書の式そのままに動きます：

```
C_t = R(x_t, M_t)      検索        Retriever.retrieve
a_t = P(x_t, C_t)      計画・状態合成 Planner.plan
y_t = E(a_t)           実行        ToolBroker.execute
M_{t+1} = U(...)       状態更新     Orchestrator._update_memory
```

攻撃者は時刻 `t0` に状態 `m*` を注入し、`t1 > t0` で発火述語
`g(x_t1, C_t1, s_t1)=1` を満たしたとき標的行動 `a*` を選択させます。`m*` は**単体では
ポリシー違反と判定されない**設計です（分割・自然文化・要約経由・信頼度昇格による）。

### 攻撃分類 A1–A8（§5.1）

| ID | 型 | 発火条件 | 実装のポイント |
|---|---|---|---|
| A1 | 時間遅延 | 経過時間 `after` / 絶対時刻 `at` | `trigger={"type":"time"}` |
| A2 | イベント条件 | 標的タスクのキーワード一致 | `trigger={"type":"event"}` |
| A3 | 分割状態 | 全断片の同時検索で再構成 | k 断片に分割、`Planner._reconstruct` が結合 |
| A4 | 要約再構成 | 要約圧縮時に命令が統合 | `SummaryMemoryAdapter.compact` が来歴に `reconstruct` を刻む |
| A5 | 信頼度昇格 | 反復利用で trust が high に昇格後 | `Orchestrator._escalate` |
| A6 | 跨セッション | 注入と別セッションで発火 | `trigger={"type":"cross_session"}` |
| A7 | 自己再保存 | 発火後に成果を再保存し持続 | `Orchestrator._self_resave` |
| A8 | 痕跡抑制 | 発火後に監査ログを改竄 | `Orchestrator._suppress_trace` |

---

## 防御（§10）

`--defense` で切替。個別ベースラインと、§10.2 の 3 段階「最小防御」を提供します。

| プリセット | 段階 | 内容 |
|---|---|---|
| `none` | — | 防御なし（ベースライン） |
| `prompt_only` | 推論時 | システムプロンプトのみ（最弱） |
| `content_filter` | 保存時 | 内容検査。**明示的な単発命令は捕捉するが、分割(A3)・要約(A4)・自然文は回避される** |
| `embedding_filter` | 検索時 | 異常記録の検索除外 |
| `llm_judge` | 実行前後 | ツール引数の危険性検査 |
| `least_privilege` | 実行時 | ユーザー意図に結び付かない高権限呼出を制限（**op が別なら捕捉、認可 op の再利用は取り逃す**） |
| `provenance` | 検索+実行時 | 来歴考慮再検証。**低信頼・外部由来・派生（要約/昇格）由来の状態が権限行動を駆動することを拒否**。低信頼断片の同時取得も制限 |
| `minimal_defense` | 保存+検索+実行 | §10.2 提案（content_filter + provenance + least_privilege + 断片同時取得制限） |
| `full` | 全段階 | 全防御 |

---

## 評価指標（§9.1）と統計（§9.2）

`Evaluator` が以下を算定します：Attack Success Rate、Delayed ASR、Persistence
Survival Rate、Trigger Precision、Stealth Score、Unauthorized Action Rate、Data
Exfiltration Rate、Benign Task Success、False Positive Rate、State Lineage
Recovery。防御間比較は **Fisher 正確確率検定**、多重比較は **Holm 補正**、効果量は
**オッズ比・絶対リスク差**（すべて標準ライブラリで実装、scipy 不要）。

### 代表的な結果（96 シナリオ × 各防御、固定シード）

```
defense          asr    delayed  persist  trig.P  stealth  unauth  exfil  benign  FPR   lineage
none             1.000  1.000    1.000    0.906   1.000    1.000   0.500  1.000   0.000 0.500
content_filter   0.760  0.760    0.771    0.667   0.760    0.760   0.354  1.000   0.000 0.562
least_privilege  0.500  0.500    0.990    0.917   0.500    0.500   0.250  1.000   0.000 0.938
provenance       0.000  0.000    0.490    0.406   0.000    0.000   0.000  1.000   0.000 1.000
minimal_defense  0.000  0.000    0.260    0.188   0.000    0.000   0.000  1.000   0.000 1.000
```

この表がそのまま設計書の検証仮説を示します：

- **H1**（遅延発火は検知率が低い）: `content_filter` は自然文化・遅延型を取り逃す。
- **H2**（分割状態は内容フィルタを回避しやすい）: A3 は Easy でも `content_filter` を通過。
- **H3**（要約は攻撃を強化しうる）: A4 は要約で信頼ロンダリングし、`content_filter` と
  `least_privilege` を回避——`provenance` のみが阻止。
- **H4**（来歴付き再検証 + 実行時最小権限の組合せが高い安全性）: `minimal_defense` は
  ASR=0 かつ Benign Task Success=1.0 / FPR=0 で**利便性を維持したまま全攻撃を抑止**。
- **State Lineage Recovery**: `none` で 0.500（A8 痕跡抑制と A4 要約が来歴を破壊）、
  `provenance`/`minimal_defense` で 1.0。

---

## 設計・拡張

- **実 LLM への差し替え**: `pdam/planner.py` の `Planner` を実装し、構造化ディレクティブ
  ではなく `State.content` の自然文を解釈させれば、商用/オープンモデルで再現できます
  （設計書 §7.2「最低 4 モデル評価」）。既定の `RuleBasedPlanner` は決定論的な感受性
  モデルで、固定シード・オフライン再現（§12.1）を保証します。
- **メモリ方式の追加**: `pdam/memory/base.py` の `MemoryAdapter` を継承。
- **新しい攻撃/防御**: `AttackSpec`（`attacks/generator.py`）と `DefenseConfig`
  （`policy.py`）にフィールドを追加。
- **ログ**: 全イベントに `run_id / session_id / task_id / state_id /
  parent_state_id` と論理時刻が付与され（§11.2）、`res.events.save(path)` で JSONL 出力。

---

## ディレクトリ構成

```
pdam/
  schema.py        付録 B のデータスキーマ（State/Provenance/Directive/ToolCall/LogEvent）
  embedding.py     依存なしの BoW 埋め込み + コサイン類似度
  memory/          M2: base / vector / summary / kv / store
  attacks/         M3: 攻撃生成器（A1–A8）
  tools/           M4: ツールサンドボックス
  policy.py        M5: 防御 6 種 + 最小防御 + 機械判定
  broker.py        Tool Broker（媒介・認可）
  retriever.py     Retriever/RAG（検索時防御）
  planner.py       LLM Planner（状態合成・発火判定）
  orchestrator.py  Agent Orchestrator（R/P/E/U ループ）
  logging_.py      イベントログ
  evaluator.py     M6: 指標算定・統計
  cli.py           コマンドライン
scenarios/         生成されたシナリオ（JSON）
tests/             ユニット + 結合テスト（23 件）
run_demo.py        エンドツーエンドのデモ
DESIGN_MAP.md      設計書の各節 → コードの対応表
```

---

## ライセンス / 位置づけ

研究・教育目的の防御研究用テストベッドです（設計書 §16.2「安全なエージェント開発教育に
利用できる実験環境と教材」）。責任ある開示の原則に従い、実運用エージェントの具体的
脆弱性の悪用を助けるものではありません。
