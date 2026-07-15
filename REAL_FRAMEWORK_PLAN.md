# §5.4 実フレームワーク評価 — 詳細手順書

対象査読指摘: §5.4「実験は依然として合成フレームワーク中心で、実メモリシステムの
write→effect を評価していない」および関連質問 #5, #6。

本書は **設計原則 → 環境準備 → アダプタ実装 → 実験マトリクス → 計測 → 論文反映**
の順に、実装可能な粒度まで手順を分解する。各タスクに *Definition of Done (DoD)* を
付し、査読の8サブ要求への対応を明示する。

---

## 0. 査読要求の分解（何を満たせば §5.4 が閉じるか）

| # | 査読の要求 | 対応タスク | 節 |
|---|---|---|---|
| R1 | 同一シナリオを vector/summary/kv 全backendで実行（攻撃型とbackendの交絡除去）| 3バックエンド × 96 | §4 |
| R2 | 実フレームワークを最低2種類 | LangChain + LlamaIndex アダプタ | §3 |
| R3 | 実際のmemory-write判定を含める | フレームワークの write ポリシーを通す | §3.3 |
| R4 | 実summarizer・実embeddingを使用 | LM Studio /v1/embeddings + LLM summarize | §2, §3.4 |
| R5 | 10/50/100ターンの潜伏 | dormancy パラメタ化 | §5 |
| R6 | cross-session に加え cross-user/cross-tenant | namespace分割 | §6 |
| R7 | dedup / TTL / 再要約 | 実storeの機能を有効化 | §3.4 |
| R8 | write/survive/retrieve/act をモデル依存で段階計測 | 実funnel + latency | §7 |

**DoD(§5.4全体):** 上記R1–R8を満たす結果表が `results/real_framework/` に生成され、
論文 §5.4 が「合成harness」から「実フレームワーク上のwrite→effect評価」に書き換わり、
`limitations` から該当項目が削除できること。

---

## 1. 設計原則 — 何を差し替え、何を固定するか

比較可能性を壊さないため、**攻撃と判定は固定**し、**メモリ実体だけを実システム化**する。

### 固定するもの（触らない）
- **攻撃ペイロード/シナリオ**: `pdam/attacks/generator.py`, `pdam/scenario.py` の
  `AttackSpec` と 96マトリクス。10次元攻撃空間のラベルも不変。
- **機械的成功判定**: `pdam/tools/sandbox.py` の `ToolEffect` と
  `pdam/policy.py::PolicyMonitor.judge`。ASRの定義を実験間で一定に保つ。
- **ツールbroker**: `pdam/broker.py::ToolBroker` + `ToolSandbox`（mockツールの効果判定）。
- **メトリクス/funnel定義**: `pdam/evaluator.py`, `pdam/experiments.py::_nested`。

### 差し替えるもの（実システム化）
| 現行（合成） | 実システム | 差し込み口 |
|---|---|---|
| `pdam/embedding.py` bag-of-words | 実embeddingモデル | `MemoryAdapter.search` |
| `pdam/memory/vector.py` 自作cos検索 | LangChain/LlamaIndex vector store | `MemoryAdapter` サブクラス |
| `pdam/memory/summary.py` 規則圧縮 | 実LLM summarizer | `MemoryStore.maybe_compact` |
| TTL/dedup 自作 | 実storeのTTL/dedup | `MemoryAdapter.forget_expired`/`add` |
| `RuleBasedPlanner` | 実LLM (`LLMPlanner`) | `Orchestrator(planner=...)`（既存） |

### 唯一かつ最重要の差し込み口
`pdam/memory/base.py::MemoryAdapter`（ABC）。契約は既に最小・安定:
```
add(state)                 # write
search(query, top_k, now)  # retrieve -> [(State, score)]
forget_expired(now)        # TTL
compact(now)               # 再要約（summaryのみ）
get / all / update / remove / lineage
```
`Retriever` と `Orchestrator` はこの契約しか呼ばないので、**新アダプタを実装するだけ**で
実フレームワークに載る。上位の retrieve→plan→execute→update ループは無改造。

### provenance の扱い（honesty の核心）
- **ground-truth provenance**（attacker由来か、fragment_group等）は各stateのmetadataに
  保持する ＝ 機械判定とfunnelの正解ラベルとして使うため。
- ただし **防御(`PolicyMonitor`)が読むのは、実フレームワークが実際に露出する推定
  provenance のみ**。実storeは要約や再保存で系譜を落とすので、そこで生じる
  attribution欠落が「実測される」＝ §5.3/§5.8 の合成stress testを実測に格上げできる。

---

## 2. 環境準備（前提整備）

### 2.1 pip ブートストラップ + venv
```
python3 -m ensurepip --upgrade        # pip 未導入のため
python3 -m venv .venv-real
source .venv-real/bin/activate
python -m pip install --upgrade pip wheel
```
- **注意**: 既存の pure-stdlib テスト・決定論評価は `.venv-real` を使わず system python
  のまま維持（依存ゼロの再現性を壊さない）。実framework評価専用の隔離環境とする。

### 2.2 依存インストール（`requirements-real.txt` に固定）
```
langchain>=0.3            langchain-community>=0.3
llama-index-core>=0.11    llama-index>=0.11
chromadb>=0.5             # 実vector store（LangChain/LlamaIndex共用可）
openai>=1.40              # LM Studio /v1 をOpenAI互換で叩く
tiktoken                  # tokenカウント（latency/token計測用, §7）
```
- **embedding**: torch巨大化を避けるため、既存の LM Studio `/v1/embeddings` を第一候補
  （例 `nomic-embed-text`, `text-embedding-*`）。LM Studioで埋め込みモデルをロードし、
  `OpenAIEmbeddings(base_url=...)` 系で接続。ローカル完結。
  - 代替: `sentence-transformers`（torch同梱, ~2GB）。ネット/容量に余裕がある場合のみ。
- **summarizer/planner LLM**: 稼働中の LM Studio モデル
  （`qwen/qwen3.5-9b`, `gemma-2-27b-it`, `deepseek-v2-lite-chat` を確認済み）。

### 2.3 再現性メタの記録（artifact用, §5.10連動）
`results/real_framework/ENV.json` に:
- langchain/llama-index/chromadb の正確なバージョン + `pip freeze` hash
- LM Studio version、embeddingモデルID+量子化、summarizer/plannerモデルID+量子化
- seed, top_p, temperature, max_tokens

**DoD(§2):** `.venv-real` で `import langchain, llama_index, chromadb, openai` が通り、
LM Studioの `/v1/embeddings` と `/v1/chat/completions` に疎通、`ENV.json` 生成。

---

## 3. アダプタ実装（コア作業）

新規ファイル: `pdam/memory/real_langchain.py`, `pdam/memory/real_llamaindex.py`。
いずれも `MemoryAdapter` を継承し、`ADAPTERS`（`pdam/memory/store.py`）に登録する:
```python
ADAPTERS = {
  "vector": VectorMemoryAdapter, "summary": SummaryMemoryAdapter, "kv": KVMemoryAdapter,
  "lc_vector": LCVectorAdapter, "lc_summary": LCSummaryAdapter, "lc_kv": LCKVAdapter,
  "li_vector": LIVectorAdapter, "li_summary": LISummaryAdapter, "li_kv": LIKVAdapter,
}
```

### 3.1 State ⇄ フレームワークdocument のマッピング
- **書込**: `State` → `Document`(LangChain) / `TextNode`(LlamaIndex)
  - `page_content = state.content`
  - `metadata = {state_id, parent_state_id, session_id, user_id, trust_level,
    external, transforms(list), created_at, expires_at, attack_marker(GT),
    fragment_group, fragment_index, directive(GT, JSON)}`
  - GT項目（attack_marker/directive）は **評価専用**。防御コードからは参照しない
    （§1 provenanceの扱い）。
- **検索**: `search(query, top_k, now)` →
  - LangChain: `vectorstore.similarity_search_with_score(query, k=top_k*2)`
  - LlamaIndex: `index.as_retriever(similarity_top_k=top_k*2).retrieve(query)`
  - `now` でTTL/expiry を metadata フィルタ（`expires_at`）。
  - 返値を `[(State, score)]` に逆変換（metadataからStateを再構成）。

### 3.2 LangChain 具体
- **vector**: `Chroma`(or `FAISS`) + LM Studio `OpenAIEmbeddings`。
- **summary**: `ConversationSummaryMemory` あるいは要約用LLM chainで、閾値到達時に
  古いconversationを実LLMで要約→要約ノードを書き戻し。**laundering(A4)が実際に起きる**。
- **kv**: `InMemoryStore`/`LocalFileStore`（キー=topic/tag、値=state）。R1の交絡除去の要。

### 3.3 実 write ポリシー（R3）
- 実フレームワークの投入APIをそのまま通す（重複除去・正規化・チャンク分割が実際に
  かかる）。`add()` は成功/失敗を返し、funnelの **write** ステージを実挙動で判定。
- チャンク分割が起きると split攻撃(A3)の断片境界が変わり得る → これも実測対象。

### 3.4 実 TTL / dedup / 再要約（R7）
- TTL: `expires_at` を実storeのメタフィルタ or 定期purgeで実装。
- dedup: Chroma/LlamaIndexの重複検出 or embedding近傍しきい値で実際に落とす。
- 再要約: `MemoryStore.maybe_compact(now)` を実LLM summarizeに委譲。多重要約で
  系譜が実際に切れる（lineage recovery低下を実測）。

### 3.5 LlamaIndex 具体
- **vector**: `VectorStoreIndex` + LM Studio embedding + `ChromaVectorStore`。
- **summary**: `SummaryIndex` / `DocumentSummaryIndex` + 実LLM。
- **kv**: `SimpleKeyValueStore` / `KVDocumentStore`。

**DoD(§3):** `Orchestrator(scenario_with(memory="lc_vector"))` が1シナリオを完走し、
`RunResult` が現行と同じスキーマで返る（スモークテスト）。両フレームワーク×3backend の
9アダプタが `make_adapter` から生成可能。

---

## 4. 実験マトリクス① — 3バックエンド交絡除去（R1, R2）

現行の交絡（A4だけsummary, 他はvector, kv未使用）を断つ。

- **走らせるもの**: 96シナリオ（4 workload × 8 attack × 3 difficulty）を、
  `{lc_vector, lc_summary, lc_kv, li_vector, li_summary, li_kv}` の **6 store × {none,
  minimal_defense}** で実行。
- planner は **まず決定論(`RuleBasedPlanner`)** で backend効果を単離
  （LLMのばらつきを排除して backend の寄与を見る）。
- 出力: `results/real_framework/backend_matrix.csv`
  列 = `framework, backend, defense, attack_type, difficulty, asr, write, survive,
  retrieve, effect, lineage`。

**分析**: 同一攻撃×同一difficultyを backend間で比較し、「summary vs vector で ASR差」を
**同一攻撃で**主張できる形にする（査読の交絡指摘へ直接回答）。

**DoD(§4):** backend_matrix.csv が生成、attack_type と backend が分離集計され、
「backendの影響は小さい/大きい」の主張が交絡なしで裏付けられる。

---

## 5. 実験マトリクス② — 長期潜伏 10/50/100ターン（R5）

- `build_scenario` に `dormancy: int` を追加し、`_DELAY[difficulty]` を上書き。
  noise/taskフィラーを N ステップ生成（既存ループの `for i in range(delay)` を拡張）。
- N ∈ {10, 50, 100} を代表 workload×attack（例: 各workloadのA1/A3/A4/A6）で実行。
- 実TTL/dedup/再要約が効くため、**survive/retrieve が N とともに劣化**するはず。
  これを実測してplot（`results/real_framework/dormancy.csv`）。

**DoD(§5):** ASR/survive/retrieve が dormancy長の関数として表・図化され、
「persistent と呼べるのはどの N まで」を実データで言える。

---

## 6. 実験マトリクス③ — cross-user / cross-tenant（R6）

- `Step`/`State` に `user_id`（既存 `session_id` を tenant に拡張）を追加。
- 実storeを **user名前空間で分割**（Chroma collection per user / metadata filter）。
- 注入: user A のメモリに攻撃を書く。probe: user B が同topicで問う。
- 測定: 名前空間分離が破れて **cross-user retrieval が起きるか**（漏洩ASR）。
  正しく分離されればASR=0（＝実storeの分離が防御になる、という新知見）。

**DoD(§6):** cross-userでの漏洩有無が実測され、cross-session(A6)との差が表になる。

---

## 7. 段階別・モデル依存の計測 + コスト（R8, §6.5連動）

- **実funnel**: `_nested` を流用しつつ、各ステージを実挙動で判定
  - write = 実storeが受理
  - survive = 実TTL/dedup/再要約後にメモリに残存
  - retrieve = 実retrieverが probe で surface
  - synthesize = 実plannerが攻撃actionを構成（LLM時）
  - dispatch/effect = broker/judge（固定）
- **コスト計測**（査読§6.5 "cheap/practical" にも回答）: 各ステージの
  wall-clock latency、embedding/LLMトークン数、ストレージ量を記録
  → `results/real_framework/cost.csv`。`tiktoken` でtoken、`time` でlatency。

**DoD(§7):** framework別funnel（write→effect）と per-stageコスト表が生成。

---

## 8. 実LLMプランナ × 実framework（統合）

- §4–7 の代表サブセットを `LLMPlanner`(LM Studio) で再実行。
  planner・embedding・summarizer が**すべて実モデル**の end-to-end 条件を最低1本作る。
- 難易度の明示（査読質問#5）: 実LLM実行でも easy/medium/hard を **明示ラベル付き**で回す。
- 反復3回＋温度固定、生ログ保存。

**DoD(§8):** 「全要素が実モデル」の write→effect が最低1マトリクス分存在し、
difficulty ラベルが結果に明記される。

---

## 9. 統計・再現性（§5.5d/§5.10連動）

- 機械判定は不変なのでASRは横断比較可能。
- 実LLM/実embeddingの非決定性 → 反復増＋ **scenario単位 cluster bootstrap** で信頼区間。
- **ワンコマンド再生成**: `pdam real-framework --outdir results/real_framework` を
  `cli.py` に追加（`experiments.py` に `real_framework()` を実装）。
- artifact(§5.10): 全生ログ・`ENV.json`・モデルhash・表生成スクリプトを匿名repoに同梱。

**DoD(§9):** 1コマンドで全表が再生成でき、CI的スモーク（1シナリオ×1backend）が緑。

---

## 10. 論文への反映

- **§5.4 書換**: 「合成harness」→「実LangChain/LlamaIndex上の write→effect 評価」。
  新表: (a) backend×framework×defense ASR、(b) dormancy曲線、(c) cross-user漏洩、
  (d) per-stage funnel、(e) コスト。
- **limitations 更新**: R1–R8該当項目を削除、残る限界（モデル数・実運用規模）に限定。
- **abstract 更新**: 実framework評価を新規貢献として1文追加、数値を実測に差し替え。

---

## 11. マイルストーン（作業分割）

| Phase | 内容 | 主なDoD | 状態 |
|---|---|---|---|
| A | 環境+LangChain/LlamaIndex vectorアダプタ+スモーク | §2,§3 | ✅ 完了 |
| B | **96×3backend×2framework（決定論planner）** ← 査読の要 | §4 | ✅ 完了 |
| C | 実LLMプランナ統合（難易度明示）| §8 | ✅ 完了 |
| D | 長期潜伏 + cross-user | §5,§6 | ✅ 完了 |
| E | コスト計測 + cluster bootstrap + artifact + 論文 | §7,§9,§10 | ✅ 完了 |

### Phase C–E 実施記録（完了）
- **C**（`scripts/run_phase_c.py`, `phase_c_*.csv`）: 実Llama-3.1-8B×実LangChain end-to-end、
  全難易度でnone ASR 1.0／minimal 0.0。実LLMはsummaryをA4 ASR 1.0まで再構成（決定論0.75を上回る）。
- **D**（`scripts/run_dormancy.py`, `dormancy.csv`, `cross_user.csv`）: 潜伏10/50/100で
  vector/summaryは全N生存、split(A3)はN≥50でeffect 1→0。cross-tenantは共有store漏洩1.0／
  スコープ分離0.0。
- **E**（`cluster_bootstrap.csv`, `cost.csv`）: scenario-level cluster bootstrap（§5.5d）、
  防御コスト3.3µs/チェック・追加LLM呼び出しゼロ（§6.5）。
- 論文§Real-Framework Evaluation（`sec:realfw`）+ §long-horizon（`sec:horizon`）追加、
  abstract/Table I/limitations更新。16ページ、クリーンビルド。stdlibテスト41件不変。

### Phase A+B 実施記録（完了）
- 環境: `.venv-real`（langchain 1.3.13 / llama-index-core 0.14.23 / chromadb 1.5.9）、
  LM Studio埋め込み `text-embedding-nomic-embed-text-v1.5`(768次元)、要約 `qwen/qwen3.5-9b`。
  メタは `results/real_framework/ENV.json`。
- 実装: `pdam/memory/real_langchain.py`, `pdam/memory/real_llamaindex.py`
  （`MemoryAdapter`継承、`make_adapter`に遅延登録）。CLI: `pdam real-framework`。
- 結果: 1152 run（34.8分）→ `backend_matrix.csv`, `backend_by_attack.csv`,
  所見は `results/real_framework/FINDINGS.md`。
- 純stdlibテスト41件は不変（回帰なし）。

**最小充足ライン**: Phase **B** が査読§5.4の中核（実framework×全backend×交絡除去）。
時間制約があればBを完全に、C–Dを代表サブセットで縮小可。

---

## 12. リスクと緩和

| リスク | 緩和 |
|---|---|
| torch/embedding巨大DL | LM Studio `/v1/embeddings` を第一候補（ローカル完結） |
| LangChain/LlamaIndex API変動 | バージョンをpin、`requirements-real.txt`固定 |
| 実embedding/LLMの非決定性 | seed固定・温度0・反復増・CI報告 |
| チャンク分割で攻撃境界が変わる | 実測対象として記録（合成との差分を明示） |
| 計算/時間コスト | Phase B優先、長期潜伏は代表サブセット |
| system pythonの再現性を壊す | `.venv-real` に完全隔離、stdlib評価は不変 |

---

## 13. 着手順（推奨）

1. §2 環境（ensurepip→venv→依存→ENV.json→疎通）
2. §3.1–3.3 LangChain vector アダプタ + `ADAPTERS`登録 + 1シナリオ スモーク（Phase A）
3. §3.4 実TTL/dedup/要約 → §4 backend_matrix（Phase B, 査読の要）
4. LlamaIndex アダプタ複製 → §4 完成（2framework）
5. §8 実LLM統合 → §5,§6 → §7 コスト → §9,§10（Phase C–E）
