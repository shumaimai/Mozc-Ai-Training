# 次タスク指示: 30m 文脈リランカー（速度回収）＋ 常駐デーモン配信

対象: リポジトリ作業エージェント
前提: `docs/reranker/tasks/NEXT_TASK_PHASE3_CTX.md` / `docs/reranker/reports/PHASE3_CTX_REPORT.md` / `docs/reranker/plans/PLAN_CONTEXTUAL_RERANKER.md`

## 背景（確定した数字）

- 採用 70m（`trackB_v2_continue`, ctx, τ=2.5, max_len=128, cand_cap=30, ONNX fp32）は**品質OK**（hook==clean eval 一致、退行<1.4%、CSΔ +29〜49pt）。
- しかし **CPU 推論（ウォーム・プロセス内・intra=12・動的パディング）で p50 111ms / p95 337ms**。p95 が IME には重い。
- 重い理由は**文脈で入力が約2倍長くなった**から（36→63トークン。モデルは同じ70m）。文脈効果の代金。**文脈は捨てない。**
- 配信は現状 `std::system` で毎回 python 起動＝実用外。

→ **代金を 30m 縮小で取り戻す（文脈は保つ）＋ 常駐デーモンで startup を消す。**

---

## タスク1: 30m 文脈リランカーを学習・検証（本命）

### 1-1. 学習（Modal L4）
- ベース: **`sbintuitions/modernbert-ja-30m`**（採用70mと同アーキ系。任意で `cl-nagoya/ruri-v3-pt-30m` も比較可）。
- データ: **`data/rerank_ctx/train_v2_clean.jsonl`**（掃除済み）。
- 設定: **70m と同一**（epochs 2, bs 512, max_len 128, fp16）。**1変数（モデルサイズ）だけ変える。**
- 70m の Track B は「文脈なしCEから継続」だったが、**30m は文脈なしCEが無いのでベースから学習（Track A 方式）**でよい（A/B は僅差だったため）。
- 出力: `/artifacts/track30m_ctx/`（Volume）。

### 1-2. τ 再スイープ（30m 専用）
- `margin.py` で τ∈{1.5,2.0,2.5} を **clean eval** で固定スイープ。
- 選定ルール（70m と同じ）: **全3セットで全体 regression <2% を満たす中で、min(CS Δ) が最大**の単一 τ。

### 1-3. 評価（clean eval・文脈ON/OFF・CSサブセット）
`eval_contextual.py` で seen/unseen/fresh を **文脈あり/なし両方**、context_sensitive 分離集計。出す数値:
- hit@1(all) / Mozc比 / 退行
- **CS Δ（ON−OFF）** ← 本命。文脈効果が 30m でも残るか
- non-CS コスト

**判定ゲート**:
- **CS Δ が実用的に残る**（目安: 各セット ≥ +15pt、理想は 70m の +29〜49 に近い）
- 全体退行 <2%
- これを満たせば 30m を速度回収の採用候補に。満たさなければ 70m 据え置き＋文脈cap短縮を検討。

### 1-4. ONNX export ＋ parity ＋ レイテンシ再計測
- `export_onnx`（fp32）＋ `parity_check`（Spearman=1.0/MAE≈0）。
- **レイテンシは 70m と同一条件で計測**（`latency_pack.py`、ウォーム・プロセス内・**intra=CPUコア数・padding=longest（動的）・cand_cap分を1バッチ1 forward**）。同じ Windows/WSL CPU で比較可能に。
- 目標: **p50 ~40-55ms / p95 ~120-170ms**（70m の 111/337 から 2〜3倍）。
- 出力: `artifacts/rerank_ctx/eval/latency_ship_profile_30m.json`、`shippable_track30m_selection.json`。

---

## タスク2: 常駐デーモン配信（startup を消す）

現状の `std::system(hook_cmd ...)` は毎回 python 起動＝致命的。**ORT セッションと SentencePiece トークナイザを1回ロードして常駐**させる。

- **デーモン**（Python）: 起動時に ONNX(fp32) ＋ tokenizer(`tokenizer.model`, Llama/SentencePiece。**WordPiece ではない**) ＋ margin_policy をロード。ローカル **UNIX ソケット/名前付きパイプ/localhost TCP** で待受。
  - リクエスト: `{reading, context_prev, candidates[]}` → 整形(`context_clip`) → tokenize → ORT → margin gate(τ) → `{ranked[], overridden, scores}`。
  - `context_clip` と margin は**既存実装を正**（hook と同一結果）。
- **`RerankRewriter`（C++）**: `std::system` を廃し、**デーモンへソケット送信**に置換。
  - **fail-safe**: デーモン未起動/接続失敗/**200ms タイムアウト**時は**何もしない＝Mozc 順**。IME を絶対に止めない。
  - 既定 OFF（`MOZC_RERANK_ENABLED=1`）は維持。
- **利点**: SentencePiece C++ 移植なしで動く（Python のまま）。native ORT は将来の最適化として後回し可。
- **計測**: デーモン・ウォームで end-to-end（IPC込み）を測り、`≈ 推論 + ~1ms` を確認。

---

## 手順の順序
1. 30m 学習（1-1）→ τ スイープ（1-2）→ clean eval（1-3）で **CS Δ が残るか**を先に判定。
2. 残るなら export＋parity＋レイテンシ（1-4）。**速度目標を満たすか**。
3. 並行/次に 常駐デーモン（タスク2）で配信を直し、実機で end-to-end 計測。
4. 30m が「文脈デルタ維持 ＆ 速度目標達成」なら **30m を出荷モデルに切替**、70m は比較用に保持。

---

## 成功基準
- 30m: clean eval で **CS Δ 実用維持（各セット ≥ +15pt 目安）**、全体退行 <2%、単一 τ 確定。
- 30m: ONNX fp32 parity 一致、**p50/p95 が 70m の 2〜3倍速**（目標 p95 ≤ ~170ms）。
- 常駐デーモン: ORT/トークナイザを1回ロードで常駐、`RerankRewriter` がソケット経由、**fail-safe/200ms タイムアウトで Mozc 順**、既定 OFF。
- end-to-end（デーモン・ウォーム）レイテンシが推論＋IPC で説明できる。
- 判定: 30m 採用 or 70m 据え置き＋文脈cap短縮、をデータで結論。

## 制約・注意
- **文脈は捨てない**（速度のために context を空にするのは配信のデグレード時のみ、恒常的にはしない）。
- **1変数だけ変える**（30m は 70m と同一データ・ハイパラ・max_len）。
- **int8 禁止**（順位崩壊）。速度はモデルサイズ・スレッド・動的パディングで取る。
- **τ はモデルごとに再スイープ**（70m の 2.5 を 30m に流用しない）。
- レイテンシは**70m と同一計測条件**（intra=コア数・動的パディング・1バッチ forward・ウォーム・プロセス内）で比較。
- デーモンの `context_clip`・tokenizer・margin は**学習/hook と同一結果**（parity 維持）。
- 既存成果物・70m は保持。新規は別名。
- Modal: 学習 L4 / 評価 T4。

## 成果物
- `/artifacts/track30m_ctx/`（ckpt, train_meta）、`onnx/`（fp32＋tokenizer＋margin_policy）
- `artifacts/rerank_ctx/eval/`: 30m の clean eval（ON/OFF・CS）、`shippable_track30m_*`、`latency_ship_profile_30m.json`
- 常駐デーモン実装（Python）＋ `RerankRewriter` のソケット化差分＋テスト
- `docs/reranker/reports/PHASE3_CTX_REPORT.md` 追記: 30m vs 70m（CSΔ・退行・レイテンシ）、常駐デーモン end-to-end、**出荷モデルの決定**
