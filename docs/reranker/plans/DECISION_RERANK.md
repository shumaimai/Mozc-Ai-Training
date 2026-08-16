# 判断用まとめ — 文脈リランカー（30m）＋実使用ガード

日付: 2026-08-15  
対象: 出荷する／しない／再学習する、の判断  
詳細ログ: `docs/reranker/reports/PHASE3_CTX_REPORT.md`  
仕様: `docs/reranker/tasks/NEXT_TASK_PHASE3_CTX.md` / `docs/reranker/tasks/NEXT_TASK_USAGE_GUARD.md`

---

## いま決めること

| 項目 | 現状 | 判断 |
|--|--|--|
| モデル | 30m 文脈 CE、τ=2.5、ONNX fp32 | 70m に戻すか、30m のままか |
| 実使用品質 | ガードなし **Mozc より −8.1pt**。ガードあり **+0.5pt** | ガード付きで試すか、OFF のままか |
| IME 入れ直し | ガード入り `mozc_server` はビルド済み。Program Files は未入れ替え | 管理者インストールするか |
| 再学習 | 未着手 | 今やるか、実機でもう少しログを積んでからか |
| 既定 | **OFF**（`MOZC_RERANK_ENABLED=1` でオン） | このままか |

**たたき台:** 30m を採用。既定 OFF は維持。デーモンはガード込みで常駐済みなので、実機確認するなら管理者インストール。再学習は「ガードで止血できたあと」の次段で、今すぐは不要。

---

## 出荷構成（決まっているもの）

| 項目 | 値 |
|--|--|
| モデル | `track30m_ctx`（`sbintuitions/modernbert-ja-30m`） |
| 入力 | `読み: {r} [SEP] 文脈: {ctx} [SEP] 候補: {c}`（リテラル ` [SEP] `） |
| 文脈 | `clean_context` 末尾 **50字**（短縮しない） |
| τ | **2.5**（モデル別。int8 禁止） |
| max_len / cand_cap | 128 / 30 |
| 配信 | WSL 常駐デーモン `127.0.0.1:17890`、ORT **intra=12**、OMP=1 |
| フェイルセーフ | デーモン死・**200ms** 超 → Mozc 順 |
| 既定 | **OFF** |

70m（`trackB_v2_continue`）は Volume に残してある。戻す理由はない（30m の方が速く、Wikipedia CS 効果も残る）。

---

## 品質は評価が二つある

Wikipedia では効く。実タイピングでは壊す。それが今回の本丸。

### A. Wikipedia holdout（副評価・従来）

τ=2.5、文脈 ON、全体退行 &lt;2%、CS Δ 全セット ≥ +15pt。

| set | 30m hit@1 | CS Δ vs Mozc | 退行 |
|--|--|--|--|
| seen | 89.36% | **+31.4pt** | 1.56% |
| unseen | 89.98% | **+24.3pt** | 1.51% |
| fresh | 92.38% | **+48.8pt** | 0.83% |

煙: `駅に`+`きしゃ` → **汽車**（上書き、margin 4.51）。`新聞の`+`きしゃ` → **記者**（Mozc 維持、margin 1.07 &lt; τ）。

### B. 実使用ログ（主評価に昇格）

`ime_usage_pairs.jsonl` **186 件**（2026-08-14 の IME セッション）。shown = リランク後 top-1、wanted = 確定面。

失敗の型: 短い読み・空/数字文脈・ありふれた語で、珍字・旧字・カタカナを margin 3〜5 で昇格。τ=2.5 では止まらない。Wikipedia の CS 分布と実入力が違う。

| | shown==wanted | Mozc 単体 | net vs Mozc | 上書き | 助けた / 壊した |
|--|--|--|--|--|--|
| ガードなし | 71.5% | 79.6% | **−8.1pt** | 37 | 6 / **21** |
| ガードあり | **80.1%** | 79.6% | **+0.5pt** | 2 | 1 / **0** |

ガード後、186 件中 **180 件はモデルを呼ばない**。壊した 21 件は消えた。残った助けは 1 件だけ（`きょうぎ` + `全国高等学校情報処理` → 教義）。

リプレイはモデル再スコアなし（当時の上書き結果にガードを後付け）。新しい入力分布では数字が変わる。

---

## ガードの中身

3経路同一（C++ `RerankRewriter` / Python hook / デーモン）。触ってよい読み以外は Mozc 順。

1. **ホワイトリスト 488 語** — Wikipedia CS（top_gold_share&lt;0.7、漢字 gold≥2、content POS）。seed **`きしゃ`**（wiki に無い IME 同音）。block **`きょうかい`**（協会→教会がリプレイ最後の傷害）。
2. 読みのコードポイント長 **≤ 2** → skip
3. 文脈が空、または数字・記号のみ → skip
4. 上書き先がカタカナのみ / 旧字 → Mozc に戻す

文脈は捨てない（読みを絞るだけ）。`MOZC_RERANK_GUARD=0` でオフ（既定オン）。

制限: `きょうかい` 除外は **この 186 件向けの穴埋め**。別セッションで協会/教会が本当に必要なら再学習側の話。

---

## 速度（WSL idle・quiet）

| | p50 | p95 |
|--|--|--|
| 70m 推論 | 112ms | 342ms（200ms ゲート外） |
| 30m 推論 | **51ms** | **128ms** |
| 30m デーモン e2e | **57ms** | **132ms**（IPC 0.30ms） |

30m @ 50字が Pareto（CS Δ を保ったまま p95&lt;200）。cap 短縮はしない。タイムアウト 250ms にはしていない。

ガードで呼び出しが減るので、実使用のもっさり露出も減る。デーモンは常駐・intra=12。

---

## 成果物（判断に使うもの）

実装の詳細パスより、判断に効くファイルだけ。

### 数値

| ファイル | 中身 |
|--|--|
| `artifacts/rerank_ctx/eval/usage_replay_report.json` | ガード有無の net、助け/壊し |
| `artifacts/rerank_ctx/eval/ime_usage_pairs.jsonl` | 186 件の wanted / shown / margin |
| `artifacts/rerank/rerank_eligible_readings.json` | ホワイトリスト 488 語 |
| `artifacts/rerank_ctx/eval/latency_ship_profile_30m_quiet.json` | 30m quiet 51 / 128 |
| `artifacts/rerank_ctx/eval/daemon_e2e_30m_quiet.json` | e2e 57 / 132 |
| `artifacts/rerank_ctx/eval/shippable_track30m_selection.json` | τ スイープ・CS Δ |

### バイナリ・モデル

| ファイル | 中身 |
|--|--|
| WSL `.../track30m_ctx/onnx/cross_encoder_fp32.onnx` | 出荷 ONNX（~141MB、fp32） |
| 同上 `margin_policy.json` | τ=2.5 等 |
| `artifacts/rerank_ctx/ime_server/mozc_server.exe` | ガード入り、SxS マニフェスト済み |
| Modal Volume `mozc-artifacts:/track30m_ctx/` | 学習 ckpt + ONNX |
| 同 `.../trackB_v2_continue/` | 70m 保持 |

### コード（ガード）

| ファイル | 役割 |
|--|--|
| `tools/rerank/usage_guard.py` | Python 判定（ソース・オブ・トゥルース） |
| `mozc_compat/rerank_guard.cc` | C++ 同一ロジック |
| `tools/rerank/phase3_hook.py` `rerank_one` | hook / デーモンが通る所 |
| `mozc_compat/rerank_rewriter.cc` | IME。非対象ならデーモンを呼ばない |
| `scripts/_start_rerank_daemon.sh` | intra=12、ホワイトリスト JSON を WSL にコピー |

### テスト

- Python: `tools.rerank.test_usage_guard` / `test_rerank_daemon` OK
- C++: `bazelisk test //rewriter:rerank_rewriter_test` PASSED
- ライブデーモン確認: `い`+`2` → skip、`駅に`+`きしゃ` → 汽車

---

## まだやっていないこと

- **管理者での IME 入れ替え**（`scripts/_install_rerank_mozc_server.ps1`）。今の Program Files はガード前のリライタの可能性あり。ただしデーモン側 Python ガードは常駐済みなので、IME がデーモンを叩いていれば **品質の止血は既に効く**。C++ 側 skip（TCP 自体を省略）だけ入れ替え待ち。
- **Task 3 再学習**（ありふれた語の Mozc-top1 アンカー、旧字ペナルティ、実ログ込み eval、τ 再スイープ）。
- 文脈を空にする速度ハック、int8、既定 ON、70m への戻し — いずれもやっていない（制約どおり）。

---

## 選択肢

**A. 様子見（推奨に近い）**  
既定 OFF。デーモン＋ガードは置く。実機でもう少し打ち、`ime_usage_pairs` を増やす。入れ替えは「IME でも C++ skip を試す」ときだけ。

**B. ガード付きで実機オン**  
管理者インストール → `MOZC_RERANK_ENABLED=1`。186 件では Mozc 以上。サンプルが小さいので、壊したら即 OFF。

**C. すぐ再学習**  
ガードは止血であって、モデルはまだ Wikipedia 寄り。本命の CS（記者/汽車以外の同音）を増やしたいなら C。今の net +0.5 は「ほとんど Mozc に戻した」結果でもある。

**D. リランカーを棚上げ**  
Wikipedia では効くが実使用では害、ガード後はほぼ Mozc。コストに見合わないなら D。成果物は残る。

---

## 一文

Wikipedia では 30m 文脈リランカーは GO。実使用ではガードなしだと Mozc より悪い。ガードで net は +0.5pt・傷害 0 まで戻した。再学習は任意の次段。既定は OFF のまま。
