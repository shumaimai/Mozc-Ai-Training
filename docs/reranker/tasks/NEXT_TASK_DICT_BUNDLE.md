# 次タスク: Mozc-diff 同梱辞書 + 正直な B-inject 再測定

作成: 2026-08-12  
前提: `docs/reranker/plans/PLAN_RERANKER.md` §5.2、`docs/reranker/reports/PHASE2_KICKOFF.md`、`artifacts/rerank/phase2_b_inject_trial.json`

---

## 明示ステータス

**リランク学習データの scale-up は一時停止。** いまの主レバーは **辞書同梱（Mozc-diff）＋候補注入**。CE 追加拡張は追わない。

---

## 68.12% とは何か（ターゲットではない）

| 項目 | 値 |
|--|--|
| 試験 | `phase2_b_inject_trial.json` |
| 辞書 | 同系 wikidata interim `classify_in`（Mozc-diff なし） |
| B rescue | 90.6% |
| hit@1 | **68.12%**（τ_native=2.5 / τ_inj=3.0） |
| regress | 1.73%（no-inject と同一） |

これは **same-family リーク上界**: holdout B の大半が辞書源（wikidata places）と同一ファミリーで、gold が辞書に **構築上** 入る。  
**改善ターゲットでも出荷ゲートでもない。**

---

## 成果物（本番パス）

ビルド: `python -m tools.dict.bundle build ...` または `scripts/_run_dict_bundle_build.sh`

| バンドル | パス | entries | readings | size |
|--|--|--|--|--|
| **production**（出荷候補） | `artifacts/dict/production/` | 170,232 | 167,613 | ~40.8 MB |
| holdout_safe（反リーク評価用） | `artifacts/dict/holdout_safe/` | 168,587 | 166,115 | ~40.5 MB |
| japanpost_only（クロスソース） | `artifacts/dict/japanpost_only/` | 141,509 | 141,091 | ~34.9 MB |
| wikidata_holdout_safe | `artifacts/dict/wikidata_holdout_safe/` | 27,502 | 25,495 | ~5.7 MB |
| wikidata_mozc_diff（リーク参照） | `artifacts/dict/wikidata_mozc_diff/` | 29,146 | 26,993 | ~6.0 MB |

各ディレクトリ:

- `mozc_diff_entries.jsonl` — `{reading, surface, source, category, license_id}`
- `reading_map.json` — `reading → [surface,…]`（IME 注入用ローダ）
- `build_report.json` — before/after・ライセンス・サイズ

**フォーマット方針**: 平らな reading マップ（JSON）。FST/巨大フレームワークは未導入（不要なため）。

### Mozc-diff カウント（production）

| | rows |
|--|--|
| before（wiki+japanpost） | 254,715 |
| dropped already-in-Mozc | 84,060 |
| **after（Mozc-diff）** | **170,232** |
| sources after | wikidata 29,146 / japanpost 141,086 |
| licenses | CC0-1.0 / JapanPost-terms-review-required |

### NEologd

**未同梱（follow-up）。** 取得は現実的だが配布前ライセンス確認が必要。現状は wiki+japanpost のみで進行。

### 実 IME ログ

**リポジトリ内に実変換ログは無い。** 正直な B 測定の次段階として要収集。

---

## 正直な再測定（anti-leak）

モデル: `artifacts/rerank/modernbert70m_ce_v3`  
制約: 注入は **末尾のみ**、τ_native=**2.5**、τ_inj sweep **2.5–4.5**、int8 なし、cand_cap=30  
ランナー: `scripts/_run_dict_bundle_honest_eval.sh`  
要約: `artifacts/rerank/phase2_b_inject_honest_summary.json`

### メトリクス表

| Protocol | Dict | B coverage (gold∈list) | hit@1 best | regress | vs no-inject 45.96% | 判定 |
|--|--|--|--|--|--|--|
| **A source-holdout** | wiki+jp Mozc-diff **excl. holdout golds** | **0.0%** (0/741) | **45.84%** (τ_inj=2.5) | **1.73%** | −0.12pt | NO_SHIP（一般化ゼロ） |
| **B cross-source** | japanpost-only Mozc-diff | **0.27%** (2/741) | **45.96%** | **1.73%** | ±0 | NO_SHIP（本 holdout では効かない） |
| A' wiki holdout-safe | wikidata Mozc-diff excl. golds | **0.0%** | **45.84%** | **1.73%** | −0.12pt | NO_SHIP |
| ref same-family Mozc-diff | wikidata Mozc-diff（gold 含む） | 90.6% | 68.84% (τ_inj=2.5) | 1.73% | +22.9pt | **リーク上界**（旧 68% と同型） |
| legacy leaky trial | classify_in wikidata（diff なし） | 90.6% | 68.12% (τ_inj=3.0) | 1.73% | +22.2pt | **リーク上界**（非ターゲット） |

no-inject 基準（全プロトコル共通）: hit@1 **45.96%** / regress **1.73%** / τ=2.5。

### 読み取り

1. holdout gold を辞書から外すと **B rescue は構築上 ~0**。旧 68% は辞書に gold が入っていたことの反映。
2. japanpost → wikidata-heavy holdout のクロスソースも **ほぼ効かない**（住所語彙と施設 gold の重なりが極小）。
3. holdout_safe では読み共有による **非 gold 注入がわずかに hit@1 を削る**（45.96→45.84）。退行率自体は 1.73% 据え置き。
4. 同梱辞書が **実際にその語を持つクエリ** では製品価値はあるが、それは「一般化性能」ではなく **カバレッジ性能**。評価はプロトコルを分けて書く。

---

## τ_inj 推奨（regress ≤2% ゲート）

| 用途 | τ_inj | 理由 |
|--|--|--|
| **正直プロトコル（現 holdout）** | 注入しない / 無効化 | B 回収なし。非 gold 注入で hit@1 微減のみ |
| **カバレッジ出荷（語彙が辞書に入る前提）** | **3.0**（ネイティブ τ=2.5） | 旧 trial と同じ。regress 1.73% 維持のまま B_covered 回収。τ=2.5 は leaky 上界で hit 最大だが、未検証語への上書きが増えるので **3.0 を既定** |
| より保守 | 3.5–4.5 | overwrite_inject を削る。honest でも微害低減 |

---

## 推奨（出荷判断）

**PAUSE dict deep-dive on accept-holdout → Phase 3**（trial インフラは ship 可、品質クレームは不可）

- **Ship trial**: Mozc-diff バンドル＋末尾注入＋τ_inj=3.0 の実装パスは用意済み（`artifacts/dict/production`）。
- **Do not claim** 68% / 90% B rescue を一般性能として使う。
- **Audit/taxonomy (2026-08-12)**: honest≈0 は lookup バグではない。この holdout では NEologd 無効 → `phase2_b_zero_audit.json` / `phase2_b_manual_taxonomy.md`。
- **Next levers**（CE 拡張・NEologd 即着手は pause）:
  1. **Phase 3**: リランカー IME 統合 + 実変換ログ（`docs/reranker/tasks/NEXT_TASK_PHASE3.md`）— 主レバー
  2. 実ログ上の B を見てから辞書 / NEologd（ライセンス後）
  3. JapanPost 配布ライセンス最終確認（辞書を載せる製品パス用）

---

## コマンド早見

```bash
# CPU: 辞書ビルド（WSL / Git Bash）
bash scripts/_run_dict_bundle_build.sh

# GPU: 正直再測定（docker rocm-torch）
bash scripts/_run_dict_bundle_honest_eval.sh
bash scripts/_monitor_dict_bundle_honest_eval.sh

# 単体
python -m tools.dict.bundle build \
  --classify data/interim/mozc_batch/wikidata/classify_in.jsonl \
             data/interim/mozc_batch/japanpost/classify_in.jsonl \
  --exclude-holdout data/rerank_v3/holdout.jsonl \
  --out-dir artifacts/dict/holdout_safe

python -m tools.rerank.b_inject_eval \
  --holdout data/rerank_v3/holdout.jsonl \
  --dict-entries artifacts/dict/holdout_safe/mozc_diff_entries.jsonl \
  --protocol source_holdout_mozc_diff \
  --ckpt artifacts/rerank/modernbert70m_ce_v3 \
  --device cuda --fp16 --require-cuda \
  --tau-native 2.5 --tau-inject-sweep 2.5,3.0,3.5,4.0,4.5 \
  --out artifacts/rerank/phase2_b_inject_honest_holdout_safe.json
```

---

## 成功基準チェック

- [x] Mozc-diff 成果物 + before/after カウント
- [x] リーク制御した B coverage + hit@1 を公開（≪ 68%）
- [x] 推奨: need more sources / real logs（trial インフラは可）
- [x] τ_inj 推奨（regress ゲート下）
