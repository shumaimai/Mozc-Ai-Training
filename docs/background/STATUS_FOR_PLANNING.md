# Mozc-Ai / PLaMo 現状まとめ（計画用）

最終更新: 2026-08-11  
目的: 別チャット/別計画で次方針を立てるためのハンズオフ用メモ。実装の詳細仕様書ではない。

---

## 1. いま何をやっているか

Mozc ベースの日本語IMEに、AIで変換候補を足す/直す方向を探索中。

直近の実験テーマ:

- 学習データ（主に accept レビュー由来）で LoRA を回す
- ベースモデルを `pfnet/plamo-2-1b` に寄せて ROCm（AMD）で学習
- 学習後に簡易ベンチ・CPUシナリオ評価

ゴール感（当初）:

- IME向けプロンプトで候補を複数返す
- adapter と base を比較評価
- 実IMEに載る候補品質を見る

実際に到達した地点:

- PLaMo LoRA の full 1 epoch は完走
- 小さなベンチと CPU 崩し入力テストまで実施
- ただし評価設計・推論速度・候補生成の質は「実用検証」にはまだ遠い

---

## 2. 環境

| 項目 | 内容 |
|--|--|
| ホスト | Windows + WSL2 (`shuhei`) |
| GPU | AMD RX 7800 XT（WSL `/dev/dxg`） |
| コンテナ | `rocm-torch`（image: `rocm-torch-snap`） |
| 作業マウント | `$HOME/work/mozc-ai-training` ↔ コンテナ `/work/mozc-ai-training` |
| Windows 側リポ | Mozc-Ai-Training のチェックアウト先 |
| 不安定要因 | 学習/推論中に WSL やコンテナが落ちることがある。増分再開前提で運用 |

主要スクリプト（Windows `scripts/` と work 側にコピーして実行）:

- `run_plamo_rocm.sh` / `run_plamo_full_detached.sh` / `run_plamo_full_watchdog.sh`
- `host_plamo_watchdog.sh`
- `bench_incremental.py` / `launch_bench.sh` / `host_bench_watchdog.ps1`
- `cpu_scenario_eval.py`

コア実装:

- `tools/train/lora_sft.py`（PLaMo ROCm 向け stub/patch あり）
- `tools/train/infer.py`（PLaMo は manual greedy decode）
- `tools/train/benchmark.py`
- `tools/dataset/export_train.py`（`build_ime_prompt()`）

---

## 3. データ

- 学習データ: `train_mixed.jsonl` **11,128 行**
- 由来: DeepSeek/Qwen などの accept レビューを IME プロンプト形式へ export
- プロンプト形式（要約）:

```text
日本語入力の変換候補を提案してください。

現在の入力: <reading>
（任意）既存候補（これら以外を提案）: ...
（任意）直前の入力: ...

3つの候補を改行区切りで出力（説明不要）:
```

注意:

- ベンチの v1 / holdout も実質この学習データからの抜き出し
- **真の未見 holdout ではない**
- 最近のベンチでは Mozc 既存候補を空にしているケースが多い

---

## 4. 学習結果（PLaMo LoRA）

| 項目 | 内容 |
|--|--|
| Base | `pfnet/plamo-2-1b` |
| 方式 | LoRA SFT、full 1 epoch |
| ステップ | 1321（`checkpoint-1321` まで存在） |
| Adapter | `/work/mozc-ai-training/artifacts/plamo2_1b_lora_full/adapter` |
| メモ | ROCm で `causal_conv1d` / `mamba_ssm` 欠落を pure-torch stub + patch で回避。遅い |

ざっくり観測:

- デモ読みでは当たることがある  
  - `とうきょうと → 東京都`  
  - `じんこうこきゅうき → 人工呼吸器`  
  - `まるばしら → 丸柱`（正解が `円柱` なら NG）
- Trainer 集計の train_loss は低め、eval_loss はおおよそ 0.6 台だった（厳密比較用というより通過記録）

---

## 5. 評価結果

### 5.1 増分ベンチ（GPU、完走）

成果物:

- `artifacts/benchmark/plamo2_1b_lora_full.json`
- `artifacts/benchmark/summary.json`

| suite | n | hit1/hit3/any | 中身の傾向 |
|--|--|--|--|
| v1 | 20 | **0.20** | 先頭固定。ほぼ `literary_ruby`（青空難語） |
| holdout | 20 | **0.40** | seed=42 シャッフル。学校名/施設名など `place_or_facility` 多め |

補足:

- 生成はだいたい候補1個だけ返りがち（「3つ改行」には弱い）
- GPT2 比較は SentencePiece 不足でスキップ
- base 比較は最終の増分ランでは未完走/未掲載
- WSL 落ち対策で example ごとに JSON checkpoint する増分ランナーを使用

### 5.2 CPU シナリオ評価（完走）

成果物: `artifacts/benchmark/cpu_scenario_eval.json`

条件: PLaMo LoRA / `device=cpu` / `max_new_tokens=16` / 11ケース

| 指標 | 値 |
|--|--|
| 平均応答 | **約 291 秒/回** |
| 中央値 | 約 251 秒 |
| 最小〜最大 | 165〜620 秒 |
| ロード | 約 8.8 秒 |

内容の傾向:

- 正常系の一部は当たる（`とうきょうと`, `じんこうこきゅうき`）
- `おきる → 起息` のように普通の読みでも崩れる
- 崩れた読み:
  - `をきる`（候補なし）→ ほぼエコーバック `をきる`
  - `をきる`（候補: 起きる/切る）→ `切る`（「起きる」には寄らない）
  - `とうきょうとと` → `東京都都`
  - `ををきる` / `ｗをきる` → 入力をほぼそのまま返す
- **CPU では実用外**（遅すぎる）

---

## 6. 当初予定からのズレ

寄せた／縮小した点:

1. 評価が学習データ抜き出し中心（厳密な未見評価ではない）
2. プロンプトに Mozc 既存候補を十分渡していない
3. 複数候補生成が弱い（単一候補化しがち）
4. GPT2 / base 比較が最終結果から欠けた
5. WSL/ROCm 不安定のため、完走優先の増分運用になった
6. 「IMEに載る速度」までは未達

いまの到達点の言い方:

> PLaMo LoRA を AMD/ROCm で回して動かすところまではできた。  
> 品質・速度・評価設計は、次の計画で立て直す段階。

---

## 7. すでに議論した次候補（未決定）

優先度は未決。別計画で選ぶ想定。

1. **Mozc 失敗例を厚くした再学習**  
   - 成功例も混ぜて忘れるのを防ぐ  
   - 崩れた入力も少し入れる
2. **再ランキングモデル**（本命候補）  
   - Mozc 候補を並べ替え/追加するだけ  
   - 自由生成フルIMEより現実的
3. **PLaMo を teacher にした蒸留**  
   - 小さいIME専用 student  
   - 「1から基盤作る」ではなく「小さい専用ネットを作る」
4. **クラウドGPU借用**  
   - ローカル ROCm/WSL 不安定さ回避  
   - ただし大きい生成モデル一択にはしない
5. **辞書を全部叩き込んで1からAI**  
   - 単独では非推奨（暗記機械化しやすい）  
   - 辞書は候補源泉/制約/補助データ向き
6. **真の holdout / 実IMEログ評価**への作り直し

現時点の感触（この実験者メモ）:

- 「1から巨大モデル」は遠回り
- 「辞書全投入で1から」も弱い
- 「Mozc残差（失敗変換）× 再ランク or 小モデル蒸留 × 安定GPU」が筋が良い

---

## 8. 重要パス早見

Windows:

- リポ: Mozc-Ai-Training のチェックアウト先
- このメモ: `docs/background/STATUS_FOR_PLANNING.md`

WSL/コンテナ:

- データ: `/work/mozc-ai-training/train_mixed.jsonl`
- Adapter: `/work/mozc-ai-training/artifacts/plamo2_1b_lora_full/adapter`
- ベンチ: `/work/mozc-ai-training/artifacts/benchmark/`
- CPU評価: `/work/mozc-ai-training/artifacts/benchmark/cpu_scenario_eval.json`
- 学習ログ: `/work/mozc-ai-training/plamo_full.log`
- ベンチログ: `/work/mozc-ai-training/plamo_bench.log`

---

## 9. 別計画で最初に決めるとよいこと

1. モデルの役割: **自由生成**か **Mozc再ランク/残差**か
2. 成功指標: hit@1 だけでなく、遅延・実IME体感・Mozc比改善
3. データ戦略: 失敗例比率、成功例アンカー、崩れた入力の扱い
4. 実行環境: ローカル ROCm 継続か、クラウドGPUか
5. 目標サイズ: オンデバイス/CPU許容か、サーバ推論前提か
6. 評価セットの作り直し: 学習と分離した真の holdout をどう作るか

---

## 10. 一言サマリ

PLaMo-2-1B LoRA はローカル ROCm で学習完了し、小さなベンチでは holdout 0.40 / 難語 v1 0.20。  
デモは一部当たるが、崩れた入力に弱く、CPU推論は約5分/回で実用外。  
次は「もっと大きい生成モデルを1から」より、**Mozcが落とせないケースに特化した再ランク/蒸留**を計画するのがよさそう。
