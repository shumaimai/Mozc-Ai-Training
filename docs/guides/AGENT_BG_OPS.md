# バックグラウンド作業の管理ルール（エージェント必読）

対象: このリポジトリで長時間ジョブを回すエージェント
関連: `docs/guides/AGENT_MODAL_USAGE.md`

---

## 原則

1. **学習・評価の本体は Modal**（ローカル ROCm は原則使わない）。
2. **同一目的のワーカーを二重起動しない**（Train A/B ローカル と Modal を並行させない）。
3. **`modal run --detach` のローカルクライアントを kill しない**  
   → 過去に `pkill modal run` でクラウドジョブまでキャンセルされた。
4. 止めてよいローカル: `_train_ctx_v2_rocm.py` / `rocm-torch` コンテナ / keepalive / watchdog。
5. 止めてはいけない: `modal run --detach ...`、Modal ダッシュボード上の ephemeral apps。

---

## オーナー表（更新すること）

| 役割 | あるべき実体 | 備考 |
|--|--|--|
| Track A/B 学習 | Modal apps (`mozc-reranker`) | `modal app list` |
| 評価 | Modal `modal_eval.py` | 学習完了後 |
| ローカル GPU | **停止** | `docker ps` に train がいないこと |
| 監視 | 1 エージェントのみ | 二重の relaunch エージェント注意 |

状態確認スクリプト: `scripts/bg_status.sh`

---

## チェックポイント（コスト対策・次ランから有効）

- `train_cross_encoder`: `--save-every N` で `checkpoint_latest.pt`（model+optim+sched）
- Modal 既定: `save_every=200` + 保存のたびに **Volume `commit`**
- 中断後: **同じ `--out`** で再投げる → `--auto-resume` が `checkpoint_latest.pt` から継続
- 現行ラン（既に投げ済み）には未適用。**次の `modal run` から有効**

再開例:
```bash
# 同じ out なら自動 resume
modal run --detach scripts/modal_train.py \
  --out /artifacts/trackA_v2_modernbert70m \
  --train-path data/rerank_ctx/train_v2.jsonl \
  --eval-path data/rerank_ctx/eval_unseen_v2.jsonl \
  --model sbintuitions/modernbert-ja-70m
```

---

## ローカル整理コマンド（Modal は触らない）

```bash
# GPU 学習だけ止める（modal クライアントは殺さない）
docker exec rocm-torch bash -lc 'pkill -f _train_ctx_v2_rocm.py || true'
docker stop rocm-torch

# 禁止
# pkill -f 'modal run'
# modal app stop <id>   # ユーザが明示しない限り
```
