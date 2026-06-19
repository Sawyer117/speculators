# Qwen3-8B DFlash on Ascend NPU — run scripts

Single-machine, online DFlash training (vLLM serves the target + extracts hidden
states; a separate trainer process trains the draft). Distilled from
[`docs/deployment/ascend-npu-dflash-training.md`](../../docs/deployment/ascend-npu-dflash-training.md).

8 NPUs are split: **serve on 0,1** (2 cards) + **train on 2-7** (6 cards). Two
processes, **one machine** (not two nodes).

## Order

1. **Prepare data once** (CPU, §4 of the guide):
   ```bash
   python speculators/scripts/prepare_data.py \
     --model /share/canada_group_folder/ckpt/Qwen3-8B \
     --data  /share/canada_group_folder/dataset/perfectblend_train_regen.jsonl \
     --output "$WORKDIR/train_data" --seq-length 3072 --overwrite --trust-remote-code
   ```
2. **Terminal 1 — serve** (wait for `Application startup complete`):
   ```bash
   bash examples/ascend_npu_dflash/serve_qwen3_8b.sh
   ```
3. **Terminal 2 — train**:
   ```bash
   bash examples/ascend_npu_dflash/train_qwen3_8b.sh
   ```

## Knobs (override via env, defaults match the guide)

`MODEL` `WORKDIR` `HS_DIR` `DATA` `SAVE` `PORT` `SERVE_CARDS` `TP` `TRAIN_CARDS` `NPROC`

- `HS_DIR` **must be identical** between the two scripts (defaults already match).
- serve defaults to **graph mode**; fall back to eager with `ENFORCE_EAGER=1 bash serve_qwen3_8b.sh`.
- backend on NPU is `--draft-attn-impl sdpa` (flex_attention is unavailable on NPU).
- run with `bash`, do **not** `source` (the scripts guard against it anyway).
