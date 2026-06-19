# Qwen3-8B DFlash on Ascend NPU — run scripts

Single-machine, online DFlash training (vLLM serves the target + extracts hidden
states; a separate trainer trains the draft). Distilled from
[`docs/deployment/ascend-npu-dflash-training.md`](../../docs/deployment/ascend-npu-dflash-training.md).

8 NPUs split: **serve on 0,1** + **train on 2-7**. Two processes, **one machine**.
The scripts locate the repo automatically (no hardcoded home dir).

## Configure once

Edit [`config.sh`](./config.sh) or override via env. Key vars (with current defaults):

| var | default | meaning |
|---|---|---|
| `TARGET_MODEL` | `/share/canada_group_folder/ckpt/Qwen3-8B` | verifier weights |
| `TRAIN_DATA` | `/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl` | raw jsonl (prepare input) |
| `OUTPUT_DIR` | `./outputs/qwen3-8b-dflash-npu` | base dir; resolved to absolute |
| `DATA_DIR` | `$OUTPUT_DIR/train_data` | tokenized Arrow dataset (prepare output = train input) |
| `HS_DIR` | `$OUTPUT_DIR/hidden_states` | online hidden-state exchange (must match serve↔train; it does) |
| `SAVE_DIR` | `$OUTPUT_DIR/checkpoints` | checkpoints |

## Run (in repo root)

```bash
# 0) prepare data (CPU, once)
bash examples/ascend_npu_dflash/prepare_qwen3_8b.sh

# 1) terminal 1 — serve (wait for "Application startup complete")
bash examples/ascend_npu_dflash/serve_qwen3_8b.sh
#    graph capture errors? fall back to eager:
#    ENFORCE_EAGER=1 bash examples/ascend_npu_dflash/serve_qwen3_8b.sh

# 2) terminal 2 — train
bash examples/ascend_npu_dflash/train_qwen3_8b.sh
```

Notes: backend on NPU is `--draft-attn-impl sdpa` (flex_attention is unavailable on
NPU); `TORCHDYNAMO_DISABLE` is no longer needed (#600). Run with `bash`, not `source`
(the scripts guard against it). `TRAIN_DATA` here is the 10k subset for a quick pass —
point it at the full `perfectblend_train_regen.jsonl` for a full run.
