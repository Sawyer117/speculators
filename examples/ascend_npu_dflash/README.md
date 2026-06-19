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

## Qwen3-4B (same recipe)

Qwen3-4B is also a 36-layer Qwen3 model, so the DFlash hyperparameters are identical
to 8B (`--target-layer-ids 1 9 17 25 33`, `--num-layers 5`, `--block-size 16`,
`--max-anchors 512`, `--mask-token-id 151669`). Only the verifier weights differ
(`hidden_size` 2560, auto-derived → `fc` = 5×2560), and the default split is TP=1
(serve on card 0, train on 1-7). The **tokenized dataset is shared** with 8B (same
Qwen3 tokenizer), so no separate prepare — `config_qwen3_4b.sh` points `DATA_DIR` at
the same Arrow set.

```bash
bash examples/ascend_npu_dflash/serve_qwen3_4b.sh    # terminal 1
bash examples/ascend_npu_dflash/train_qwen3_4b.sh    # terminal 2
```

Config in [`config_qwen3_4b.sh`](./config_qwen3_4b.sh) (TARGET_MODEL points at the
resolved HF-cache snapshot; verify a checkpoint with `check_ckpt.py`).
