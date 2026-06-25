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
(serve on card 0, train on 1-7). Config: [`config_qwen3_4b.sh`](./config_qwen3_4b.sh).

### Quickstart — end-to-end (background / nohup, recommended)

Run from the repo root. **Order matters: serve must be healthy before train.**

```bash
# pick where data + outputs live (set in ONE shell; serve & train both read these)
export DATA_DIR=/share/canada_group_folder/dataset/<your-tokenized-arrow-dir>
export OUTPUT_DIR=./outputs/qwen3-4b-dflash-npu

# 0) tokenize a raw jsonl into DATA_DIR (CPU, one-time). Skip if DATA_DIR already
#    contains a prepared Arrow dataset.
source examples/ascend_npu_dflash/config_qwen3_4b.sh
python scripts/prepare_data.py --model "$TARGET_MODEL" \
  --data /share/canada_group_folder/dataset/<raw>.jsonl \
  --output "$DATA_DIR" --max-samples <N> --seq-length 3072 --overwrite

# 1) serve (background; survives SSH disconnect)
bash examples/ascend_npu_dflash/serve_qwen3_4b_nohup.sh
tail -f "$OUTPUT_DIR"/logs/serve_4b_*.log           # wait for "Application startup complete"
curl -s --noproxy '*' http://localhost:8001/v1/models | head      # model listed = ready

# 2) train (background)
bash examples/ascend_npu_dflash/train_qwen3_4b_nohup.sh
tail -f "$OUTPUT_DIR"/logs/train_4b_*.log            # loss should fall from ~3.6

# 3) analyze the log (loss / per-position acceptance / throughput + charts)
python examples/ascend_npu_dflash/analyze_train_log.py "$OUTPUT_DIR"/logs/train_4b_*.log
```

> ⚠️ **Do NOT run the trainer on a bare TTY** (`bash train_qwen3_4b.sh` straight in the
> terminal): a torch_npu fork + rich-logging deadlock hangs a DataLoader worker after a
> few steps. The `*_nohup.sh` wrappers — or any `| tee` / `> log` redirect — make stdout
> non-TTY and sidestep it (details in [`ascend-npu-torch-fork-deadlock.md`](../../docs/deployment/ascend-npu-torch-fork-deadlock.md)).
> The foreground `serve_qwen3_4b.sh` / `train_qwen3_4b.sh` remain for interactive/piped use.

### Env knobs (override inline, or edit `config_qwen3_4b.sh`)

| var | default | meaning |
|---|---|---|
| `DATA_DIR` | shared Arrow | tokenized dataset (train input) |
| `OUTPUT_DIR` | `./outputs/qwen3-4b-dflash-npu` | base dir; `HS_DIR`/`SAVE_DIR`/`logs` derive from it |
| `EPOCHS` | `6` | training epochs (set `1` for a single-epoch run) |
| `USE_OFF_POLICY` | `0` | passes `--use-off-policy-tokens` — **EAGLE3-only; DFlash IGNORES it (no effect on training)** |
| `MAX_MODEL_LEN` | `SEQ_LEN+256` | vLLM served-context cap (keeps KV cache small → faster serve) |
| `GPU_MEM_UTIL` | `0.90` | vLLM HBM fraction |
| `SEQ_LEN` | `3072` | per-sample / per-rank batch token length |
| `VLLM_DP` / `SERVE_CARDS` | `1` / `0` | serve replicas + their cards (HS extraction throughput) |
| `TRAIN_CARDS` / `NPROC` | `1,2,3,4,5,6,7` / `7` | trainer cards + FSDP world size |

> **Device split — recommended: 2 serve + 6 train.** The defaults serve on one card
> (`SERVE_CARDS=0`, `VLLM_DP=1`, `TRAIN_CARDS=1-7`, `NPROC=7`), which usually leaves
> hidden-state extraction as the bottleneck. For faster runs prefer two serve replicas
> (cards 0–1, ~2× HS gen) and six trainer cards — set this in the same shell before
> launching serve & train:
> ```bash
> export VLLM_DP=2 SERVE_CARDS=0,1 TRAIN_CARDS=2,3,4,5,6,7 NPROC=6
> ```

### Example — full open_perfectblend, single epoch (non-regen data)

```bash
export DATA_DIR=/share/canada_group_folder/dataset/open_perfectblend_full.qwen3.seq3072
export OUTPUT_DIR=./outputs/qwen3-4b-dflash-npu-openblend
export EPOCHS=1          # (USE_OFF_POLICY is a no-op for DFlash — EAGLE3-only)
source examples/ascend_npu_dflash/config_qwen3_4b.sh
python scripts/prepare_data.py --model "$TARGET_MODEL" \
  --data /share/canada_group_folder/dataset/open_perfectblend_full.jsonl \
  --output "$DATA_DIR" --max-samples 1420909 --seq-length 3072 --overwrite
# then run serve + train (steps 1-2 above)
```

> `prepare_data` warns `No assistant response spans found` for conversations whose
> assistant turn falls past `--seq-length` (open_perfectblend has ~4.3k-token convs; at
> 3072 they're truncated and dropped). **Expected** — only a small fraction is dropped;
> confirm the kept count with `load_from_disk(DATA_DIR).num_rows`.

TARGET_MODEL points at the resolved HF-cache snapshot; verify a checkpoint with
[`check_ckpt.py`](./check_ckpt.py).
