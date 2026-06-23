# DFlash Qwen3-4B — NPU baseline (pinned)

Reproducible baseline for DFlash draft training of **Qwen3-4B** on Ascend NPU
(vLLM serves the verifier + extracts hidden states online; a separate FSDP trainer
trains the draft on disjoint cards).

> **Pinned tag:** [`npu-baseline-2026-06-23`](https://github.com/Sawyer117/speculators/tree/npu-baseline-2026-06-23)
> All links below point at this tag (a stable ref — it won't drift even as `main` moves on).
> To reproduce: clone, then `git checkout npu-baseline-2026-06-23`.

## References (pinned)

- **Environment setup:**
  https://github.com/Sawyer117/speculators/blob/npu-baseline-2026-06-23/docs/deployment/ascend-npu-conda.md
- **Training scripts + end-to-end how-to-run** (see the Qwen3-4B "Quickstart"):
  https://github.com/Sawyer117/speculators/tree/npu-baseline-2026-06-23/examples/ascend_npu_dflash

## Dataset

- **Raw jsonl:** `/share/canada_group_folder/dataset/open_perfectblend_full.jsonl`
  (`mlabonne/open-perfectblend`, **1,420,909** samples)
- Must be tokenized with `scripts/prepare_data.py` (`--seq-length 3072`) into an Arrow
  dataset before training — it is **not** consumed as raw jsonl. Conversations whose
  assistant turn falls past 3072 tokens are truncated/dropped (expected `No assistant
  response spans found` warnings; only a small fraction).

## How to reproduce

Run from the repo root. **Order matters: serve must be healthy before train.**

```bash
# 0) data + output locations (set in ONE shell; serve & train both read these)
export DATA_DIR=/share/canada_group_folder/dataset/open_perfectblend_full.qwen3.seq3072
export OUTPUT_DIR=./outputs/qwen3-4b-dflash-npu-openblend
export EPOCHS=1 USE_OFF_POLICY=0          # original (non-regen) data -> off-policy OFF

# 1) tokenize (CPU, one-time)
source examples/ascend_npu_dflash/config_qwen3_4b.sh
python scripts/prepare_data.py --model "$TARGET_MODEL" \
  --data /share/canada_group_folder/dataset/open_perfectblend_full.jsonl \
  --output "$DATA_DIR" --max-samples 1420909 --seq-length 3072 --overwrite

# 2) serve (background; survives SSH disconnect). Wait until ready:
bash examples/ascend_npu_dflash/serve_qwen3_4b_nohup.sh
curl -s --noproxy '*' http://localhost:8001/v1/models | head        # model listed = ready

# 3) train (background). Startup log must show: epochs=1 off_policy=0
bash examples/ascend_npu_dflash/train_qwen3_4b_nohup.sh

# 4) analyze the log (loss / per-position acceptance / throughput + charts)
python examples/ascend_npu_dflash/analyze_train_log.py "$OUTPUT_DIR"/logs/train_4b_*.log
```

> ⚠️ Use the `*_nohup.sh` wrappers (or any `| tee` / `> log` redirect). Do **not** run
> the trainer on a bare TTY — a torch_npu fork + rich-logging deadlock hangs a
> DataLoader worker after a few steps (see `examples/ascend_npu_dflash/torch_npu_getenv_deadlock_report.md`).

## Baseline hyperparameters (for the record)

| group | values |
|---|---|
| draft arch | `qwen3`, `--num-layers 5`, `hidden_size 2560` (auto-derived from verifier) |
| DFlash | `--block-size 16`, `--max-anchors 512`, `--target-layer-ids 1 9 17 25 33`, `--mask-token-id 151669` |
| vocab | **full** verifier vocab (151,936) — `--draft-vocab-size` omitted |
| data/batch | `--total-seq-len 3072`, multipack token-budget batching, 7-way FSDP |
| schedule | `--epochs 1`, `--lr 6e-4`, `--loss-fn ce`, `--noise-std 0.05`, `--scheduler-type linear` |
| device split | serve TP=1 on card 0; train on cards 1-7 |
| serve | `--max-model-len 3328` (=SEQ_LEN+256), `--gpu-memory-utilization 0.90`, graph mode |

`--noise-std` and `--scheduler-type` are the **upstream `scripts/train.py` defaults**
(the official `examples/train/dflash_qwen3_8b_sharegpt_online_5k.sh` omits them too).
**`--loss-fn ce`** is set explicitly: it is DFlash's validated/hardcoded default per
[issue #541](https://github.com/vllm-project/speculators/issues/541); PR #542's `kl_div`
default for DFlash is an unvalidated regression (separate upstream fix pending).
