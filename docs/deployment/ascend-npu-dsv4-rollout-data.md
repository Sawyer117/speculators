# DSV4 rollout data pipeline (reproduction guide)

End-to-end recipe for producing the DeepSeek-V4-Flash (DSV4) DSpark **draft training data** on
Ascend NPU: from raw prompts → rolled responses → garbage filtering → tokenized Arrow dataset →
(online) hidden-state extraction. This is the **how-to / current-state** doc; the chronological
narrative + dead-ends live in [`ascend-npu-dsv4-worklog.md`](ascend-npu-dsv4-worklog.md), and the
HS-extraction/training side in [`ascend-npu-dsv4-hs-dumper-planB.md`](ascend-npu-dsv4-hs-dumper-planB.md).

## 0. Pipeline at a glance

```
prompts (open-perfectblend, 16 shards)
  └─ ROLLOUT  (DSV4-bf16 serve, greedy, resume-safe)      → out_bf16/rollout_<NN>.jsonl   (ShareGPT conversations)
       └─ CLEAN (detect_garbage.py --clean)               → out_bf16/rollout_<NN>.clean.jsonl
            └─ PREP (prepare_data.py, chat-template)       → arrow/  (input_ids + loss_mask + seq_len)
                 └─ ONLINE HS  (train + HS_DUMP serve)     → hs_<idx>.safetensors (rolling, deleted after read)
                      └─ TRAIN DSpark draft
```

## 1. Environment

- conda env **`dspark-dsv4-base`**: py3.11, torch/torch_npu **2.10.0**, numpy **2.3.5**.
- **CANN 9.0.0** (source `900env_npu.sh`; must put lld/ccec/patch on PATH + the 9.0.0 `nnal/atb/set_env.sh` — else the serve dies with `libatb.so` / `Mki::Dl undefined symbol`).
- **vLLM 0.23.0+empty** (`VLLM_TARGET_DEVICE=empty`, editable) + **vllm-ascend @ `feat/dsv4-hs-dumper`** (compiles the V4 CANN ops; editable — branch switch = `git checkout` + serve restart, NO rebuild).
- **Install layout (per node):** `<base>/dspark_2026/{installation/{vllm-ascend-v4, vllm-v0.23.0}, speculators}`; `<base>=/home/a00652497` on the A2 fleet (a SHARED-account home, login user differs).
- **One-shot deploy (every node):** `git clone -b feat/dspark-confidence-head … speculators && CANN_ENV=<900env> bash speculators/examples/ascend_npu_dflash/setup_dsv4_env.sh`. Prereqs the script fail-loud checks: CANN 9.0.0, `sudo yum install -y patch gcc gcc-c++ make`, conda, 8× NPU.

## 2. Servers

| | A2 (dual-node) | A3 (single-node) |
|---|---|---|
| HW | 2× Atlas 800 A2, 64 GB × 8 each | 1× A3, 128 GB × 8 = 16 logical devices |
| Why | bf16 ≈ 568 GB > 512 GB single-node → **needs 2 nodes** | fits bf16 on one node |
| Parallel | TP8 / DP2, **EP OFF** (cross-node EP16 all-gather DEADLOCKS — the `shm_broadcast` hang) | TP8 / DP2 / **EP16** (intra-node HCCS, no deadlock) |
| Nodes | 115 (head, API) + 116 (worker); HEAD_IP 80.5.5.115 / WORKER_IP 80.5.5.116 | one box per shard, embarrassingly parallel |
| Shared FS | **`/share/canada_group_folder`** (real mount) | **`/home/canada_group_folder`** (symlink-faked per box) |
| Serve script | `serve_dsv4_bf16_dualnode.sh head\|worker` | `serve_dsv4_a3_singlenode.sh` (run under nohup) |
| Load time | ~16 min/node (568 GB) | faster (single node) |

Detailed throughput: [`ascend-npu-dsv4-bf16-dualnode-benchmark.md`](ascend-npu-dsv4-bf16-dualnode-benchmark.md),
[`ascend-npu-dsv4-a3-singlenode-benchmark.md`](ascend-npu-dsv4-a3-singlenode-benchmark.md),
[`ascend-npu-dsv4-rollout-benchmark.md`](ascend-npu-dsv4-rollout-benchmark.md).

## 3. Data source & shards

- Source: **`mlabonne/open-perfectblend`** (ShareGPT-style `conversations`; prompt = first human turn).
- Pre-split once into 16 balanced shards (round-robin): `split -n r/16 -d -a 2 --additional-suffix=.jsonl full.jsonl shards/shard_`. Each shard ≈ **88,807 rows**.
- Base dir: `/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/` (A2) — `shards/` (input prompts), `out/` + `out_bf16/` (rolled outputs), `dflash_online_hidden_states/` (legacy DFlash HS).

## 4. Rollout generation

Roll each shard through the DSV4-bf16 serve (`rollout_a3_shard.sh <SID>` for A3; the same
`response_regeneration/script.py` client against any serve). Key config:

- **Sampling: greedy `temperature=0`** (self-consistent gen/train/eval end-to-end — user's call; tripwire if ever benchmarked sampled). `max_tokens=3072`.
- **Concurrency 64** = 32 seqs/DP-replica = the **validated-clean ceiling**. Above it the bf16 serve KV-overflows and emits HTTP-200 GARBAGE (errors=0 is NOT a quality signal).
- **Quality gate (before rollout):** gsm8k full 1319, temp0, abort if acc < 90%. Measured **96.66%** (bf16 ref ~97.27%). Reads a LOCAL gsm8k parquet (box has no HF CDN).
- **Resume-safe** (`--resume`): skips completed rows by key (`str(uuid or idx)`). **Failed rows are now retried** — `load_seen` skips `metadata.error` rows (fix `675d835`), so a serve that dies mid-run no longer permanently blocks the remainder. Progress bar shows **X/Y + ETA** starting at already-done (fix `9a165bd`).
- Throughput (A3 bf16, measured): ~**1.15 rows/s**, ~**657 gen tok/s** (fluctuates 1.0–1.4 row/s). One 88.8k-row shard ≈ many hours → run under nohup.
- Output row: `{id, conversations:[{from:human,value:prompt},{from:gpt,value:response}], metadata:{idx, finish_reason, …}}`. **Error rows** carry `metadata.error` and only the human turn.

**⚠️ Failure mode seen:** a serve that dies mid-rollout while the client keeps running → the client
races through all remaining rows, each instantly `ConnectionError` → tens of thousands of error rows
written fast (0 gen tokens). The output then *looks* complete (88,807 lines) but is mostly errors.
The `load_seen` fix + a one-time clean (drop error rows) recover it; re-run against a healthy serve.

## 5. Quality filtering (`detect_garbage.py`)

Rollout has NO ground truth, so filter by text-shape heuristics on the gpt turn. Flags:

| flag | catches | knob |
|---|---|---|
| `EMPTY` | error / no response | — |
| `REPEAT` | **degenerate loops** (zlib ratio < `--rep-comp` 0.12 **or** distinct-4gram < `--rep-ngram` 0.25) — the DSV4 KV-overflow garbage; runs to `max_tokens` (finish_reason=length) | rep-comp / rep-ngram |
| `LOW_ALPHA` | non-letter-heavy start | `--min-alpha` 0.30 |
| `TOO_SHORT` | response < `--min-len` chars | `--min-len` 12 |

`--clean <out.jsonl>` writes the NON-flagged rows verbatim (also drops error/empty) — the cleaned
rollout to feed prep.

**Tuning (decided): keep short/numeric answers, drop only real garbage.** Evidence: with defaults,
~1% flagged but the flagged rows are **mostly `finish_reason=stop`** (complete short/numeric answers,
wrongly caught by TOO_SHORT/LOW_ALPHA); only the `finish_reason=length` handful are true REPEAT loops.
So we clean with **`--min-len 0 --min-alpha 0`** → only REPEAT + EMPTY drop.

Measured (shard 00 / 01, `--min-len 0 --min-alpha 0`):

| shard | rows | flagged | clean | REPEAT | EMPTY |
|---|---|---|---|---|---|
| 00 | 88,806 | 42 (0.05%) | 88,764 (99.95%) | 38 | 4 |
| 01 | 88,807 | 32 (0.04%) | 88,775 (99.96%) | 26 | 6 |

(default `--min-len 12 --min-alpha 0.30` would have flagged ~1% — 776/839 TOO_SHORT + 215/244 LOW_ALPHA,
almost all `finish_reason=stop` = valid short answers.)

```bash
for i in 00 01; do
  python examples/ascend_npu_dflash/detect_garbage.py \
    out_bf16/rollout_$i.jsonl --clean out_bf16/rollout_$i.clean.jsonl --min-len 0 --min-alpha 0
done
```

## 6. Tokenize → Arrow (`prepare_data.py`)

Applies the DSV4 chat template → `input_ids` + `loss_mask` (loss on the assistant/gpt turns) +
`seq_len`, saved as a HF Arrow dataset (`save_to_disk`). This Arrow dataset is what the trainer's
`ArrowDataset` loads, and the **source of `loss_mask`** (NOT the HS files).

```bash
python scripts/prepare_data.py \
  --model /share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 --trust-remote-code \
  --data out_bf16/rollout_00.clean.jsonl --data out_bf16/rollout_01.clean.jsonl \
  --seq-length 8192 --num-preprocessing-workers 8 --minimum-valid-tokens 1 \
  --output /share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow
```

- `--data` takes local `.jsonl` (ShareGPT `conversations`) and can repeat for multiple shards.
- `--minimum-valid-tokens 1` drops zero-trainable-token samples (backstop; garbage already gone).

**⚠️ DSV4 chat-template gotcha (resolved).** DeepSeek-V4 ships **NO Jinja `chat_template`** — the
serve renders chat via vLLM's custom Python encoder `vllm/tokenizers/deepseek_v4_encoding.py`
(`encode_messages`, chat/non-thinking mode). So `apply_chat_template` fails ("does not support chat
templates"). Also: `AutoProcessor`/`AutoTokenizer` can't even load the config on older transformers
(`deepseek_v4` unregistered → rope_scaling crash) — needs transformers ≥ ~5.12 (native support; vLLM
0.23.0 still imports, so safe) OR the `PreTrainedTokenizerFast` fallback in `load_processor`. The fix:
pass **`--chat-template examples/ascend_npu_dflash/dsv4_chat_template.jinja`** — a Jinja reconstructed
**byte-for-byte** from `encode_messages` (verified `ALL MATCH` vs vllm-project v0.23.0 in-sandbox):
```
<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜></think>{response}<｜end▁of▁sentence｜>
```
The `{% generation %}` markers around the assistant span drive `return_assistant_tokens_mask` →
`loss_mask` = 1 on the response only. Verify after prep: `frac` of `loss_mask.sum()/seq_len` ≈ 0.2–0.9
(response-heavy), non-zero on every row. (Do NOT use the community mlx `chat_template.jinja` — it has a
double-`</think>` bug and won't match the serve.)

## 7. Next: online HS extraction + training

The Arrow dataset + a live **HS_DUMP serve** feed the online rolling HS pipeline (no disk explosion):
train with `DSPARK_HS_DUMP=1`, `--hidden-states-path == the serve's DSPARK_HS_DIR`,
`on_missing="generate"` / `on_generate="delete"`. See
[`ascend-npu-dsv4-hs-dumper-planB.md`](ascend-npu-dsv4-hs-dumper-planB.md) §7 and
[worklog](ascend-npu-dsv4-worklog.md).
