# DSV4 rollout data generation — fresh A3 handoff runbook

Self-contained deployment guide for generating **DeepSeek-V4-Flash (DSV4) DSpark draft training
data** (the *rollout* stage) on a **fresh Atlas 800 A3 node**. Written to be followed top-to-bottom
by a human or an AI assistant. You only need to run three things: **clone → `setup_dsv4_env.sh` →
`rollout_a3_shard.sh <SID>`**. Everything else (serve bring-up, quality gate, resume) is automated by
the scripts in this repo.

> Scope: this covers ONLY rollout generation (prompts → model responses). Cleaning, tokenization to
> Arrow, hidden-state extraction and training are downstream and owned by the requesting team — you
> just hand back the `rollout_<SID>.jsonl` files. Full pipeline context:
> [`ascend-npu-dsv4-rollout-data.md`](ascend-npu-dsv4-rollout-data.md).

---

## 0. What you must have on the box first

CANN and the large data files are **provided by the owning team / your environment** — this runbook
does not install them.

| Item | Provided by | Where / note |
|---|---|---|
| **CANN 9.0.0** + a source script | your env (installed separately) | You pass its path as `CANN_ENV` (default `/home/a00652497/900env_npu.sh`). It MUST source the 9.0.0 `nnal/atb/set_env.sh` (puts `libatb.so` on `LD_LIBRARY_PATH`) — otherwise the serve dies with `libatb.so` / `Mki::Dl undefined symbol`. |
| host build tools | you (`sudo`) | `sudo yum install -y patch gcc gcc-c++ make` — `patch` is the easy-to-miss one; without it the vllm-ascend build exits 127. |
| conda + 16 NPU visible | you | `npu-smi info` must show 16 logical devices (A3 = 128 GB × 8 cards = 16 × 64 GB dies). |
| **Checkpoint** `DeepSeek-V4-Flash-bf16` (~568 GB) | owning team | `/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16`. On A3 the convention is `/home/canada_group_folder` (symlink-fake it if you have a real shared mount elsewhere). |
| **Prompt shards** | owning team | `/home/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/shards/shard_00.jsonl … shard_15.jsonl` (pre-split, ≈ 88,807 rows each). |
| **gsm8k parquet** (quality gate) | owning team | `/home/canada_group_folder/dataset/gsm8k/main/test-00000-of-00001.parquet`. Fetch once with `curl -kL https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet -o <path>` (the box proxy resolves huggingface.co; `-k` skips its self-signed cert). |

All script paths are overridable via env vars (`ROOT`, `CANN_ENV`, `CONDA_ENV`, `MODEL`, `BASE`,
`SHARDDIR`, `OUTDIR`, `PORT`, `CONC`, …) — the defaults below are the canonical layout.

**Network note (Ascend fleet):** the box sits behind an MITM `http_proxy`/`https_proxy`. The
huaweicloud mirror + github are reachable through it (the setup script uses
`mirrors.huaweicloud.com`). But the proxy will hijack *internal* IPs — any manual `curl` to the local
serve must bypass it: `curl --noproxy '*' http://localhost:7000/v1/models`. The rollout scripts
already pass `--noproxy '*'` internally.

---

## 1. Clone this repo (once per node)

```bash
git clone -b feat/dspark-confidence-head https://github.com/Sawyer117/speculators.git \
  /home/a00652497/dspark_2026/speculators
```
(Override the target dir with `ROOT=<base>/dspark_2026` if your box uses a different home.)

## 2. Deploy the environment (once per node)

One script builds the whole serve stack: the conda env `dspark-dsv4-base` (py3.11, torch/torch_npu
2.10.0, numpy 2.3.5), **vLLM 0.23.0+empty**, and **vllm-ascend @ `feat/dsv4-hs-dumper`** (compiles the
V4 CANN ops). It fail-loud checks CANN/patch/gcc/lld/ccec and that the ATB ops load.

```bash
# ★ set CANN_ENV to THIS node's CANN 9.0.0 source script
CANN_ENV=/path/to/your/900env_npu.sh \
  bash /home/a00652497/dspark_2026/speculators/examples/ascend_npu_dflash/setup_dsv4_env.sh
```

On success it prints the vllm-ascend commit / vLLM / numpy versions — these must be **identical across
all nodes** (version drift is the #1 cause of "works on one box, breaks on another"). If the ATB gate
fails, fix `CANN_ENV` so it sources the node's 9.0.0 `nnal/atb/set_env.sh`, then re-run.

## 3. Place the checkpoint + data (from the owning team)

Ensure these exist (see the table in §0):
```bash
ls /home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16/                       # ~568 GB
ls /home/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/shards/     # shard_00..15.jsonl
ls /home/canada_group_folder/dataset/gsm8k/main/test-00000-of-00001.parquet     # gate
```

## 4. Run the rollout (one command per assigned shard)

`rollout_a3_shard.sh <SID>` is a **single self-contained driver**: it (1) starts this node's A3 bf16
serve (DP2 / TP8 / **EP16**, graph mode), (2) waits until ready, (3) runs a **gsm8k quality gate**
(aborts if accuracy < 90 % — the KV-overflow garbage guard; `errors=0` is NOT a quality signal), then
(4) rolls out the shard (**greedy `temperature=0`, `max_tokens=3072`, concurrency 64, `--resume`**).

```bash
cd /home/a00652497/dspark_2026/speculators
# <SID> = the shard number (00..15) assigned to this box. Run under nohup — this takes many hours.
CANN_ENV=/path/to/your/900env_npu.sh \
  nohup bash examples/ascend_npu_dflash/rollout_a3_shard.sh <SID> > ~/a3_shard<SID>.log 2>&1 &
tail -f ~/a3_shard<SID>.log
```

- **One A3 box per shard, embarrassingly parallel** — no cross-node comms. Different boxes just run
  different `<SID>`. (One box can also do shards sequentially.)
- **Concurrency 64 is the ceiling** (= 32 seqs/DP-replica). Do NOT raise it: above it the bf16 serve
  KV-overflows and returns HTTP-200 **garbage** while reporting `errors=0`. The gsm8k gate is what
  catches a bad serve config before you waste hours.
- **Resume-safe.** If the serve dies mid-run, just re-run the same command: `--resume` skips completed
  rows AND retries previously-failed (`metadata.error`) rows.
- **Throughput** (measured, A3 bf16): ~**1.15 rows/s** (~657 gen tok/s). One 88,807-row shard ≈ 15–20 h
  → always `nohup`.

## 5. Output + handback

```
/home/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/out/rollout_<SID>.jsonl
```
Each line is a ShareGPT conversation: `{id, conversations:[{from:human,value:prompt},
{from:gpt,value:response}], metadata:{idx, finish_reason, …}}`. When done, `wc -l` should be ≈ 88,807.
**Hand these `rollout_<SID>.jsonl` files back to the owning team** — that's the deliverable. (They run
`detect_garbage.py --clean` → `prepare_data.py` → Arrow → online HS → training.)

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| build exits 127 during `setup_dsv4_env.sh` | missing host tool — `sudo yum install -y patch gcc gcc-c++ make`, then re-source CANN and re-run. |
| serve log: `libatb.so` / `Mki::Dl undefined symbol` | `CANN_ENV` doesn't source the 9.0.0 `nnal/atb/set_env.sh`, or a stale CANN 8.5.1 atb is on `LD_LIBRARY_PATH`. Fix `CANN_ENV`, verify in a fresh shell: `python -c "from torch_npu.op_plugin.atb import _atb_ops; print(1)"`. |
| gate aborts: `gsm8k <X>% < 90%` | the serve is producing garbage (bad parallel config / KV overflow). Inspect `out/gsm8k_a3_<SID>.jsonl` + `out/serve_a3_<SID>.log`. Do NOT roll out — the data would be poisoned. |
| manual `curl` to serve returns proxy 504 HTML | the box proxy hijacked the internal IP — add `--noproxy '*'` (the scripts already do). |
| serve won't start / OOM | check `out/serve_a3_<SID>.log`. First bring-up can use `EAGER=1 bash rollout_a3_shard.sh <SID>` (reliable, slower) before the default graph mode. |
| output looks complete (88,807 lines) but mostly errors | a serve died mid-run while the client raced through remaining rows writing `metadata.error`. The `--resume` retry recovers it against a healthy serve; re-run the command. |

## References (same branch)

- [`ascend-npu-dsv4-rollout-data.md`](ascend-npu-dsv4-rollout-data.md) — full end-to-end pipeline (rollout → clean → Arrow → HS → train).
- [`ascend-npu-dsv4-a3-singlenode-benchmark.md`](ascend-npu-dsv4-a3-singlenode-benchmark.md) — A3 serve layout, throughput, EP details.
- Scripts: `examples/ascend_npu_dflash/{setup_dsv4_env.sh, serve_dsv4_a3_singlenode.sh, rollout_a3_shard.sh, detect_garbage.py}`, `scripts/response_regeneration/script.py`.
