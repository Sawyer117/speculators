# Multi-node hidden-states transport for online DFlash training — state of the problem

Investigation report (2026-06-24). Captures the architecture, the constraints we hit,
the measured facts on our cluster, and the upstream development status — so we don't
re-derive it next time. Companion to `ascend-npu-torch-fork-deadlock.md`.

---

## TL;DR

- **Online DFlash training needs the verifier's hidden states (HS) per token.** In
  speculators these are produced by a **separate vLLM server** and handed to the trainer
  **by file path** (vLLM writes a `.safetensors`, returns the path; the trainer reads it).
- That file/path mechanism means **multi-node training needs the HS files to be reachable
  by every train rank** — either a shared filesystem at the same absolute path, or some
  transport. There is **no "send HS bytes over the wire" mode** in the shipped connector.
- **SpecForge (SGLang) avoids the whole problem** by *embedding the target engine in the
  training process* (one torch world) → HS is produced on the same rank that consumes it,
  never crosses machines. The only cross-machine traffic is the (tiny) draft FSDP gradient
  sync.
- The fast path that keeps a **standalone vLLM server** but still moves HS GPU→GPU over
  RDMA is the **NIXL hidden-states connector** — **planned upstream, not yet shipped**.
  Today vLLM ships only the **disk connector** (a debug-grade "starting point").
- **We are pinned to vLLM v0.20.2** by vllm-ascend 0.20.2rc1, so even when NIXL lands we
  can't get it until vllm-ascend ships a build on a newer vLLM. → For now: **offline +
  subset**, or **co-located / embedded target** (SpecForge-style), not online cross-machine.

---

## 1. The core questions we worked through

1. How do hidden states get from the target (vLLM) to the trainer? → **by file path**, not bytes.
2. Can we do **A-serve / B-train** (serve and train on different machines)? → Yes, **but only with a shared filesystem at the same absolute path**, or an explicit transport.
3. **No shared disk?** → online cross-machine is impossible as-is; options are NFS/sshfs (make a shared mount), `scp`/`rsync` (offline ship), or co-locate.
4. Can we just `scp` per file at train time? → don't hardcode creds; the productized form is `sshfs` (on-demand) or offline `rsync` (batch). Keep transport at the FS/ops layer, not in training code.
5. Why not send HS over the FSDP collective comm? → because the **vLLM server is not a member of the trainer's process group**; FSDP collectives only connect train ranks.
6. Is there a similar connector we can copy? → vLLM has network KV connectors (NIXL/Mooncake/P2pNccl) but they are for **vLLM↔vLLM PD-disaggregation**; the trainer is not a vLLM consumer, so they're not drop-in. The HS-specific connector that ships is **disk-only**.
7. How does **SpecForge** solve multi-node HS? → it doesn't transfer HS; it **embeds the engine** so HS is local per rank.
8. Is **co-location (same cards) required**? → No. **Same process group (same world) is required.** Co-location is one layout; you can disaggregate target/draft onto different ranks *within one world* and reshard HS via collectives.
9. Does vLLM have a model_runner-style embeddable mechanism like SGLang? → **No** (by design). NIXL is vLLM's *alternative* (cross-world RDMA transport), not an embedding mechanism.

---

## 2. How HS actually flows today (speculators + vLLM)

Code paths (this repo):

- Serve: `scripts/launch_vllm.py` runs `vllm serve` with
  `--kv_transfer_config {"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path": <HS_DIR>}}`
  and `--speculative_config {"method":"extract_hidden_states", ...}`.
- Per request, vLLM runs a `max_tokens=1` forward, the connector **writes a `.safetensors`**
  to `shared_storage_path`, and returns the **path string** in
  `kv_transfer_params["hidden_states_path"]` (`src/speculators/data_generation/vllm_client.py`).
- The trainer (an `openai.OpenAI` client) gets the path and **reads it from its own
  filesystem**: `src/speculators/train/data.py:_maybe_load_hs_file` → `safetensors load_file(path)`.
- `on_generate="delete"` (online default) → the trainer unlinks the file after reading.
  So the online `hidden_states/` dir is a small **in-flight working set**, not the dataset.

`ExampleHiddenStatesConnector` is the upstreamed form of the PoC plugin
`fynnsu/vllm-hidden-states-extractor` — a **store-only / `kv_producer`** connector,
**disk-backed, no abstracted backend** (it's explicitly a debug-grade starting point).

**Lock detail:** `.lock` files coordinate producer/consumer (`wait_for_lock`, fcntl.flock).
The read path only flocks **if a `.lock` exists** → at **offline** train time (files
pre-generated, `--on-missing raise`) there are **no locks, zero flock overhead**. flock is
purely an *online + shared-FS* cost.

---

## 3. The multi-node transfer problem

Three distinct topologies — don't conflate:

| Topology | What's multi-node | Online works as-is? | HS path |
|---|---|---|---|
| **A serve / B train** (train single-node) | physically separate, train is 1 node | Yes, **iff shared FS at same abs path** | B reads A's file over the shared mount |
| **Multi-node train** (FSDP across nodes) | the trainer (draft) | Hard | every train node needs the HS files (shared FS / per-node serve) |
| **Multi-node serve** (verifier sharded, e.g. DSv4) | the target serve | Yes, transparent | single endpoint; one trainer reads one path |

DSv4 ran here as **multi-node serve + single-node train** (TP=16 over 2 nodes, headless
rank-1, one endpoint) — which is why the HS-path problem never appeared then.

---

## 4. Measured reality on our cluster (2026-06-24)

- Nodes: **A/serve = 80.5.5.108**, **B/train = 80.5.5.42** (DSv4 pair was 108/109).
- IP data NIC `enp189s0f0`, /22, same subnet; ping ~0.1 ms.
- **Bandwidth (HTTP over this NIC): ~117 MB/s ≈ 1 GbE line rate** — confirmed identical
  over a direct connection and over an SSH tunnel, so SSH wasn't the cap; this looks like
  a genuine 1 GbE link. (Pending `ethtool enp189s0f0 | grep Speed` to confirm vs a faster
  HCCL/RDMA fabric, which collectives would use instead.)
- **firewalld** model: NIC `enp189s0f0` is in the `public` zone; cross-node trust is by
  **source IP in the `trusted` zone** (108/109 were whitelisted → that's how DSv4 worked).
  B (42) was not trusted → its TCP was REJECTed ("No route to host" while ping worked).
  Fix: `firewall-cmd --zone=trusted --add-source=80.5.5.42/32` (runtime).
  Training ports `29500-29600/tcp` (torchrun) + `50000-60000/tcp` (HCCL) are open.
  **→ Strong suspect for the colleague's 45+54 deadlock: nodes must whitelist each other's
  data-net IP in the `trusted` zone, else HCCL/serve ports get REJECTed.**

---

## 5. HS size (real, from on-disk files)

Per token = `num_layers(6) × hidden × 2 bytes (bf16)`. For Qwen3-4B (hidden 2560):
**30 KiB/token**. (6 layers = 5 `target_layer_ids` + last layer.)

Measured on a live online run (18 in-flight files, `(T,6,2560)` bf16 confirmed):
- mean **525 tok → ~15.4 MiB/sample**; median 387; max-in-flight 1206; one sample as small as 85 tok (=2.5 MiB).
- Max per sample (full 3072 seq) = **90 MiB** (8B = 144 MiB).

Storage projections (4B, at ~15.4 MiB/sample; full-length upper bound in parens):

| Samples | ~avg | full-len upper bound | 1 GbE one-shot transfer |
|---|---|---|---|
| 10k | ~0.15 TB | 0.9 TB | ~22 min |
| 50k | ~0.75 TB | 4.5 TB | ~1.9 h |
| 1.42M (full) | ~21 TB | 133 TB | infeasible |

Caveat: the 18-file working set is biased short (max 1206 << 3072 cap, long tail not
sampled) → use the **tokenized dataset `seq_len` column** for the authoritative mean.

---

## 6. Why online over 1 GbE is bandwidth-starved

Per step, multipack fills ~`total-seq-len = 3072` tokens **per rank** → **90 MiB/step/rank**
of HS regardless of average sample length. At ~0.6 s/step that needs **~157 MB/s per rank**
> 1 GbE (117 MB/s) **already for one rank**; ×7 train ranks ≈ **1.1 GB/s** ≈ 9× over 1 GbE.
Average sample size only affects **total** volume (→ storage), not the **per-step rate**.
**Conclusion: on a 1 GbE IP link, online cross-machine HS streaming can't keep up.** Offline
pays the same total transfer **once** (then reads local at GB/s) and decouples from serve.

---

## 7. SpecForge's answer: embed the engine (one world)

`sgl-project/SpecForge`, `scripts/train_dflash.py` + `specforge/modeling/target/dflash_target_model.py`:

- `--target-model-backend {sglang, hf}` — **both run in-process**:
  - `hf`: `AutoModelForCausalLM(..., output_hidden_states=True)` on the local device; pick
    `capture_layer_ids` + concat. Pure torch/HF.
  - `sglang`: imports `sglang.srt...`, builds a `ForwardBatch` with
    `CaptureHiddenMode.FULL`, calls `self.model_runner.forward(...)` — **SGLang as a
    library**, not an HTTP server.
- Training loop: `target_output = target_model.generate_dflash_data(...)` →
  `hidden_states.to(device)` → draft forward. The target is **frozen** and **not** FSDP-wrapped;
  **only the draft is wrapped in FSDP** (`train_dflash.py`).
- `specforge/distributed.py`: one `init_process_group`; a `(dp, tp)` device mesh for the
  target and a separate `(draft_dp, sp)` mesh for the draft, **over the same ranks/world**.

**Therefore HS never crosses machines:** each rank runs the target forward on its own data
shard → HS is a local tensor → trains the draft locally. Multi-node = data sharding + draft
FSDP gradient sync (small, over the fast fabric). This is the "共卡" (card-sharing): target
inference + draft training share the same cards because they're in the same process.

---

## 8. The key distinction: same-cards vs same-world

- **Co-location (same physical cards, 共卡) is NOT required.**
- **Same process group (same world) IS required** to use torch collectives (HCCL/NCCL
  all-to-all / send-recv) for HS resharding/transfer.
- Within one world you may put target on some ranks and draft on others and reshard HS via
  collectives (a memory-vs-comm tradeoff): co-located = HS mostly local, min comm, but both
  models per card (HBM pressure); disaggregated-within-world = less HBM/card, but all HS
  moves over collectives (fine over a fast fabric).
- Collectives run over the **HCCL/RDMA fabric**, NOT the 1 GbE IP net — that's why
  same-world transport is fast even cross-machine (pending fabric-speed confirmation).

**EP16 co-location config model** (the "16 cards full of serve, where does train go?"):
training does **not** get separate cards — it runs the **draft `(draft_dp × sp)` over the
SAME 16 ranks**; the partition is in **HBM** (serve `gpu-memory-utilization` < 1 to leave
room for draft + optimizer + activations) and in **time** (target forward → draft fwd/bwd
per step), **not** in card allocation. Binding constraint at this scale is **HBM**, tuned
via serve mem-util, draft SP, gradient checkpointing, optimizer offload.

---

## 9. vLLM vs SGLang: connector vs embedding (two philosophies)

| | SGLang / SpecForge | vLLM (speculators) |
|---|---|---|
| Engine form | **embedded as a library** in the training world | **standalone server**, its own world |
| HS export | same world → native collectives / local | **KV-connector framework** (the boundary) |
| Coupling | tight (one world) | loose (separate worlds, bridged by a connector) |
| Fast GPU→GPU HS | native (collectives over fabric) | **needs the NIXL connector** (cross-world RDMA) |

vLLM lacks a clean SGLang-style embeddable `model_runner` **by design** — it bets on the
connector route. **NIXL is the cross-world transport that makes loose coupling fast; it is
not an embedding mechanism.** So "disaggregate + collective-pull HS" only works if the
engine is in the trainer's world (SGLang-embed); with a standalone vLLM server you must
bridge worlds → that bridge is NIXL.

---

## 10. Upstream development status (as of 2026-06)

- **vLLM RFC #33118 "Hidden States Extraction"** — *closed* (design). Existing connectors:
  disk + CPU-offload. Network/NCCL/RDMA/async **not** listed done.
- **vLLM PR #33736** — *merged 2026-03-02*, in-tree `extract_hidden_states`
  (`ExtractHiddenStatesProposer/Model` + config) with **only the disk connector**
  (`ExampleHiddenStatesConnector`) — self-described "not the most performant … a starting
  point for future hidden states connectors". **Present in v0.20.2** (what we run).
- **speculators RFC #335** — *closed*; adopted vLLM-native extraction (path references,
  online/offline/hybrid). Shipped in **speculators v0.5.0**. Explicitly names planned work:
  *"a Nixl Connector that transfers hidden states directly from vllm (GPUs) to the training
  process (GPUs)."*
- **NIXL hidden-states connector: NOT merged.** Disk connector currently has **blocking
  writes**; async writes are "active efforts". Latest vLLM = **v0.23.0** (2026-06-13).
- **SpecForge**: sglang-embedded; `hf` + `sglang` DFlash backends; co-location. See RFC
  **#412 "dLLM (DFlash) Online Training in SpecForge"**.

Links: vLLM #33118, vLLM #33736, speculators #335, SpecForge #412; blog
`vllm.ai/blog/2026-03-30-extract-hidden-states`; NixlConnector usage guide.

---

## 11. The NPU constraint (why we can't just wait/upgrade)

vllm-ascend **0.20.2rc1 pins vLLM v0.20.2** (newer vLLM breaks vllm-ascend's `mla.prefill`
patch). So even when the NIXL HS connector lands in vLLM main, we **cannot** use it until
vllm-ascend ships a build on that newer vLLM. **Double dependency → not a near-term path on
NPU.** v0.20.2 already has the full disk-based extraction, so upgrading buys nothing for
this feature until both move.

---

## 12. Options for us (ranked by near-term practicality on NPU)

1. **Offline + subset** (recommended now): pre-generate HS for a 10k–50k subset
   (`scripts/data_generation_offline.py`, wrapped in `pregen_hs_qwen3_4b.sh`), train
   reading local files (`--on-missing raise`). Kills flock, serve-coupling, and the 1 GbE
   per-step bottleneck. Storage ~0.15–0.75 TB. If train is on B, one-shot `rsync` (~min–hrs).
2. **Co-located / embedded target** (SpecForge-style): target forward in the trainer's
   world → HS local, multi-node = plain FSDP. Best long-term for online multi-node.
   - `sglang-ascend` embedded — needs sglang-ascend to support the target (big-MoE/EP).
   - embedded plain HF/torch target — works today on torch_npu, but **no vLLM/sglang EP/
     kernel optimization** → slow for large MoE; fine for 4B/8B dense.
3. **Standalone vLLM server + NIXL connector** — the "ideal" disaggregated path; **wait for
   upstream** (NIXL HS connector + vllm-ascend uptake). Hand-rolling the cross-world bridge
   is the NIXL-level effort, not worth it.
4. **Online cross-machine via the file path + a transport** (the stopgap we built, §13) —
   works without a shared FS but is 1 GbE-bound (see §6); only sensible for small scale.

---

## 13. The stopgap we built this session (env-gated, fork-only, NOT pushed)

For "online A-serve / B-train without a shared filesystem":

- `src/speculators/train/data.py`: `_maybe_load_hs_file` gains an **`HS_FETCH_BASE`** env
  gate — when set, the trainer pulls each HS file from a sidecar over HTTP (pooled
  keep-alive `requests.Session`) instead of reading a shared path; `on_generate` local
  move/unlink is skipped in that mode. **Default (unset) = byte-for-byte unchanged.**
- `examples/ascend_npu_dflash/hs_sidecar.py`: an aiohttp file server on the serve box that
  streams (1 MiB chunks) and deletes-after-send, path-traversal-guarded, optional
  `HS_SIDECAR_TOKEN`.
- Run: serve box `HS_DIR=... PORT=9009 python hs_sidecar.py`; trainer
  `HS_FETCH_BASE=http://A_IP:9009 bash train_qwen3_4b_nohup.sh`.
- **Caveat:** it rides the 1 GbE IP net → §6 applies (per-step rate too high for multi-rank).
  It's a fork-only stopgap to be retired once the upstream NIXL connector lands. Prefer
  offline (§12.1) for anything beyond small scale.

Also created: `pregen_hs_qwen3_4b.sh` (offline HS pre-generation wrapper).
