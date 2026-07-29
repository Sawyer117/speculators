# Mooncake hidden-states transfer on Ascend NPU — findings & port plan

**Status:** investigation complete (2026-07-29). Implementation deferred ("明天再做").
**Goal:** let serve nodes and the training node exchange verifier hidden states (HS)
**over the network with NO shared filesystem** — so we can run e.g. **4-serve + 1-train**
where the serve boxes are not on the `/share` NFS mount. Replaces our current
shared-NFS Plan-B (`DSPARK_HS_DIR` self-tap-to-disk).

Related: [`ascend-npu-multinode-hidden-states.md`](./ascend-npu-multinode-hidden-states.md)
(the older HS-by-file-path analysis, whose "NIXL not shipped / no bytes-over-wire" conclusion
this supersedes).

---

## 1. The mechanism (speculators side)

The HS-transfer backend is now **pluggable** (abstraction merged in **#735**):

- `hs_connectors/src/hs_connectors/transfer.py` — `HiddenStatesTransfer` (get_cached / get_generated /
  cache / delete) + `HiddenStatesBackend` (register / add_*_args / from_train_args /
  **`build_kv_transfer_config`** — the backend emits vLLM's `kv_transfer_config`).
- **`FileBackend` / `FileTransfer`** = the current shared-disk path (`cache()` = `shutil.move` →
  `hs_{idx}.safetensors`). This is our Plan-B.
- **`MooncakeBackend` / `MooncakeHiddenStatesConnector`** (`hs_connectors/src/hs_connectors/mooncake_hidden_states_connector.py`
  + `mooncake_store.py`) — a **vLLM KV-connector V1** (`class MooncakeHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA)`),
  the Mooncake-backed sibling of vLLM's `ExampleHiddenStatesConnector`. Serve PUTs
  `{"hidden_states","token_ids"}` into a **Mooncake distributed store** keyed by (sanitized) request id;
  training GETs by key. Transport TCP or RDMA, **zero shared FS**.

**PR status:** #735 MERGED (abstraction, on `main`). Mooncake backend = **OPEN** #605 (prototype) /
**#710 ([Multi-node] over TCP/RDMA — our exact use case)** / #836 (extraction backend) / #811 (async fan-out).
Lives in the standalone `hs_connectors/` package; **not on `main`**, GPU-first.

`MooncakeStoreConfig` (defaults): `protocol="tcp"`, `master_server_address="localhost:50051"`,
`metadata_server="http://localhost:8080/metadata"`, `device_name=""`, `num_writer_threads=16`.
`setup()` does `from mooncake.store import ...` (⚠ the **Store** submodule) and connects to a
Mooncake **master + metadata server**.

---

## 2. What is ALREADY on Ascend (the big unlock — verified in local trees)

vLLM 0.23.0 (`/workspace/vllm-0.23.0`, our DSV4 serve's vLLM) and vllm-ascend
(`/workspace/vllm-ascend` @ 6036507165, 2026-07-03) already have the whole stack:

| Framework piece | vLLM 0.23.0 | vllm-ascend (NPU) |
| --- | --- | --- |
| `SupportsHMA` | ✅ `.../kv_connector/v1/base.py` | ✅ **used by `MooncakeConnector` (l.1317), `MooncakeLayerwiseConnector` (l.690), `AscendMultiConnector` (l.19)** — all `(..., SupportsHMA)` |
| `KVConnectorBase_V1` | ✅ | ✅ driven on NPU model runner: `worker/model_runner_v1.py:2409` `hidden_states.kv_connector_output = …`, `maybe_get_kv_connector_output`, `worker.py` handles `kv_connector_output` |
| `HiddenStateCacheSpec` | ✅ `vllm/v1/kv_cache_interface.py:416` | ✅ `vllm_ascend/utils.py:1652` `isinstance(spec, HiddenStateCacheSpec)` |
| `extract_hidden_states` spec method | ✅ `vllm/config/speculative.py` | ✅ **passing e2e** `tests/e2e/.../spec_decode/test_extract_hidden_states.py` (extracts + saves via safetensors on NPU) |
| Mooncake transport | GPU pip `mooncake-transfer-engine-cuda-13` | ✅ **shipped natively**: `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` (+ `_layerwise` / `_hybrid`), `from mooncake.engine import TransferEngine`, `MooncakeConnectorV1` registered in `.../kv_transfer/__init__.py`, official `pd_disaggregation_mooncake_multi_node` tutorial, a Mooncake column in the support matrix |

⟹ **The three things I feared were hard are all solved on Ascend:** Mooncake transport, KV-connector-V1
driving, and `SupportsHMA`/`HiddenStateCacheSpec`. `ExampleHiddenStatesConnector` (the GPU HS connector)
is present in vLLM 0.23.0 and mirrors exactly what we need.

---

## 3. Remaining port work (small)

### 3a. torch.cuda → torch.npu (9 sites, the async D2H copy path)
In `hs_connectors/.../mooncake_hidden_states_connector.py` (and identically in the GPU
`ExampleHiddenStatesConnector`, l.177-363 — this is the **standard vLLM HS-extraction copy pattern**):

```
160  self._copy_stream: torch.cuda.Stream | None = None
202  self._copy_stream = torch.cuda.Stream()
204  def _get_copy_stream(self) -> torch.cuda.Stream:
207  self._copy_stream = torch.cuda.Stream()
224  ready_event: torch.cuda.Event                 # param annotation
244  with torch.cuda.stream(copy_stream):
245  slot_mapping.to(..., non_blocking=True)
250  torch.empty_like(hidden_states, device="cpu", pin_memory=True)
251  pinned_hs.copy_(hidden_states, non_blocking=True)
276  ready_event = torch.cuda.Event()
```
vllm-ascend does **NOT** use `transfer_to_npu` (no torch.cuda→npu monkeypatch); it has its own
`torch.npu.Stream` / `current_stream()` (`vllm_ascend/utils.py:464`). So port = **mechanical**
`torch.cuda.{Stream,Event,stream}` → `torch.npu.*` (device-agnostic via `torch.accelerator` or a
factory); `pin_memory` / `non_blocking` are NPU-native.

### 3b. Open nuance — Mooncake **Store** vs **P2P TransferEngine**
- speculators uses the Mooncake **Store** (`mooncake.store`): a persistent distributed KV store with a
  **master + metadata server**, async **put/get by key** — right for serve→train where training reads
  asynchronously and possibly later.
- vllm-ascend ships the **P2P TransferEngine** (`mooncake.engine`): producer→consumer direct transfer,
  designed for PD (prefill sends KV to decode, both live).
- **TODO tomorrow:** confirm the Ascend Mooncake build includes the `.store` submodule. If yes → use it.
  If not → either enable it, **or graft the HS-payload store logic onto vllm-ascend's already-NPU-correct
  `mooncake_connector.py`** (reuse its npu streams + `SupportsHMA.request_finished_all_groups` hook, add
  the `{hidden_states, token_ids}` payload put/get). **The graft is likely the cleaner route.**

---

## 4. Recommended path (tomorrow)

1. **Verify the Ascend mooncake `.store` component** exists in the serve env (`python -c "from mooncake.store import ..."`).
2. **Read vllm-ascend `mooncake_connector.py`** to see how it does put/get + the `SupportsHMA`
   `request_finished` hook on NPU — decide port-the-GPU-one vs graft-onto-this.
3. **Use TCP first** (`protocol="tcp"`) — skips RDMA/GPUDirect device issues; TCP transfer is host-byte,
   device-agnostic.
4. Port the 9 `torch.cuda` sites (or graft) → run a Mooncake master → **1-serve + 1-train TCP smoke**
   (train reads HS from the store instead of `/share`) → then scale to 4-serve + 1-train.
5. **Bonus to measure:** whether Mooncake's transfer is faster/more uniform than NFS writes — could
   reduce the current **HS-straggler (~37% of wall-clock)** on the A2 run.

**Do NOT block the current A2 shared-NFS run** — Mooncake is the scaling unlock for a serve topology
off the `/share` mount, not a fix for the running job.

---

## 5. Env / repos (branch-protected experimentation)
`/workspace/vllm-ascend` carries our DSV4 patches (don't clobber its state); `/workspace/vllm-ascend-gh`
is a cleaner clone (2026-07-09). Both freely pullable per user — **work on a branch** to isolate.
`/workspace/vllm-0.23.0` = the pinned serve vLLM (has the full HMA framework).
