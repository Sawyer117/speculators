# DSV4 target should implement vLLM's standard `SupportsEagle3` aux-hidden interface

> Draft for communicating with the vllm-ascend maintainers. Paste-ready as a GitHub issue / PR discussion. Evidence cited against `main` (`9e0c6ba`) and the DSpark branch (`dspark-dsv4` / #11571).

## Summary

`AscendDeepseekV4ForCausalLM` (vllm-ascend's DeepSeek-V4-Flash target) does **not** implement vLLM's standard `SupportsEagle3` interface. As a result, `--speculative_config method=extract_hidden_states` — and any EAGLE3-style aux-hidden consumer — fails on DSV4 with:

```
RuntimeError: Model does not support EAGLE3 interface but aux_hidden_state_outputs was requested
```

even though vllm-ascend **already ships** the `AscendExtractHiddenStatesProposer` plumbing. This blocks offline target-hidden-state extraction, which is the standard input to draft/speculator training (`speculators`, and vLLM's own `extract_hidden_states`).

Separately, the DSpark PR (#11571) already captures target-layer hidden states for its in-process draft, but through a **bespoke buffer** (`_dspark_hidden_buffer` / `get_mtp_target_hidden_states()`) rather than the standard interface — so the two paths do not compose.

## Environment / repro

- vllm-ascend: `main` (`9e0c6ba`) and the DSpark branch carrying #11571; vLLM `0.23.0`.
- model: DeepSeek-V4-Flash bf16, 2-node TP8/DP2 (PP=1).

Serve DSV4 with the standard aux-extraction flags:

```
--speculative_config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[40,41,42]}}}'
--kv_transfer_config  '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"<path>"}}'
```

Weights load fine; the runner then raises the RuntimeError above during model load.

## Root cause

The runner gates aux-hidden output on the model implementing `SupportsEagle3`:

- `vllm_ascend/worker/model_runner_v1.py` (~L3863): `if not supports_eagle3(self.model): raise RuntimeError("Model does not support EAGLE3 interface but aux_hidden_state_outputs was requested")`

But the registered DSV4 class does not implement it:

- `vllm_ascend/models/deepseek_v4.py`: `class AscendDeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts, SupportsLoRA, SupportsEagle)` — inherits `SupportsEagle` (base), **not** `SupportsEagle3`.
- registered as `DeepseekV4ForCausalLM` (`vllm_ascend/models/__init__.py`), so this Ascend override is what runs.

For contrast, `SupportsEagle3` is the standard, broadly-adopted aux-hidden interface:

- upstream vLLM: **25** model files implement it, including `DeepseekV2ForCausalLM` (`deepseek_v2.py`) — which keeps an `aux_hidden_state_layers` gate (empty by default) and returns `(hidden_states, aux_hidden_states)` from `forward` when it is non-empty.
- upstream vLLM's own `DeepseekV4ForCausalLM` also lacks it (the gap exists upstream too, since DSV4 is new).
- vllm-ascend does not override v2/v3, so at runtime those honor the standard via upstream; **DSV4 is the only overridden model that drops it.**

## Precedent: vllm-ascend's own Qwen3 DSpark already uses the standard interface

The Qwen3 DSpark PR (#11153) is the direct precedent, and it follows the standard:

- #11153 **touches no `models/*.py` file** — it only modifies proposers (`dflash_proposer.py`, `patch_dspark_proposer.py`), the runner, and triton utils.
- It drives target-hidden capture entirely through the standard `SupportsEagle3` machinery: `set_aux_hidden_state_layers(...)`, `get_eagle3_default_aux_hidden_state_layers()`, `_get_eagle3_aux_layers_from_config()` (relying on upstream `qwen3.py` / `qwen3_moe.py`, which implement `SupportsEagle3`).
- It contains **zero** bespoke model-buffer tokens (`_dspark_hidden_buffer` / `get_mtp_target_hidden_states` / `dspark_target_layer_ids`). The proposer stages target hidden in its own buffer (`self._dflash_hidden_states[:num_context] = target_hidden_states`), fed by the runner's standard aux output.

So the DSpark line already established the standard pattern with Qwen3; **DSV4 (#11571) diverged from your own prior PR** by adding a model-side buffer instead of implementing `SupportsEagle3`. The fix below simply brings DSV4 back in line with how Qwen3 DSpark already works.

## Why it matters

The `SupportsEagle3` aux path is the integration point the training/extraction ecosystem consumes:

- `speculators` reads the connector's `hs_*.safetensors` (`hidden_states` + `verifier_last_hidden_states`) to train drafts;
- vLLM's `extract_hidden_states` uses the same aux contract.

Without DSV4 implementing the interface, none of these can obtain DSV4 target hidden states through the standard channel — despite vllm-ascend already shipping the proposer.

## Observation on #11571 (DSpark)

#11571 captures target-layer hiddens for its in-process draft:

- in `DeepseekV4Model.forward`: `if layer.layer_idx in dspark_target_ids: dspark_hiddens.append(hidden_states.mean(dim=1))` → copied into a fixed-address `_dspark_hidden_buffer`, exposed via `ForCausalLM.get_mtp_target_hidden_states()`.

This is a second, bespoke mechanism for the same "emit target hidden states" concern. If DSV4 emitted via `SupportsEagle3` instead, both the in-process draft and offline extraction could share one path (one capture, two consumers).

## Proposed fix

1. **Implement `SupportsEagle3` on `AscendDeepseekV4ForCausalLM`** — add the base class + `set_aux_hidden_state_layers` / `get_eagle3_(default_)aux_hidden_state_layers`, and have `DeepseekV4Model.forward` return `(hidden, aux)` when aux layers are configured, mirroring `DeepseekV2ForCausalLM`. A working, self-contained patch that **coexists with #11571's buffer** (single capture over the union, routed to both consumers) is available on `Sawyer117/vllm-ascend@feat/dsv4-supports-eagle3`; happy to open a PR or fold it into #11571.
2. *(Optional, cleaner)* Migrate #11571's in-process draft to consume the standard aux and retire `_dspark_hidden_buffer`, collapsing the two capture paths into one.

## Questions for maintainers

- Was `_dspark_hidden_buffer` chosen for (a) intra-forward access timing by the draft, (b) cudagraph address stability, or (c) convenience? `DeepseekV2ForCausalLM` implements `SupportsEagle3` under `@support_torch_compile`/cudagraph and returns fresh aux tuples with **no** such buffer — so if the reason is (b), the standard path may already suffice.
- Do you prefer the `SupportsEagle3` wiring as a standalone PR, or folded into #11571?
