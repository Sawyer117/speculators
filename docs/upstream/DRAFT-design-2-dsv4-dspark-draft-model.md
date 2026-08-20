# DRAFT — Design: DeepSeek-V4-Flash DSpark draft model (NOT SENT)

> 状态:草稿,给用户过目用。按 [[no-community-push]] 不外发。
> 依赖 [DRAFT-design-1-expert-parallel-training.md](./DRAFT-design-1-expert-parallel-training.md)。

---

## What it is

A `dsv4_dspark` speculator: the DSpark algorithm (block drafting + a Markov transition head +
a confidence head) on a draft whose backbone mirrors DeepSeek-V4-Flash's decoder layer.

Everything lives in one new directory. Against `main` it is **+2622 lines across 15 files with
zero deletions**, plus three lines of registration in `models/__init__.py`.

```
models/dsv4_dspark/
  config.py    139     core.py      750     weights.py   234
  backbone/
    attention.py 176   moe.py       268     block.py     102
    hyper.py     117   rotary.py    105     norm.py       48
    kernels.py    95   moe_ep.py    177     moe_grouped_gemm.py 248
    moe_compile.py 120
```

## Why it has to look like this

The draft's job is to guess what the target will say. Every structural choice below exists
because the target has it, not because it seemed like a good idea:

- **MLA with q/o dual LoRA and per-head attention sinks.** The target's attention is MLA; a
  draft with ordinary MHA has to learn a different function of the same hidden states.
- **mHC (hyper-connections) in place of the residual stream.** Same reason.
- **256 routed experts + 1 shared, per layer.** This is the expensive one, and it is why the
  companion EP proposal exists. We tried nothing smaller, because the released DeepSeek draft
  has this shape and reproducing its acceptance length was the bar.
- **Full 129,280 vocabulary, no draft-vocab reduction.** The released draft does not reduce,
  and a reduced head trains a mapping the target never uses.

The result is 3 layers but ~21B total parameters, ~1.5B active per token.

## What it touches outside its own directory

Three lines in `models/__init__.py` (registration). Nothing else — no shared file is modified
by this proposal.

## Relationship to the EP proposal

**(2) cannot be trained upstream without (1).** The 256-expert backbone does not fit a pure
FSDP recipe. We would rather say that plainly than land a model definition that nobody can
train.

There is one boundary we would like your call on. Three files in `backbone/` are
parallelism, not modelling:

```
moe_ep.py            177    expert-parallel dispatch / all-to-all
moe_grouped_gemm.py  248    grouped GEMM for the stacked expert weights
moe_compile.py       120    torch.compile wrapper for the expert path
```

Removing them leaves **~2077 lines of pure modelling code**, which is a much easier diff to
review. Having looked at how the project already handles this, we think most of it can come
along and the vendor part cannot:

| | lines | where |
|---|---:|---|
| `moe_ep.py` (expert-parallel all-to-all) | 177 | **upstream** — zero `torch_npu`, plain `torch.distributed` |
| kernel registry + plugin point | ~110 | **upstream** — no vendor name appears in it |
| `_grouped_matmul_torch` (reference, and the parity oracle) | ~30 | **upstream** |
| `moe_grouped_gemm.py`'s NPU half | ~200 | out-of-tree bridge |
| `moe_compile.py` | 120 | out-of-tree bridge |

The reasoning is your own precedent. **#775** ("opt-in `transfer_to_npu`") was closed unmerged
after two days for putting a vendor shim in `src/`; **#589** ("selectable attention backend
(sdpa/eager)") merged because it solved the same class of problem — flex attention unavailable
on Ascend — by adding a portable option instead. And `src/` today has zero direct `torch_npu`
calls and zero `is_cuda` branches, going through `torch.accelerator` throughout. We read that
as: portable in, vendor out.

So the model ships a registry with a pure-torch reference for every heavy op, resolved at call
time, and **is correct with zero accelerated kernels registered** — the reference is also the
parity oracle the kernels are validated against. That turns the Ascend kernels from a
dependency into a pluggable accelerator: nobody needs Ascend hardware to work on this code.

One request attached to that. Shipping only an *interface* leaves an accelerator user having
to know to import a bridge by hand, and an install that forgets is correct but quietly slower
with nothing saying so. A ~15-line `discover_plugins()` scanning a `speculators.kernels` entry
point group would make `pip install` the whole story, with no vendor name anywhere in the
scanning code and no behaviour change when no such distribution is installed. We would propose
it as part of this, but it is equally fine as its own small PR if you would rather look at it
separately.

## Serving

Worth stating because it affects how useful this is upstream.

`vllm/transformers_utils/configs/speculators/algos.py` already registers `dspark` and already
passes through exactly the fields this model needs — `markov_rank`, `markov_head_type`,
`block_size`, `enable_confidence_head`, `confidence_head_with_markov`. It maps them to
`Qwen3DSparkModel`, a hardcoded architecture.

So a checkpoint trained by this proposal is a standard speculators checkpoint that vLLM
*almost* knows how to serve; what is missing is an architecture branch, in the same shape
`eagle3` and `peagle` already have (Llama vs Qwen3 variants). Today we bridge the gap with a
converter in our fork that renames `layers.*` to the released checkpoint's `mtp.*` layout.
That converter is a stopgap and we would rather fix it upstream than ship it — but it is a
vLLM-side change, not a speculators-side one, so it is out of scope here and mentioned only
so the picture is complete.

## Evidence

Trained on 775,965 rows self-distilled from the target at temperature 0, on 8× Ascend NPU
(EP8 + FSDP2) with the hidden states supplied online by a live serving instance of the target.

Measured on the same serving stack, greedy, 5 speculative tokens:

| | gsm8k | math500 | humaneval | mbpp | mt-bench | 5-set mean | non-chat 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| released DeepSeek draft | 4.665 | 4.639 | 4.939 | 4.526 | 3.347 | **4.4232** | 4.6922 |
| this draft (5 epochs) | 4.845 | 4.565 | 4.971 | 4.546 | 3.154 | **4.4162** | **4.7317** |
| | | | | | | 99.84% | **100.84%** |

Ahead of the released draft on the four non-chat datasets; the whole of the remaining gap is
multi-turn chat (3.154 vs 3.347), which we read as a training-data coverage issue rather than
an architectural one.

## Open questions for you

1. Does the project want a model this large in-tree at all? It is a pure addition, but it is
   2622 lines that someone has to be willing to maintain. A plausible alternative is that
   DSV4-scale drafts live out-of-tree and speculators only owns the EP mechanism — we would
   not argue against that.
2. The `backbone/` boundary above.
3. Whether the vendor-specific parts matter to you. Development was on Ascend NPU, but the
   model itself is portable PyTorch; `torch_npu` appears only in two throughput-oriented
   modules, gated by device type, and can be dropped from a first submission.
