# DSV4 draft layerwise supervision

This branch adds **training-only auxiliary prediction heads** to the public
DSV4 DSpark implementation. It is a minimal port of the auxiliary-supervision
recipe, not a release of a complete experiment environment or trained weights.
It builds on the existing public `feat/dsv4-dspark` commit
`13ae97a030c5bb80d8f778e2ea998dd7194ee0ee`; that branch is left unchanged.

## Start here: links, version and reading order

Share this guide as the entry point; readers do not need to search for the
repository or guess which branch contains the feature:

- **Guide:** https://github.com/Sawyer117/speculators/blob/feat/dsv4-layerwise-supervision/docs/developer/dsv4-layerwise-supervision.md
- **Branch:** https://github.com/Sawyer117/speculators/tree/feat/dsv4-layerwise-supervision
- **Fixed implementation and its diff:** https://github.com/Sawyer117/speculators/commit/8d07ab951646b23ccaf7a0fe9bb6e9ca4373caf1

The branch can receive documentation updates. The full implementation commit
ID is `8d07ab951646b23ccaf7a0fe9bb6e9ca4373caf1`; use it when citing or reproducing
this code. Its diff against the existing public DSV4 base contains the feature
and its tests, so there is no need to read the entire repository first.

Read the diagram and loss below, then follow this order:

| Step | Source | What to check |
| --- | --- | --- |
| 1 | [inter_heads.py](../../src/speculators/models/dsv4_dspark/inter_heads.py) | `IntermediateHead`, chunked CE, and the two-stage weight schedule. |
| 2 | [core.py](../../src/speculators/models/dsv4_dspark/core.py) | Search `inter_heads` and `_inter_state`: capture intermediate states, reuse the main labels/mask, add the weighted loss, and bypass the heads in eval. |
| 3 | [trainer.py](../../src/speculators/train/trainer.py) | Search `set_train_step`: supply the resumed optimizer step to the schedule. |
| 4 | [conversion](../../scripts/convert_dspark_to_vllm.py) and [verification](../../scripts/verify_dspark_conversion.py) | Search `inter_heads.`: remove training-only tensors and verify the export exclusions. |
| 5 | [focused tests](../../tests/unit/models/test_dsv4_layerwise_supervision.py) | Read executable checks for loss/gradients, RNG, eval bypass, schedule and export. |

To obtain exactly the reviewed implementation:

```bash
git clone --single-branch --branch feat/dsv4-layerwise-supervision \
  https://github.com/Sawyer117/speculators.git speculators-layerwise
cd speculators-layerwise
git checkout --detach 8d07ab951646b23ccaf7a0fe9bb6e9ca4373caf1
git rev-parse HEAD
```

This obtains the source only. To train, first have a working DSV4 DSpark baseline
environment, target weights and aligned hidden-state data, then enable the
switches below in that same training command. The inherited Ascend launcher is
[train_dsv4_dspark.sh](../../examples/ascend_npu_dflash/train_dsv4_dspark.sh);
its machine-specific defaults must be replaced with your own paths and runtime
settings. The focused CPU tests do not require the full target-model weights.

## Method

For a three-layer draft:

```text
draft input -> layer 1 -> layer 2 -> layer 3 -> original final prediction
                  |          |
              aux head 1  aux head 2
                  |          |
              teacher CE  teacher CE
```

Each auxiliary head performs mean pooling over mHC streams, RMS normalization,
and two low-rank projections: `hidden_size -> rank -> draft_vocab_size`.
The default rank is 256 (the DSV4 hidden width is 4096).
Both heads use the **same aligned teacher argmax, valid-position mask and
position decay** as the main training path. No teacher token is fed into the
draft input and no intermediate prediction is injected into the next layer.

```text
loss = original_loss + weight(step) * (aux_CE_1 + aux_CE_2)
```

The sum is over the non-final layers; it is not divided by the number of heads.
The original final-head loss is unchanged. Gradients from the auxiliary heads
reach the preceding draft layers. Vocabulary logits are computed in chunks
with activation recomputation to avoid retaining a full token-by-vocabulary
tensor for every head. Heads use a private initialization generator so enabling
them does not consume the global sampling RNG.

## Enable the two-stage recipe

Export these variables before the existing DSV4 training command:

```bash
export DSPARK_INTER_HEADS=aux
export DSPARK_INTER_RANK=256
export DSPARK_INTER_AUX_WEIGHT=0.2
export DSPARK_INTER_AUX_SCHEDULE=step
export DSPARK_INTER_AUX_DECAY_STEPS=6118
export DSPARK_INTER_AUX_WEIGHT_END=0.08
```

`6118` is an experiment-specific optimizer-step boundary, not a universal epoch
length. Before that boundary the auxiliary coefficient is 0.2; at and after it
the coefficient is 0.08. Set it to the desired boundary for your dataset and
batching. The trainer passes its resumed `global_step` to the model before each
training forward. The resolved recipe is saved as `intermediate_supervision`
in the model config.

For a constant-weight ablation use `DSPARK_INTER_AUX_SCHEDULE=const`. Unset
`DSPARK_INTER_HEADS` on a fresh model to leave the baseline unchanged, or set
`DSPARK_INTER_HEADS=off`. When resuming an auxiliary-head checkpoint, retain its
saved settings and optimizer state. This minimal port supports the existing
`fixed-exp-decay` position weighting; it does not add token injection, CPA,
AR auxiliary training or another experimental loss.

## Inference and export

`model.eval()` bypasses all intermediate heads. The existing
`scripts/convert_dspark_to_vllm.py` removes `inter_heads.*`, and
`scripts/verify_dspark_conversion.py` permits only that newly declared
training-only namespace in addition to its pre-existing exclusions.
The serving model therefore has the original backbone and final heads.
There is no extra auxiliary-head computation or auxiliary-head parameter
storage in the converted serving model. This structural fact alone is not a
measured throughput improvement.

## Sliding window 512 is a separate change

The existing training launcher accepts `SWA_WINDOW=512`, which supplies
`--sliding-window 512`. Keep the auxiliary recipe fixed when comparing windows
128 and 512. The draft's serving attention and KV-cache layout must also honor
512; changing only training, or inheriting the target model's window 128 at
serving time, does not evaluate the intended change.

This branch adds the supervision feature. It does not patch a particular
vLLM-Ascend release to implement a new serving-window contract. A longer
window can increase inference cost; compare acceptance length **and** actual
latency/throughput on the same stack, concurrency, prompts and output limits.

## Validation and scope

```bash
python -m pytest tests/unit/models/test_dsv4_layerwise_supervision.py -q
```

The CPU tests cover FP32/BF16 chunked CE and gradients, private initialization
RNG, loss decomposition, unchanged main-path metrics, evaluation bypass,
schedule restoration, and removal of auxiliary tensors at export. The public
port has not been independently retrained or benchmarked on NPU/GPU hardware.
Do not treat historical results from a separately pinned experiment stack as
measurements of this branch.

Files: `src/speculators/models/dsv4_dspark/inter_heads.py`, the small integration
in `core.py`, the trainer step hook, and the conversion/verification exclusions.
All additions use the repository's Apache-2.0 license.
