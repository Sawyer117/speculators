# DSV4 draft layerwise supervision

This branch adds **training-only auxiliary prediction heads** to the public
DSV4 DSpark implementation. It is a minimal port of the auxiliary-supervision
recipe, not a release of a complete experiment environment or trained weights.
It builds on the existing public `feat/dsv4-dspark` commit
`13ae97a030c5bb80d8f778e2ea998dd7194ee0ee`; that branch is left unchanged.

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
