# DSpark intermediate-layer supervision

This change adds independent low-rank prediction heads after every draft layer except the last. Each head predicts the **same aligned teacher argmax** as the final head, using the same anchor validity, loss mask and exponential position decay. The auxiliary losses are summed and multiplied by the configured weight. The existing final-head loss and metrics retain their meaning.

The feature is disabled by default. Add this option to an existing DSpark training command (two or more layers):

```bash
--intermediate-supervision '{"rank":256,"weight":0.2,"decay_steps":6118,"weight_end":0.08,"logit_chunk_size":512}'
```

Choose `decay_steps` for your dataset and batch size; 6118 is an example, not a universal epoch length. The trainer uses its checkpointed global optimizer step, so resuming does not restart the schedule. Omitting `decay_steps` gives a constant weight. This implementation supports `fixed-exp-decay`, not DPACE.

Heads use RMSNorm, a width-to-rank projection and an independent vocabulary codebook. Vocabulary logits are computed in chunks and recomputed in backward. They do not inject tokens into the next layer and do not alter inference logits. `interN_ce`, `inter_aux_loss` and `inter_aux_weight` are logged separately.

Training checkpoints keep the recipe and heads for exact resumption. Before saving a **separate inference export**, remove the heads:

```python
model.eval()
model.remove_intermediate_supervision()
model.save_pretrained("draft-for-inference")
```

Keep the original training checkpoint for resuming. `--from-pretrained` reads the saved recipe; it cannot be combined with the new recipe override. A saved HF config does not necessarily preserve the execution backend: select the appropriate eager/SDPA backend when loading for a CPU/NPU test.

Validation covers real tiny-model forward/backward, both anchor conventions, masked teacher-label alignment, zero valid tokens, schedule boundaries, config serialization, and identical backbone outputs after removing the heads.

```bash
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/models/test_dspark_intermediate.py
```

## Native training validation

On 2026-11-11, commit `1ba7580145a2caf3ad21b2287bfce4459c51240e` passed a 16-Ascend-NPU test using an independently installed checkout and the native `scripts/train.py`: train 50 optimizer steps, save, then start a new process and resume to 100. Import and dependency audits passed. Both checkpoints had finite model and optimizer tensors; optimizer and scheduler counters matched 50 and 100. The auxiliary heads had nonzero optimizer moments. Their loss weight correctly changed from 0.20 to 0.08 at global step 75, across the process restart. The first resumed LR was `5.33e-5` in the rounded log, matching the saved `5.327015646150716e-5`.

Main training loss was finite throughout (first/last logged values: 35.534 / 3.157). The test used a 5-layer dense draft with four auxiliary heads at width 4096, vocabulary 129280, sequence length 256 and block size 7, with 16 distinct training samples repeated to 800 rows and 16 held-out functional samples. These are locally available DSV4 hidden states, not the customer's private checkpoint/data. Native teacher normalization also differs from the original DSV4 benchmark. This establishes the training, saving and recovery path, not acceptance length, throughput or quality gains.

The [sanitized validation summary](dspark-layerwise-validation.json) records exact versions, counters and source-receipt hashes. This branch also includes a small checkpoint fix: restoring scheduler state must restore each optimizer group's LR before the first resumed update. A regression test checks exact equality with an uninterrupted update.

No dataset, compressed-HS codec, local server paths or training orchestration is part of this change. Customer-model acceptance length and NPU performance remain to be measured.
