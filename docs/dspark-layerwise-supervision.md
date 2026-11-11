# DSpark intermediate-layer supervision

This change adds independent low-rank prediction heads after every draft layer
except the last. Each head predicts the **same aligned teacher argmax** as the
final head, using the same anchor validity, loss mask and exponential position
decay. The auxiliary losses are summed and multiplied by the configured weight.
The existing final-head loss and metrics retain their meaning.

The feature is disabled by default. Add this option to an existing DSpark
training command (two or more layers):

```bash
--intermediate-supervision '{"rank":256,"weight":0.2,"decay_steps":6118,"weight_end":0.08,"logit_chunk_size":512}'
```

Choose `decay_steps` for your dataset and batch size; 6118 is an example, not a
universal epoch length. The trainer uses its checkpointed global optimizer step,
so resuming does not restart the schedule. Omitting `decay_steps` gives a constant
weight. This implementation supports `fixed-exp-decay`, not DPACE.

Heads use RMSNorm, a width-to-rank projection and an independent vocabulary
codebook. Vocabulary logits are computed in chunks and recomputed in backward.
They do not inject tokens into the next layer and do not alter inference logits.
`interN_ce`, `inter_aux_loss` and `inter_aux_weight` are logged separately.

Training checkpoints keep the recipe and heads for exact resumption. Before
saving a **separate inference export**, remove the heads:

```python
model.eval()
model.remove_intermediate_supervision()
model.save_pretrained("draft-for-inference")
```

Keep the original training checkpoint for resuming. `--from-pretrained` reads
the saved recipe; it cannot be combined with the new recipe override. A saved
HF config does not necessarily preserve the execution backend: select the
appropriate eager/SDPA backend when loading for a CPU/NPU test.

Validation covers real tiny-model forward/backward, both anchor conventions,
masked teacher-label alignment, zero valid tokens, schedule boundaries, config
serialization, and identical backbone outputs after removing the heads.

```bash
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/models/test_dspark_intermediate.py
```

No dataset, compressed-HS codec, local server paths or training orchestration is
part of this change. Customer-model acceptance-length and NPU performance results
remain to be measured; existing DSV4 results are not a result for this model.
