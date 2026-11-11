# Optional MoE FFN for DSpark

This change replaces only the DSpark draft's dense FFN with routed/shared SwiGLU
experts. Attention, RoPE, target-HS conditioning, Markov/confidence heads, loss
alignment and the offline evaluation interface remain the existing DSpark code.
It does not import the DSV4 attention or mHC architecture.

The default remains dense. For a three-layer experiment, add these options to
the existing training command:

```bash
--speculator-type dspark --num-layers 3 \
--draft-moe '{"num_experts":64,"experts_per_token":4,"expert_intermediate_size":1024,"num_shared_experts":1,"score_function":"softmax","normalize_topk":true,"routed_scaling_factor":1.0}'
```

The example mirrors the public Xing4 expert dimensions, not a verified private
customer checkpoint. Adjust the expert budget explicitly before comparing a
three-layer MoE to a five-layer dense draft. For width H and expert width I, the
active FFN parameters per layer are `H*E + 3*H*I*(K+S)`, including the router,
K selected routed experts and S shared experts. This is not a latency estimate.

The router supports softmax, sigmoid and sqrt(softplus), optional top-k
normalization and output scaling. Expert matrices are packed into a small number
of parameters. The portable reference dispatch evaluates selected token/expert
pairs without token dropping or a dense token-by-expert activation tensor. It
is intended for correctness and initial training validation, not a claim of
fused-kernel performance. There is no draft expert-parallel implementation in
this PR; the repository's existing layer FSDP can shard these parameters.

Initialization follows the existing DSpark training path: target embeddings and
output head are loaded as before; the new router and experts are randomly
initialized. This PR does not silently copy the target's MLA weights into the
draft's MHA/GQA projections. A target-expert warm start would need an explicit,
shape-checked checkpoint mapping and a separate validation result.

The MoE recipe and expert weights are saved in the normal draft checkpoint.
`--from-pretrained` restores that architecture and cannot be combined with
`--draft-moe`. A separate vLLM serving adapter is needed before using the new
packed expert layout in an external vLLM implementation; this change targets
the repository's native DSpark model and offline evaluator.

```bash
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/models/test_dspark_moe.py
```

Tests compare routing against an independently written per-token reference,
check input/router/expert gradients, active-parameter accounting, an actual
DSpark forward/backward, save/reload parity, and the unchanged dense default.
Customer NPU training, acceptance length and latency remain to be measured.
Compressed-HS and site-specific dataset adapters are intentionally outside this PR.
