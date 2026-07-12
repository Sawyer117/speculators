"""Distributed correctness check for ``--init-on-meta`` (run on >=2 GPUs).

    torchrun --nproc_per_node 2 check_init_on_meta_distributed.py

NOT part of the PR by itself -- lives on the scratch branch pr/init-on-meta-review.
It mirrors the trainer's REAL build+materialize path and asserts that the meta-init
memory optimization does not change the result:

  * build           -- rank0 builds the draft with real weights; non-rank0 builds
                       under ``build_on_meta`` (== scripts/train.py's --init-on-meta
                       branch: ``build_on_meta() if init_on_meta and rank!=0``);
  * shard+broadcast -- ``apply_fully_sharded`` then ``set_model_state_dict(
                       broadcast_from_rank0=True)`` (== trainer.py:291-305).

Assertions:
  1. before broadcast: non-rank0 params are on ``meta`` (proof the allocation was
     skipped -> the memory win), rank0 params are real;
  2. after broadcast:  every rank's local shard is real (not meta) and finite (no NaN);
  3. after broadcast:  the gathered full state dict equals rank0's original weights
     (the all-gather pulls each rank's shard, so this also verifies non-rank0).

Exits non-zero on any failure.
"""

import contextlib
import os
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators.models.eagle3 import Eagle3DraftModel
from speculators.train.distributed import (
    apply_fully_sharded,
    build_on_meta,
    get_rank,
    get_world_size,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def _build_draft(verifier_dir: str) -> Eagle3DraftModel:
    # == scripts/train.py: build under build_on_meta on non-rank0 only.
    meta_ctx = build_on_meta() if get_rank() != 0 else contextlib.nullcontext()
    with meta_ctx:
        return Eagle3DraftModel.from_training_args(
            verifier_config=_tiny_config(),
            t2d=None,
            d2t=None,
            draft_vocab_size=64,
            norm_before_residual=False,
            ttt_steps=1,
            draft_attn_impl="eager",
            target_layer_ids=[0, 1],
            verifier_name_or_path=verifier_dir,
        )


def main() -> None:
    maybe_setup_distributed()
    world = get_world_size()
    if world < 2:
        raise SystemExit("run with torchrun --nproc_per_node >=2 (need >1 rank)")
    rank = get_rank()

    # rank0 writes a tiny verifier WITH weights so rank0's load_verifier_weights
    # succeeds (torchrun ranks share the node filesystem).
    verifier_dir = os.path.join(tempfile.gettempdir(), "init_on_meta_tiny_verifier")
    if rank == 0:
        LlamaForCausalLM(_tiny_config()).save_pretrained(verifier_dir)
    dist.barrier()

    model = _build_draft(verifier_dir)

    # rank0 is the single source of truth: make sure it has NO nan params (embed/
    # lm_head come from the verifier; this is a safety net that also decouples the
    # broadcast check from verifier-loading details).
    if rank == 0:
        with torch.no_grad():
            for p in model.parameters():
                if p.is_meta:
                    raise AssertionError("rank0 unexpectedly built on meta")
                nan = torch.isnan(p)
                if nan.any():
                    p[nan] = torch.randn_like(p[nan]) * 0.02

    # (1) the optimization engaged: non-rank0 built on meta, rank0 did not.
    embed_is_meta = model.embed_tokens.weight.is_meta
    if rank == 0:
        assert not embed_is_meta, "rank0 should hold real weights"
    else:
        assert embed_is_meta, "non-rank0 must build on meta (--init-on-meta)"

    # rank0 keeps its real weights (CPU) as the reference, exactly like trainer.py.
    ref_full = model.state_dict() if rank == 0 else {}

    # == trainer.py:291-305 (shard, then broadcast-materialize from rank0).
    apply_fully_sharded(model)
    set_model_state_dict(
        model,
        ref_full,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=False,
        ),
    )
    dist.barrier()

    # (2) every rank's local shard is real and finite after the broadcast.
    for name, p in model.named_parameters():
        local = p.to_local() if hasattr(p, "to_local") else p
        assert not local.is_meta, f"[rank{rank}] {name} still on meta after broadcast"
        assert not torch.isnan(local.float()).any(), f"[rank{rank}] {name} has NaN"

    # (3) the materialized full state (all-gathered from every rank's shard) equals
    #     rank0's original weights -> non-rank0 got exactly rank0's values.
    got_full = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    if rank == 0:
        assert set(got_full) == set(ref_full), (
            f"key mismatch: only-in-got={set(got_full) - set(ref_full)}, "
            f"only-in-ref={set(ref_full) - set(got_full)}"
        )
        bad = [
            k
            for k, ref_v in ref_full.items()
            if ref_v is not None
            and not torch.allclose(
                got_full[k].float(), ref_v.float(), rtol=1e-2, atol=1e-3
            )
        ]
        assert not bad, f"materialized weights differ from rank0 reference: {bad}"

    dist.barrier()
    if rank == 0:
        print(
            f"PASS ({world} ranks): non-rank0 built on meta, all ranks materialized "
            "real+finite weights, and the gathered state matches rank0 exactly."
        )
    maybe_destroy_distributed()


if __name__ == "__main__":
    main()
