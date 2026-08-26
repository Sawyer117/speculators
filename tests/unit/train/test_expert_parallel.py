"""Expert parallelism: the contract between a model and the training stack.

Everything here runs in one process. What that can pin is the contract -- which
parameters a model declares rank-local, that the sharding wrapper turns exactly those
into DTensors, that FSDP is handed the declared plan, and that the all-to-all dispatch
computes what the replicated one does. What it cannot pin is a real multi-rank
exchange; that is the evidence in the PR body, not a unit test.
"""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

transformers = pytest.importorskip("transformers")

from torch.distributed.tensor import DTensor  # noqa: E402

from speculators.models.dsv4_dspark.backbone import moe_ep  # noqa: E402
from speculators.models.dsv4_dspark.backbone.moe import (  # noqa: E402
    GroupedExperts,
    _moe_dispatch,
)
from speculators.train import expert_parallel  # noqa: E402
from speculators.train.distributed import (  # noqa: E402
    rank_local_param_keys,
    shard_rank_local_params,
)
from tests.unit.models.test_dsv4_released_layout import (  # noqa: E402
    N_EXPERTS,
    tiny_model,
)


@pytest.fixture(autouse=True)
def _no_leftover_context():
    """The context is process-wide; a test that installs one must not leak it."""
    expert_parallel.reset()
    yield
    expert_parallel.reset()


# --- the context ------------------------------------------------------------------


def test_no_context_means_not_expert_parallel():
    assert expert_parallel.context() is None
    assert not expert_parallel.is_active()


def test_a_single_rank_group_is_not_a_partition():
    expert_parallel.configure(None, rank=0, size=1)
    assert not expert_parallel.is_active()


@pytest.mark.parametrize(("rank", "size"), [(0, 0), (2, 2), (-1, 4)])
def test_an_impossible_group_is_refused(rank, size):
    with pytest.raises(ValueError):
        expert_parallel.configure(None, rank=rank, size=size)


# --- what the model declares ------------------------------------------------------


def test_a_replicated_model_declares_nothing():
    model = tiny_model()
    assert not model.expert_parallel
    assert model.ep_local_param_keys() == set()
    assert rank_local_param_keys(model) == set()


def test_a_model_without_the_hook_declares_nothing():
    assert rank_local_param_keys(nn.Linear(4, 4)) == set()


def test_each_rank_builds_only_its_own_experts():
    expert_parallel.configure(None, rank=1, size=2)
    model = tiny_model()
    ffn = model.blocks[0].ffn
    assert model.expert_parallel
    assert ffn.experts.num_local_experts == N_EXPERTS // 2
    assert ffn.expert_offset == N_EXPERTS // 2
    assert ffn.experts.w1.shape[0] == N_EXPERTS // 2


def test_the_rank_local_keys_are_the_routed_experts():
    expert_parallel.configure(None, rank=0, size=2)
    model = tiny_model()
    keys = model.ep_local_param_keys()
    assert keys == {
        f"layers.{i}.ffn.experts.{w}"
        for i in range(len(model.blocks))
        for w in ("w1", "w2", "w3")
    }
    assert rank_local_param_keys(model) == keys
    # The shared expert is replicated: every rank runs it on every token.
    assert not any("shared_experts" in k for k in keys)


def test_ranks_do_not_build_identical_experts():
    """Each rank's slice draws from its own stream.

    Built directly rather than through ``tiny_model``, whose fixture re-initializes
    every parameter afterwards and would hide this.
    """
    torch.manual_seed(0)
    first = GroupedExperts(8, 16, 2, 0.0, seed=1).w1.detach().clone()
    torch.manual_seed(0)
    second = GroupedExperts(8, 16, 2, 0.0, seed=2).w1.detach()
    assert not torch.allclose(first, second)


def test_a_replicated_build_is_reproducible_from_the_global_seed():
    """Without a seed the experts follow the run's own RNG, as they always have."""
    torch.manual_seed(0)
    first = GroupedExperts(8, 16, 2, 0.0).w1.detach().clone()
    torch.manual_seed(0)
    second = GroupedExperts(8, 16, 2, 0.0).w1.detach()
    assert torch.equal(first, second)


def test_an_indivisible_expert_count_is_refused():
    expert_parallel.configure(None, rank=0, size=3)
    with pytest.raises(ValueError, match="divisible"):
        tiny_model()


# --- what FSDP is handed ----------------------------------------------------------


def test_the_wrap_plan_lists_children_before_their_block():
    model = tiny_model()
    plan = model.fsdp_wrap_plan()
    for block in model.blocks:
        children = [block.ffn.experts, block.ffn.shared_experts, block.attn]
        assert all(plan.index(c) < plan.index(block) for c in children)


def test_partitioned_experts_are_left_out_of_the_wrap_plan():
    expert_parallel.configure(None, rank=0, size=2)
    model = tiny_model()
    plan = model.fsdp_wrap_plan()
    assert all(block.ffn.experts not in plan for block in model.blocks)
    # everything else still is
    assert all(block.attn in plan for block in model.blocks)


# --- the sharding wrapper ---------------------------------------------------------


@pytest.fixture
def single_rank_mesh():
    """A one-rank gloo mesh: ``Shard(0)`` over it is a no-op that still types as a
    DTensor, which is the part under test. Torn down, so the default process group does
    not outlive the test and collide with the next one to want it."""
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_sharding_wraps_exactly_the_declared_parameters(single_rank_mesh):
    expert_parallel.configure(None, rank=0, size=2)
    model = tiny_model()
    keys = model.ep_local_param_keys()

    shard_rank_local_params(model, single_rank_mesh, keys)

    for name, param in model.named_parameters():
        assert isinstance(param.data, DTensor) is (name in keys), name


def test_sharding_is_idempotent(single_rank_mesh):
    expert_parallel.configure(None, rank=0, size=2)
    model = tiny_model()
    keys = model.ep_local_param_keys()
    shard_rank_local_params(model, single_rank_mesh, keys)
    before = {k: model.get_parameter(k) for k in keys}
    shard_rank_local_params(model, single_rank_mesh, keys)
    for k in keys:
        # The second pass leaves the Parameter alone rather than re-wrapping it.
        assert model.get_parameter(k) is before[k]
        assert isinstance(model.get_parameter(k).data, DTensor)


# --- the dispatch -----------------------------------------------------------------


def test_the_grouped_matmul_matches_a_plain_one():
    torch.manual_seed(0)
    counts = torch.tensor([3, 0, 2])
    x = torch.randn(int(counts.sum()), 4)
    w = torch.randn(3, 4, 5)
    got = moe_ep._grouped_matmul(x, w, counts)
    want = torch.cat([x[0:3] @ w[0], x[3:5] @ w[2]])
    assert torch.allclose(got, want, atol=1e-6)


def test_the_grouped_matmul_handles_no_rows():
    out = moe_ep._grouped_matmul(
        torch.zeros(0, 4), torch.randn(2, 4, 5), torch.tensor([0, 0])
    )
    assert out.shape == (0, 5)


def test_the_dispatch_matches_the_replicated_one_when_nothing_is_partitioned():
    """With every expert local the exchange is an identity, so the two must agree.

    This is what pins the routing arithmetic -- the flatten, the per-expert grouping,
    the router weight, the top-k sum -- independently of any collective.
    """
    torch.manual_seed(0)
    model = tiny_model()
    ffn = model.blocks[0].ffn
    x = torch.randn(11, ffn.dim)
    weights, indices = ffn.router(x)

    reference = _moe_dispatch(x, weights, indices, ffn.experts, ffn.n_routed_experts)
    dispatched = moe_ep.moe_dispatch_ep(x, weights, indices, ffn.experts)

    assert torch.allclose(reference, dispatched, atol=1e-5)


def test_the_dispatch_carries_gradients_back_to_the_router():
    torch.manual_seed(0)
    model = tiny_model()
    ffn = model.blocks[0].ffn
    x = torch.randn(7, ffn.dim, requires_grad=True)
    weights, indices = ffn.router(x)

    moe_ep.moe_dispatch_ep(x, weights, indices, ffn.experts).sum().backward()

    assert ffn.router.weight.grad is not None
    assert ffn.router.weight.grad.abs().sum() > 0
    assert ffn.experts.w1.grad is not None
