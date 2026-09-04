"""A matrix whose rows do not divide across the mesh must still survive Muon's gather.

WHY THIS EXISTS. `DistributedMuon` orthogonalizes a row-sharded 2D parameter by
gathering it, so it has to rebuild a DTensor from the local shard. `DTensor.from_local`
without `shape=`/`stride=` INFERS the global shape as ``local_shape * mesh_size``, which
is correct only for an even split. `hc_head.hc_fn` on the DSV4 draft is
``[hc_mult, hc_mult * hidden] == [4, 16384]`` on an 8-wide mesh: ranks 0-3 hold one row
each and ranks 4-7 hold none, so the inference yields ``[8, 16384]`` on the first four
and ``[0, 16384]`` on the last four. The ranks then disagree on the size of the
all-gather they are jointly issuing and the first four block forever -- the exact
{0,1,2,3} vs {4,5,6,7} split that cost several days of training slots.

The regression is invisible on any evenly-divided parameter, and every other matrix in
that draft divides evenly, so the test parameter is deliberately chosen NOT to.

Runs on 4-rank gloo/CPU, so it needs no accelerator.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Global row count on purpose NOT divisible by the mesh: 4 ranks, 2 rows.
# Ranks 0-1 get one row each; ranks 2-3 get an empty shard.
ROWS, COLS, WORLD = 2, 8, 4

# The module is loaded by PATH, not as ``speculators.train.muon_distributed``.
# Importing the package pulls in transformers (and, on an NPU box, torch_npu) for a test
# that needs one file and four CPU processes -- the sibling wiring test avoids the same
# import for the same reason. It also sidesteps `spawn` starting a fresh interpreter
# that does not inherit this one's sys.path.
MUON_PY = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "speculators"
    / "train"
    / "muon_distributed.py"
)


def _load_muon():
    spec = importlib.util.spec_from_file_location("_muon_under_test", MUON_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _worker(rank: int, port: str, out) -> None:
    # Imported here, not at module scope: `torch` is behind importorskip above, and
    # this body runs in a spawned interpreter.
    import torch.distributed as dist  # noqa: PLC0415
    from torch.distributed.device_mesh import init_device_mesh  # noqa: PLC0415
    from torch.distributed.tensor import Shard, distribute_tensor  # noqa: PLC0415

    DistributedMuon = _load_muon().DistributedMuon

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(WORLD)
    )
    dist.init_process_group("gloo")
    try:
        mesh = init_device_mesh("cpu", (WORLD,))
        full = torch.arange(ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS)
        param = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
        param.grad = distribute_tensor(torch.ones_like(full), mesh, [Shard(0)])

        before = param.to_local().clone()
        opt = DistributedMuon([("hc_head.hc_fn", param)], lr=0.1)
        opt.step()  # hangs here on the regression
        after = param.to_local()

        out[rank] = (
            tuple(before.shape),
            bool(before.shape[0] == 0 or not torch.equal(before, after)),
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.regression
def test_muon_gathers_an_unevenly_sharded_matrix():
    import torch.multiprocessing as mp  # noqa: PLC0415 - see importorskip above

    mgr = mp.Manager()
    out = mgr.dict()
    mp.spawn(_worker, args=("29607", out), nprocs=WORLD, join=True)

    assert len(out) == WORLD, f"only {sorted(out)} finished -- the gather deadlocked"
    shapes = {r: out[r][0] for r in range(WORLD)}
    assert shapes == {0: (1, COLS), 1: (1, COLS), 2: (0, COLS), 3: (0, COLS)}, (
        f"the test no longer exercises an UNEVEN split ({shapes}); pick a global row "
        "count that does not divide by the mesh width or it proves nothing"
    )
    for r in range(WORLD):
        assert out[r][1], f"rank {r} came back with the parameter unchanged"
