import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from speculators.model import SpeculatorModel
from speculators.train.checkpointer import (
    BaseCheckpointer,
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.distributed import (
    apply_fully_sharded,
    get_local_rank,
    get_rank,
    is_distributed,
    shard_experts_as_dtensor,
)
from speculators.train.graceful_shutdown import with_graceful_shutdown
from speculators.train.optimizers import build_optimizers
from speculators.train.utils import normalize_counted_metrics

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")


class _StepTimer:
    # Each mark()/now() forces an accelerator.synchronize to capture true GPU time.
    # This serialises the CUDA pipeline, so profiled steps are slower; keep
    # log_freq > 1 in perf-sensitive runs.
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._marks: dict[str, float] = {}

    def reset(self, enabled: bool) -> None:
        self.enabled = enabled
        self._marks.clear()

    def mark(self, name: str) -> None:
        if self.enabled:
            torch.accelerator.synchronize()
            self._marks[name] = time.perf_counter()

    def mark_value(self, name: str, value: float) -> None:
        if self.enabled:
            self._marks[name] = value

    def now(self) -> float | None:
        if not self.enabled:
            return None
        torch.accelerator.synchronize()
        return time.perf_counter()

    def profile(self, num_tokens: int) -> dict[str, float] | None:
        if not self.enabled:
            return None
        m = self._marks
        has_start = "start" in m
        fwd_ms = (m["fwd"] - m["fetch"]) * 1000
        bwd_ms = (m["bwd"] - m["fwd"]) * 1000
        opt_ms = (m["opt"] - m["bwd"]) * 1000
        fetch_ms = (m["fetch"] - m["start"]) * 1000 if has_start else 0.0
        step_ms = (m["opt"] - m["start"]) * 1000 if has_start else 0.0
        tokens_per_s = num_tokens / (step_ms / 1000) if step_ms > 0 else 0.0
        fetch_frac = fetch_ms / step_ms if step_ms > 0 else 0.0
        return {
            "fetch_ms": fetch_ms,
            "fwd_ms": fwd_ms,
            "bwd_ms": bwd_ms,
            "opt_ms": opt_ms,
            "step_ms": step_ms,
            "tokens_per_s": tokens_per_s,
            "fetch_frac": fetch_frac,
        }


_MEM_WARNED: dict[str, str] = {}


def _gpu_mem_stats() -> dict[str, float] | None:
    """Per-device memory in GB: current + cumulative peak.

    ``max_*`` are cumulative (never reset here), so the last logged step of a run
    carries the whole-run peak -- the number that answers "does it fit". Returns
    None only when there is genuinely no accelerator.

    ⚠ This used to gate on ``torch.cuda.is_available()``. On Ascend that is FALSE
    even though every function below works, so ``mem`` logged ``None`` on every
    step of every NPU run -- all 124,480 steps of the production log carry no
    memory number, leaving only ``npu-smi``, which shows the driver's
    never-shrinking reserved pool and cannot see an optimizer-state delta. The
    cause is in ``scripts/train.py``: when ``torch_npu``'s ``transfer_to_npu``
    raises mid-init, the recovery path re-runs ``_apply_patches()`` with the patch
    list DROPPED, then hand-shims exactly seven ``torch.cuda.*`` functions -- the
    four used below among them, but NOT ``is_available``. So the guard answered no
    while the body would have answered fine.

    Ask torch's device-agnostic API instead, and read the stats off the live
    accelerator's own module (``torch.npu`` / ``torch.cuda``), which is right
    whether or not a shim ran. Same idiom as ``npu_bridge.npu_available``.
    """
    gb = 1024**3
    try:
        acc = torch.accelerator.current_accelerator()
        if acc is None:
            reason = "no accelerator (CPU-only stack)"
        else:
            mod = getattr(torch, acc.type, None)
            if mod is None or not hasattr(mod, "memory_allocated"):
                reason = f"torch.{acc.type} exposes no memory API"
            else:
                return {
                    "alloc_gb": round(mod.memory_allocated() / gb, 2),
                    "reserved_gb": round(mod.memory_reserved() / gb, 2),
                    "max_alloc_gb": round(mod.max_memory_allocated() / gb, 2),
                    "max_reserved_gb": round(mod.max_memory_reserved() / gb, 2),
                }
    except Exception as exc:  # noqa: BLE001 - probe must never break training
        reason = f"{type(exc).__name__}: {exc}"

    # Say WHY, once. A silent None is what let a whole 5-epoch run finish with no
    # memory record; one warning at step 0 would have cost nothing.
    if _MEM_WARNED.get("reason") != reason:
        _MEM_WARNED["reason"] = reason
        root_logger.warning("memory metrics unavailable: %s", reason)
    return None



# DSPARK_TRACE=1: a per-rank, immediately-flushed phase marker.
#
# WHY. The metric log is rank-0 only and emits once per STEP, so a hang that happens
# before the first step completes logs NOTHING -- and says nothing about which rank
# reached which phase. Three hangs in a row were reported at three different Python
# frames (timer.mark, an optimizer collective, metrics .item()) because a traceback
# names whichever collective was issued LAST, not the one that stalled.
#
# This writes from EVERY rank, unbuffered, with a timestamp. The last line each rank
# printed is exactly how far that rank got, so a stalled collective becomes a diff
# between rank groups instead of a guess. Off by default, zero cost when off.
_TRACE_SYNC = os.environ.get("DSPARK_TRACE_SYNC") == "1"
_TRACE = os.environ.get("DSPARK_TRACE") == "1" or _TRACE_SYNC


def _collective_seq() -> str:
    """How many collectives this rank has issued on the default process group.

    THE POINT. A phase marker says how far a rank got in PYTHON; it does not say how
    many collectives that rank put on the wire. HCCL matches ops by position in the
    stream, so if two ranks ever disagree on the COUNT, every later collective is
    matched to the wrong partner -- an all-gather on one rank meeting a reduce on
    another -- and the run hangs with both ranks apparently "in the same place".

    That is exactly the observed signature: ranks {4,5,6,7} finished Muon's last
    parameter and queued the metrics reduces (``seq_num 180-211``, ``HcclReduce``)
    while ranks {0,1,2,3} sat in an all-gather for that same parameter.

    Printing the counter next to every phase turns that into a diff: walk the two rank
    groups down the trace and the FIRST phase where their seq deltas differ is the code
    that issues a rank-dependent number of collectives. Pure getter, no collective, no
    sync -- safe to call on every line.
    """
    try:
        pg = dist.distributed_c10d._get_default_group()  # noqa: SLF001
        return str(pg._get_sequence_number_for_group())  # noqa: SLF001
    except Exception:  # noqa: BLE001 - a debug print must never take the run down
        return "?"


def _trace(phase: str, step: object = None) -> None:
    if not _TRACE:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    now = time.time()
    ts = time.strftime("%H:%M:%S", time.localtime(now))
    ms = int(now * 1000) % 1000
    print(  # noqa: T201 - the whole point is an unbuffered per-rank marker
        f"[TRACE {ts}.{ms:03d} rank={rank} step={step} "
        f"seq={_collective_seq()}] {phase}",
        file=sys.stderr,
        flush=True,
    )


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
MIN_STEP_PCT = 0.25

# Re-synchronise ranks every N validation batches to bound cross-rank skew, which would
# otherwise blow the NCCL watchdog at the end-of-epoch metrics all-reduce. 0 disables.
_VAL_SYNC_INTERVAL = 50


class TrainerConfig(NamedTuple):
    lr: float
    num_epochs: int
    save_path: str
    resume_from_checkpoint: bool = False
    train_call_kwargs: dict | None = None
    val_call_kwargs: dict | None = None
    optimizer: Literal["adamw", "muon"] = "adamw"
    weight_decay: float = 0.01
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"
    # Off by default, matching upstream torchtitan-npu: it measures closer to
    # orthogonal at no extra cost, but orthogonality is not the training objective.
    # ⚠ Adding a Muon knob means touching THREE places, not one -- this NamedTuple is
    # what build_optimizers reads, scripts/train.py has its own hand-written argparse,
    # and train/config/schema.py is a separate entry point that this launcher never
    # uses. Changing only the schema gets you "unrecognized arguments" at rank 0.
    muon_hybrid_ns: bool = False
    scheduler_type: Literal["linear", "cosine", "none"] = "linear"
    scheduler_warmup_steps: int | None = None
    scheduler_warmup_ratio: float | None = None
    scheduler_total_steps: int | None = None
    scheduler_num_cosine_cycles: float = 0.5
    checkpoint_freq: float = 1
    save_best: bool = False
    hidden_states_dtype: torch.dtype = torch.bfloat16
    # AMP master weights for the EP routed experts. Default False = option A = upstream #711
    # semantics: EVERY trainable param (incl experts) gets a fp32 master (all-trainable upcast).
    # True = option B (fork memory path): experts stay bf16 (no fp32 master). Only meaningful with
    # EP experts under FSDP2; needed for the EP-off faithful path where rank0 materialises the full
    # unsharded model (fp32 experts would OOM there). Under EP=1 (the training norm) A fits fine.
    bf16_experts: bool = False
    log_freq: int = 1
    fsdp_shard: bool = False
    max_steps: int | None = None


def _resolve_scheduler_steps(
    config: TrainerConfig,
    train_loader_len: int,
) -> tuple[int, int]:
    """Resolve ``(warmup_steps, total_steps)`` for the LR scheduler.

    Explicit ``scheduler_warmup_steps`` wins; otherwise ``scheduler_warmup_ratio``
    (a fraction of total steps, validated to ``[0, 1]``) is used; otherwise the
    default of 1% of the resolved total steps. ``scheduler_total_steps`` defaults
    to ``num_epochs * train_loader_len``.
    """
    default_total_steps = config.num_epochs * train_loader_len
    scheduler_total_steps = (
        config.scheduler_total_steps
        if config.scheduler_total_steps is not None
        else default_total_steps
    )

    if config.scheduler_warmup_steps is not None:
        scheduler_warmup_steps = config.scheduler_warmup_steps
        if config.scheduler_warmup_ratio is not None:
            warnings.warn(
                "Both scheduler_warmup_steps and scheduler_warmup_ratio are set; "
                "using scheduler_warmup_steps.",
                stacklevel=2,
            )
    elif config.scheduler_warmup_ratio is not None:
        if not 0 <= config.scheduler_warmup_ratio <= 1:
            raise ValueError("scheduler_warmup_ratio must be between 0 and 1.")
        scheduler_warmup_steps = int(
            scheduler_total_steps * config.scheduler_warmup_ratio
        )
    else:
        scheduler_warmup_steps = scheduler_total_steps // 100

    return scheduler_warmup_steps, scheduler_total_steps


class Trainer:
    def __init__(
        self,
        model: SpeculatorModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.model = model
        self.config = config
        self.local_rank = get_local_rank()
        self.rank = get_rank()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_distributed = is_distributed()
        self.resume_from_checkpoint = config.resume_from_checkpoint
        checkpointer_class: type[BaseCheckpointer] = (
            DistributedCheckpointer if self.is_distributed else SingleGPUCheckpointer
        )
        self.checkpointer: BaseCheckpointer = checkpointer_class(self.config.save_path)

        self.setup_trainer()
        self.setup_model()
        self.setup_optimizer()

    def _training_state_path(self, epoch: int) -> Path:
        return self.checkpointer.path / str(epoch) / "training_state.json"

    def _save_training_state(self, epoch: int, local_step: int) -> None:
        if not self.is_distributed or dist.get_rank() == 0:
            state = {
                "epoch": epoch,
                "local_step": local_step,
                "global_step": self.global_step,
            }
            p = self._training_state_path(epoch)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(state))

    def _load_training_state(self) -> dict:
        epoch = self.checkpointer.previous_epoch
        p = self._training_state_path(epoch)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError as e:
                root_logger.warning(f"Failed to decode training state {p}: {e}")
            except (FileNotFoundError, PermissionError, OSError) as e:
                root_logger.warning(f"Failed to read training state {p}: {e}")
        return {}

    def setup_trainer(self):
        if self.checkpointer.previous_epoch != -1:
            root_logger.info(f"Found checkpoint at {self.checkpointer.prev_path}.")
            self.current_epoch = self.checkpointer.previous_epoch + 1
            if self.resume_from_checkpoint:
                # Check if this was a mid-epoch checkpoint — if so, resume
                # from within that epoch rather than jumping to the next one.
                state = self._load_training_state()
                is_mid_epoch = (
                    state
                    and state.get("epoch") == self.checkpointer.previous_epoch
                    and state.get("local_step", 0) > 0  # 0 means end-of-epoch
                )
                if is_mid_epoch:
                    # Resume within the same epoch from the exact step.
                    self.current_epoch = state["epoch"]
                    self._resume_local_step = state["local_step"]
                    self._resume_global_step = state.get("global_step", 0)
                    root_logger.info(
                        f"Resuming mid-epoch from epoch={self.current_epoch} "
                        f"local_step={self._resume_local_step} "
                        f"global_step={self._resume_global_step}."
                    )
                else:
                    # End-of-epoch or no state — advance to next epoch.
                    self._resume_local_step = 0
                    resume_global = state.get("global_step", 0) if state else 0
                    self._resume_global_step = resume_global
                    root_logger.info(
                        f"Resuming training on epoch {self.current_epoch}."
                    )
            else:
                root_logger.warning(
                    "`resume_from_checkpoint` is False, starting "
                    "training from scratch. This will overwrite the "
                    f"existing checkpoints in {self.checkpointer.path}."
                )
                self.current_epoch = 0
                self._resume_local_step = 0
                self._resume_global_step = 0
        else:
            root_logger.info(
                "No previous training checkpoint found in "
                f"'{self.checkpointer.path}'. Starting fresh training run."
            )
            self.current_epoch = 0
            self._resume_local_step = 0
            self._resume_global_step = 0
        self.global_step = self._resume_global_step
        self.best_val_loss = float("inf")

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            saved = self.checkpointer.load_best_val_loss()
            if saved is not None:
                self.best_val_loss = saved
                root_logger.info(
                    f"Restored best_val_loss={self.best_val_loss:.6f} from checkpoint"
                )

    def setup_model(self):
        # Verify model is compatible with training infrastructure
        SpeculatorModel.verify_training_compatible(self.model)

        load_checkpoint = (
            self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1
        )

        if not self.is_distributed:
            # Single device case: no FSDP mixed-precision policy, so cast the whole
            # model to the compute dtype directly.
            self.model.to(self.config.hidden_states_dtype)  # type: ignore[arg-type]
            self.model.to(self.local_rank)  # type: ignore[arg-type]
            if load_checkpoint:
                self.checkpointer.load_model_state_dict(self.model)
            return

        # Distributed (FSDP) AMP master weights. Cast the whole model to the compute
        # dtype FIRST (this gives buffers — incl. the complex64 RoPE ``freqs_cis`` — the
        # exact treatment the old path relied on; a selective loop that skips complex
        # buffers via ``is_floating_point()`` leaves ``freqs_cis`` complex64 and NPU
        # ``aclnnIndex`` crashes on it). THEN upcast the SMALL trainable params
        # (norm/MLA/mHC/heads) back to fp32 as MASTER weights, so their sub-ULP updates
        # (norm ~1e-7) + AdamW weight decay survive bf16 — FSDP2's MixedPrecisionPolicy
        # keeps the pre-shard dtype, so a bf16-only param has no master and gets silently
        # frozen. By DEFAULT (option A = upstream #711) EVERY trainable param — experts
        # included — is upcast, so it gets a fp32 master; this is safe under EP=1 (each rank
        # only holds/upcasts its own 1/EP expert slice, no rank0 full-fp32 materialisation).
        # ``bf16_experts=True`` (option B) keeps the EP experts bf16 — the memory path for the
        # EP-OFF faithful run, where rank0 materialises the FULL unsharded model and fp32
        # experts (~15B) would OOM. Their large grads are far less rounding-sensitive, so the
        # bf16 compromise is numerically cheap. The one-time upcast precision loss is negligible
        # (warm-start source is bf16); what matters is the fp32 ACCUMULATION across steps.
        self.model.to(self.config.hidden_states_dtype)  # type: ignore[arg-type]
        _ep_local = getattr(self.model, "ep_local_param_keys", None)
        _expert_keys = set(_ep_local()) if callable(_ep_local) else set()
        for _name, _p in self.model.named_parameters():
            if _p.requires_grad and (
                not self.config.bf16_experts or _name not in _expert_keys
            ):
                _p.data = _p.data.to(torch.float32)

        # Distributed case
        # Capture full state dict on rank 0 before FSDP sharding
        full_state_dict = {}
        if not load_checkpoint and dist.get_rank() == 0:
            full_state_dict = self.model.state_dict()
            # EP: routed experts are rank-local Shard(0) slices (each rank owns a disjoint,
            # per-rank-initialized set) -> drop them from the rank0 broadcast, since rank0
            # doesn't hold the global expert set. Non-expert params still broadcast.
            ep_local = getattr(self.model, "ep_local_param_keys", None)
            if callable(ep_local):
                for k in ep_local():
                    full_state_dict.pop(k, None)

        # EP: build the expert-parallel DeviceMesh, move the model to the device, and wrap
        # each GroupedExperts' stacked weights as Shard(0) DTensors on that mesh BEFORE FSDP.
        # Experts and the FSDP-sharded rest are then uniform DTensors on ONE mesh -> the
        # optimizer / clip / DCP checkpoint need no plain-vs-DTensor special casing.
        ep_active = callable(getattr(self.model, "ep_local_param_keys", None)) and bool(
            self.model.ep_local_param_keys()
        )
        mesh = None
        if ep_active:
            self.model.to(self.local_rank)  # type: ignore[arg-type]
            dev = next(self.model.parameters()).device.type
            mesh = init_device_mesh(dev, (dist.get_world_size(),))
            shard_experts_as_dtensor(self.model, mesh)

        apply_fully_sharded(self.model, mesh=mesh)

        if load_checkpoint:
            self.checkpointer.load_model_state_dict(self.model)
        else:
            # Broadcast full state dict from rank 0 to all ranks
            set_model_state_dict(
                self.model,
                full_state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=False,
                ),
            )
            del full_state_dict
            dist.barrier()

    def _setup_model_ddp(self, load_checkpoint: bool):
        self.model.to(self.local_rank)  # type: ignore[arg-type]

        if load_checkpoint:
            if dist.get_rank() == 0:
                self.checkpointer.load_model_state_dict(self.model)
        else:
            # Fresh init: broadcast rank 0's random initialization to all ranks
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
            dist.barrier()

        # DDP constructor broadcasts rank 0's params to all ranks
        self.model = DistributedDataParallel(self.model)  # type: ignore[assignment]

    def setup_optimizer(self):
        # Setup optimizer(s). The "muon" option returns two optimizers (Muon for the
        # 2D weight matrices, AdamW for everything else); "adamw" returns a single one.
        self.optimizers = build_optimizers(self.model, self.config)
        last_epoch = -1
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_optimizer_state_dict(self.model, self.optimizers)
            last_epoch = self.checkpointer.previous_epoch

        # Setup scheduler(s) — one per optimizer so each optimizer's base LR (e.g.
        # Muon's higher LR vs AdamW's) is warmed up / decayed independently.
        if self.config.scheduler_type == "none":
            self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
            return

        scheduler_warmup_steps, scheduler_total_steps = _resolve_scheduler_steps(
            self.config, len(self.train_loader)
        )

        def make_scheduler(opt: torch.optim.Optimizer):
            if self.config.scheduler_type == "linear":
                return get_linear_schedule_with_warmup(
                    opt,
                    num_warmup_steps=scheduler_warmup_steps,
                    num_training_steps=scheduler_total_steps,
                    last_epoch=last_epoch,
                )
            return get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=scheduler_total_steps,
                num_cycles=self.config.scheduler_num_cosine_cycles,
                last_epoch=last_epoch,
            )

        self.schedulers = [make_scheduler(opt) for opt in self.optimizers]

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_scheduler_state_dict(self.schedulers)

    def _optimizers_zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def _optimizers_step(self):
        # Traced PER OPTIMIZER, not as one block. In muon mode this list is
        # [DistributedMuon, AdamW], and a trace that lumps them together cannot say
        # which one a stuck rank is in -- the run that reached "all params done" on
        # every rank and still never completed a step is exactly that ambiguity.
        for opt in self.optimizers:
            name = type(opt).__name__
            _trace(f"{name}.step -> enter", self.global_step)
            opt.step()
            if _TRACE_SYNC:
                # Collectives are queued, not awaited: without this the line below
                # proves only that the call returned, not that its communication
                # completed. Debugging only.
                torch.accelerator.synchronize()
                _trace(f"{name}.step <- DRAINED", self.global_step)
            else:
                _trace(f"{name}.step <- returned", self.global_step)

    def _schedulers_step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def _prepare_resume_skip(self, epoch: int) -> int:
        """Prepare fast-skip state for mid-epoch resume and return skipped steps."""
        skip_steps = 0
        if epoch == getattr(self, "current_epoch", epoch):
            skip_steps = getattr(self, "_resume_local_step", 0)
            # Only skip once — clear after use.
            self._resume_local_step = 0

        # Fast-skip: slice the sampler's pre-generated batch list so we never
        # call __getitem__ (and thus never call vLLM) for skipped batches.
        sampler = self.train_loader.batch_sampler
        has_fast_skip_api = hasattr(sampler, "_generate_batches") and hasattr(
            sampler, "_cached_generated_batches"
        )
        if skip_steps > 0 and has_fast_skip_api:
            all_batches = sampler._generate_batches(epoch)  # type: ignore[union-attr]  # noqa: SLF001
            remaining = all_batches[skip_steps:]
            # Temporarily override the sampler cache with the sliced list.
            sampler._cached_generated_batches = (  # type: ignore[union-attr]  # noqa: SLF001
                epoch,
                remaining,
            )
            root_logger.info(
                f"Fast-skipping {skip_steps} batches via sampler slice "
                f"(no vLLM calls for skipped batches). "
                f"epoch={epoch}, global_step={self.global_step}."
            )
        elif skip_steps > 0:
            root_logger.warning(
                "Sampler lacks fast-skip API; resume will replay "
                f"{skip_steps} batches from the start of the epoch."
            )
        return skip_steps

    def train_epoch(self, epoch: int):
        self.model.train()
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]

        # Capture full-epoch step count before any resume fast-skip mutation.
        num_steps = len(self.train_loader)

        # Determine how many batches to skip for mid-epoch resume.
        skip_steps = self._prepare_resume_skip(epoch)

        train_loader = self.train_loader
        if self.rank == 0:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        step_interval = (
            max(1, round(num_steps * self.config.checkpoint_freq))
            if self.config.checkpoint_freq < 1
            else None
        )
        t_before_fetch = time.perf_counter()
        timer = _StepTimer()
        for local_step_rel, batch in enumerate(train_loader, 1):
            # local_step is 1-based index into the *full* epoch (not the slice).
            local_step = local_step_rel + skip_steps
            timer.reset(self.global_step % self.config.log_freq == 0)
            _trace("step-begin", self.global_step)

            timer.mark_value("start", t_before_fetch)
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            timer.mark("fetch")
            # --- DIAGNOSTIC (profiled steps only): per-rank fetch_ms + straggler-align wait --------
            # fwd_ms was mis-reading the EP-MoE all-to-all's wait-for-straggler as "forward compute":
            # a rank whose HS arrives late from the serve makes the FAST ranks block at the all-to-all,
            # and that wait lands in THEIR fwd_ms while their OWN fetch_ms stays clean (rank0 = 24ms),
            # so the analyzer mis-labels a serve/HS stall as a "recompile" fwd spike. all_gather each
            # rank's fetch_ms here -- the gather doubles as an ALIGN BARRIER, so the straggler-wait is
            # absorbed into align_ms (measured on this rank) instead of the forward below, and
            # fetch_ms_ranks exposes WHICH rank stalled. If the spikes move fwd->align, it's HS-bound.
            # The SUPERVISED-TOKEN COUNT rides this same all_gather (no extra collective, log
            # steps only). Per-rank loss normalization weights rank r's tokens by 1/(R*n_r)
            # instead of the token-weighted 1/N, so mean(n)/n_r is exactly how far off that
            # rank's weight is -- 1.0 everywhere means the two objectives coincide and
            # DSPARK_GLOBAL_LOSS_REDUCE is a no-op. See models/metrics.py.
            if _TRACE:
                # WHY a rank ends up with no gradient is a fact about ITS batch, not
                # about the optimizer, so record the batch before anything consumes
                # it. Shapes of every tensor plus the supervised-token count: a rank
                # whose loss_mask sums to 0 contributes nothing to the loss, and the
                # parameters that only that path touches then come back from backward
                # with `grad is None`.
                # `sup_tokens_zero_ranks` in the profile below already tracks that this
                # happens here; this says WHICH rank and WHEN, on every step.
                _shapes = " ".join(
                    f"{k}={tuple(v.shape)}"
                    for k, v in gpu_batch.items()
                    if isinstance(v, torch.Tensor)
                )
                _lm0 = gpu_batch.get("loss_mask")
                _sup = "-" if _lm0 is None else int(_lm0.sum().item())
                _trace(f"batch: sup_tokens={_sup} {_shapes}", self.global_step)
            _trace("batch+HS ready", self.global_step)
            fetch_all: list[float] | None = None
            tokens_all: list[int] | None = None
            align_ms = 0.0
            if timer.enabled and self.is_distributed:
                _fetch_local = (timer._marks["fetch"] - timer._marks["start"]) * 1000
                _lm = gpu_batch.get("loss_mask")
                _t_align = time.perf_counter()
                _buf = torch.tensor([_fetch_local, 0.0], device=self.local_rank, dtype=torch.float32)
                if _lm is not None:
                    _buf[1] = _lm.sum().to(torch.float32)
                _out = [torch.zeros_like(_buf) for _ in range(dist.get_world_size())]
                _trace("align all_gather -> enter", self.global_step)
                dist.all_gather(_out, _buf)
                _trace("align all_gather <- done", self.global_step)
                align_ms = (time.perf_counter() - _t_align) * 1000
                fetch_all = [round(float(t[0].item()), 1) for t in _out]
                tokens_all = [int(t[1].item()) for t in _out]
            # --------------------------------------------------------------------------------------
            _draft_tokens, loss, metrics = self.model(
                **gpu_batch, **(self.config.train_call_kwargs or {})
            )

            _trace("forward done", self.global_step)
            timer.mark("fwd")
            self._optimizers_zero_grad()
            loss.backward()
            # pre-clip total grad norm (FSDP2-global reduce) -- watch it climb toward
            # the lr peak (lr-driven blow-up) vs spike on one batch (dirty data); a
            # non-finite value pinpoints the step where NaN enters via the gradients.
            # (Under EP the routed experts are Shard(0) DTensors on the same mesh as the
            # FSDP-sharded rest, so a single clip over all params is uniform -- no split.)
            if _TRACE:
                # THE CENSUS. Which parameters came out of backward with no gradient
                # on THIS rank, and in what dtype the ones that did. Both are
                # rank-local, and both feed the same failure: a parameter missing on
                # some ranks and present on others made the old optimizer issue a
                # different NUMBER of collectives (1b62bd8c guarded `_step_param`
                # behind `if param.grad is not None`), and makes the current one issue
                # a different SIZE. Model-wide rather than Muon-only, so it still says
                # something if the divergence is in a parameter Muon never touches.
                _none, _dts = [], {}
                for _n, _p in self.model.named_parameters():
                    if not _p.requires_grad:
                        continue
                    if _p.grad is None:
                        _none.append(_n)
                    else:
                        _k = str(_p.grad.dtype).removeprefix("torch.")
                        _dts[_k] = _dts.get(_k, 0) + 1
                _trace(
                    f"grad census: none={len(_none)} have={sum(_dts.values())} "
                    f"dtypes={_dts} {_none[:16]}",
                    self.global_step,
                )
            _trace("backward done -> clip_grad_norm (collective)", self.global_step)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Per-rank, not just rank 0 on log steps: a norm that is 0 or non-finite on
            # one rank group is itself the answer to "what was different about them".
            _trace(f"clip_grad_norm done norm={float(grad_norm):.6g}", self.global_step)

            timer.mark("bwd")
            _trace("optimizer.step -> enter", self.global_step)
            self._optimizers_step()
            _trace("optimizer.step <- done", self.global_step)

            current_lrs = {
                type(opt).__name__: opt.param_groups[0]["lr"] for opt in self.optimizers
            }
            self._schedulers_step()
            _trace("schedulers done -> timer.mark(opt) sync", self.global_step)
            timer.mark("opt")
            _trace("timer.mark(opt) done", self.global_step)
            t_before_fetch = timer.now() or time.perf_counter()

            profile = None
            if timer.enabled:
                num_tokens = int((gpu_batch["document_ids"] != -1).sum().item())
                profile = timer.profile(num_tokens)
                if profile is not None:
                    profile["grad_norm"] = float(grad_norm)
                    if fetch_all is not None:
                        # per-rank fetch + the straggler wait that was hiding in fwd_ms
                        profile["fetch_ms_ranks"] = fetch_all
                        profile["fetch_ms_max"] = max(fetch_all)
                        profile["align_ms"] = round(align_ms, 1)
                    if tokens_all and sum(tokens_all) > 0:
                        _mean = sum(tokens_all) / len(tokens_all)
                        profile["sup_tokens_ranks"] = tokens_all
                        # Skew over ranks that HAVE tokens. A zero-token rank is reported
                        # separately, not folded in: mean/0 is not a weight ratio, and letting
                        # it through as mean/1 fabricates a huge "skew" out of a divide guard.
                        _nz = [t for t in tokens_all if t > 0]
                        if _nz:
                            _skew = [_mean / t for t in _nz]
                            # how much the per-rank objective over/under-weights a rank's tokens
                            profile["sup_tokens_skew_max"] = round(max(_skew), 4)
                            profile["sup_tokens_skew_min"] = round(min(_skew), 4)
                        if len(_nz) < len(tokens_all):
                            # the extreme case of the same bug: a rank with no supervised tokens
                            # contributes nothing to the loss yet still takes 1/R of the
                            # per-rank mean-of-ratios.
                            profile["sup_tokens_zero_ranks"] = len(tokens_all) - len(_nz)
                        # spread as a fraction of the mean; 0.0 = perfectly balanced
                        profile["sup_tokens_spread"] = round(
                            (max(tokens_all) - min(tokens_all)) / _mean, 4
                        )
                if self.is_distributed:
                    # One reduce PER KEY, so the key set must match across ranks. It is
                    # built by the model's forward, which makes the collective count
                    # data-dependent in principle -- name the keys once so a mismatch is
                    # readable rather than a hang.
                    _trace(
                        f"metrics reduce -> enter n={len(metrics)} "
                        f"keys={sorted(metrics)}",
                        self.global_step,
                    )
                    for v in metrics.values():
                        dist.reduce(v, dst=0, op=dist.ReduceOp.SUM)
                    _trace("metrics reduce <- done", self.global_step)

                metrics = {k: v.item() for k, v in metrics.items()}
                world_size = dist.get_world_size() if self.is_distributed else 1
                metrics = normalize_counted_metrics(metrics, world_size)
                lr_info = (
                    current_lrs
                    if len(current_lrs) > 1
                    else next(iter(current_lrs.values()))
                )
                metric_logger.info(
                    {
                        "train": metrics,
                        "profile": profile,
                        "mem": _gpu_mem_stats(),
                        "epoch": epoch,
                        "lr": lr_info,
                        "global_step": self.global_step,
                    },
                    extra={"step": self.global_step},
                )
            self.global_step += 1

            if (
                self.config.max_steps is not None
                and self.global_step >= self.config.max_steps
            ):
                break

            if (
                step_interval is not None
                and not self.config.save_best
                and local_step % step_interval == 0
                and num_steps - local_step >= step_interval * MIN_STEP_PCT
                # Avoid saving back to back ay the end of each epoch
            ):
                self.maybe_save_checkpoint(epoch, local_step=local_step)

    def _maybe_val_sync(self, batch_index: int) -> None:
        if not self.is_distributed or _VAL_SYNC_INTERVAL <= 0:
            return
        if batch_index > 0 and batch_index % _VAL_SYNC_INTERVAL == 0:
            dist.barrier()

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader = self.val_loader
        if self.rank == 0:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        accumulated: dict[str, torch.Tensor] = {}
        num_batches = len(val_loader)
        for i, batch in enumerate(val_loader):
            self._maybe_val_sync(i)
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            _draft_tokens, _loss, metrics = self.model(
                **gpu_batch, **(self.config.val_call_kwargs or {})
            )

            for k, v in metrics.items():
                acc = accumulated.get(k)
                accumulated[k] = v.float() if acc is None else acc + v.float()

        val_metrics: dict[str, float] = {}
        if accumulated:
            stacked = torch.stack(list(accumulated.values()))
            if self.is_distributed:
                dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
            val_metrics = dict(zip(accumulated, stacked.tolist(), strict=True))

        world_size = dist.get_world_size() if self.is_distributed else 1
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        val_metrics = normalize_counted_metrics(val_metrics, world_size)
        val_metrics = {f"{k}_epoch": v for k, v in val_metrics.items()}

        metric_logger.info(
            {"val": val_metrics, "epoch": epoch}, extra={"step": self.global_step}
        )

        return val_metrics

    def maybe_save_checkpoint(self, epoch: int | str, local_step: int = 0):
        if epoch != "interrupted" and (
            self.config.save_best
            or (
                self.config.checkpoint_freq >= 1
                and isinstance(epoch, int)
                and epoch != 0
                and (epoch + 1) % self.config.checkpoint_freq != 0
            )
        ):
            return

        root_logger.info(f"Saving checkpoint to {self.checkpointer.path / str(epoch)}")
        self.checkpointer.save_checkpoint(self.model, self.optimizers, epoch)
        if self.schedulers:
            self.checkpointer.save_scheduler_state_dict(self.schedulers, epoch)
        if isinstance(epoch, int):
            self._save_training_state(epoch, local_step)
            # Create a human-readable symlink for checkpoint readability.
            # e.g. epoch0_step16626 -> 0/ (mid) or epoch0_end -> 0/ (end)
            if not self.is_distributed or dist.get_rank() == 0:
                ckpt_dir = self.checkpointer.path
                suffix = f"step{local_step}" if local_step > 0 else "end"
                link_name = ckpt_dir / f"epoch{epoch}_{suffix}"
                target = Path(str(epoch))  # relative symlink
                # Remove any previous link for this epoch
                for old in ckpt_dir.glob(f"epoch{epoch}_*"):
                    if old.is_symlink():
                        old.unlink()
                link_name.symlink_to(target)
        root_logger.info(f"Checkpoint saved to {self.checkpointer.path / str(epoch)}")

    def maybe_update_best(self, epoch: int, val_metrics: dict | None):
        if val_metrics is None or "loss_epoch" not in val_metrics:
            return
        if val_metrics["loss_epoch"] >= self.best_val_loss:
            return

        if self.config.save_best:
            self.checkpointer.save_checkpoint(self.model, self.optimizers, epoch)
            if self.schedulers:
                self.checkpointer.save_scheduler_state_dict(self.schedulers, epoch)
        elif self.config.checkpoint_freq >= 1 and not (
            epoch == 0 or (epoch + 1) % int(self.config.checkpoint_freq) == 0
        ):
            return

        self.best_val_loss = val_metrics["loss_epoch"]
        self.checkpointer.save_val_metrics(epoch, val_metrics)
        self.checkpointer.update_best_symlink(epoch)
        root_logger.info(
            f"Updated checkpoint_best -> {epoch} (loss_epoch={self.best_val_loss:.6f})"
        )
        if self.config.save_best:
            self.checkpointer.cleanup_keep_only_best(best_epoch=epoch)

    @with_graceful_shutdown()
    def run_training(self):
        n_epochs = self.config.num_epochs
        for epoch in range(self.current_epoch, n_epochs):
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} started")
            self.train_epoch(epoch)
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            self.maybe_save_checkpoint(epoch)

            if self.is_distributed:
                dist.barrier()

            val_metrics = None

            if self.val_loader is None:
                root_logger.warning("No val loader, skipping validation epoch")
            else:
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} started")
                val_metrics = self.val_epoch(epoch)
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            self.maybe_update_best(epoch, val_metrics)

            if self.is_distributed:
                dist.barrier()
